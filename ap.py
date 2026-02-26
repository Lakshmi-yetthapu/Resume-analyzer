import streamlit as st
from pathlib import Path
import re
from PIL import Image
import pytesseract
import pandas as pd
import requests
from io import BytesIO, StringIO
import csv
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import time
import fitz  # PyMuPDF
import tempfile
import os
import logging
import json
from datetime import datetime

# Imports from the new PDF processing module
import pdfplumber
from urllib.parse import urlparse, urlunparse
import urllib.request
from typing import Tuple, List, Any, Dict, Optional, Set
from random import uniform

import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ==============================================================================
#  CONFIGURATION
# ==============================================================================

# NEW: DYNAMICALLY READ UP TO 10 API KEYS
# This loop checks for MISTRAL_API_KEY_1, MISTRAL_API_KEY_2, ... up to 10.
MISTRAL_API_KEYS = []

# Try to load keys from Streamlit secrets or Environment Variables
for i in range(1, 13):
    key_name = f"MISTRAL_API_KEY_{i}"
    try:
        # Check secrets.toml first
        key = st.secrets.get(key_name)
        # If not in secrets, check OS environment
        if not key:
            key = os.getenv(key_name)
        
        if key:
            MISTRAL_API_KEYS.append(key)
    except (AttributeError, KeyError):
        pass

# If no numbered keys found, try looking for a single unnumbered key
if not MISTRAL_API_KEYS:
    single_key = os.getenv("MISTRAL_API_KEY")
    if single_key:
        MISTRAL_API_KEYS.append(single_key)

MISTRAL_MODEL = "mistral-medium-latest"
MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ##############################################################################
#  GOOGLE SHEETS CONFIGURATION
# ##############################################################################

# Replace with the name of the Google Sheet you created and shared.
GSHEET_NAME = "AI Resume Analysis Results"

@st.cache_resource
def get_gspread_client():
    """Initializes and returns a gspread client, caching it for performance."""
    try:
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        # Load credentials safely from secrets
        if "gcp_service_account" in st.secrets:
            creds = ServiceAccountCredentials.from_json_keyfile_dict(
                st.secrets["gcp_service_account"], scope
            )
            client = gspread.authorize(creds)
            logger.info("Successfully authorized with Google Sheets API.")
            return client
        else:
            logger.warning("No Google Cloud credentials found in secrets.")
            return None
    except Exception as e:
        logger.error(f"Failed to connect to Google Sheets: {e}")
        st.error(f"Failed to connect to Google Sheets. Check your secrets. Error: {e}")
        return None

def get_or_create_worksheet(client, sheet_name, subsheet_name):
    """Gets a subsheet (worksheet) by name, creating it if it doesn't exist."""
    if not client:
        return None
    try:
        spreadsheet = client.open(sheet_name)
        try:
            worksheet = spreadsheet.worksheet(subsheet_name)
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=subsheet_name, rows=100, cols=50)
            logger.info(f"Created new worksheet: '{subsheet_name}' in '{sheet_name}'.")
        return worksheet
    except gspread.SpreadsheetNotFound:
        logger.error(f"Spreadsheet '{sheet_name}' not found.")
        st.warning(f"⚠️ Spreadsheet '{sheet_name}' not found. Results will strictly be available via CSV download.")
        return None
    except Exception as e:
        logger.error(f"Google Sheets Network Error: {e}")
        st.warning("⚠️ Could not connect to Google Sheets (Network/DNS Error). Saving to Sheet is disabled.")
        return None

# ##############################################################################
#  END OF GOOGLE SHEETS CONFIGURATION
# ##############################################################################

# Skills configuration for analysis
SKILLS_TO_ASSESS = [
    'JavaScript', 'Python', 'Node', 'React', 'Java', 'Springboot', 'DSA',
    'AI', 'ML', 'PHP', '.Net', 'Testing', 'AWS', 'Django', 'PowerBI', 'Tableau'
]
SKILL_COLUMNS = [f'{skill}_Probability' for skill in SKILLS_TO_ASSESS]

# ##############################################################################
#  AI-DRIVEN PROJECT CLASSIFICATION
# ##############################################################################
INTERNAL_PROJECT_LIST_FILE = "INTERNAL_PROJECT_LIST.txt"

@st.cache_data
def get_internal_projects_as_string(file_path: str) -> str:
    """
    Loads internal project names from a text file and returns them as a single
    comma-separated string, suitable for embedding in an AI prompt. This is cached.
    """
    if not os.path.exists(file_path):
        logger.warning(
            f'"{file_path}" not found. '
            'Cannot classify internal projects. AI will treat all projects as "External".'
        )
        return ""

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            projects = [line.strip() for line in f if line.strip()]
        logger.info(f"Successfully loaded {len(projects)} internal project names for AI prompt.")
        return ", ".join(projects)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}. All projects will be treated as 'External'.")
        return ""

# Get the project list once at the start.
INTERNAL_PROJECTS_STRING = get_internal_projects_as_string(INTERNAL_PROJECT_LIST_FILE)
# ##############################################################################

# ==============================================================================
#  SESSION STATE INITIALIZATION
# ==============================================================================
if 'comprehensive_results' not in st.session_state:
    st.session_state.comprehensive_results = []
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False
if 'last_analysis_mode' not in st.session_state:
    st.session_state.last_analysis_mode = ""
if 'shortlisting_mode' not in st.session_state:
    st.session_state.shortlisting_mode = "Probability Wise (Default)"

# ==============================================================================
#  UTILS — SANITIZATION & SAFE OPS
# ==============================================================================

CONTROL_CHARS_RE = re.compile(r'[\x00-\x1f\x7f-\x9f]')

def safe_str(x: Any) -> str:
    """Convert any value to a clean string without control characters."""
    try:
        s = str(x)
    except Exception:
        s = ""
    return CONTROL_CHARS_RE.sub('', s)

def coerce_probability_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all probability columns are valid ints (Arrow-friendly)."""
    for col in SKILL_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    if 'GitHub Probability' in df.columns:
        df['GitHub Probability'] = pd.to_numeric(df['GitHub Probability'], errors='coerce').fillna(0).astype(int)
    return df

def sanitize_json_text(text: str) -> str:
    """
    Cleans model JSON by:
    1. Removing markdown fences and any surrounding conversational text.
    2. Isolating the core JSON object or array.
    3. Removing control characters and escaping invalid backslashes.
    4. Attempting to fix truncation errors.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()

    first_brace = text.find('{')
    first_bracket = text.find('[')

    start_pos = -1
    if first_brace == -1 and first_bracket == -1:
        logger.warning("Could not find a JSON start '{' or '[' in the AI response.")
        return '{"error": "No valid JSON object or array found in response"}'

    if first_brace != -1 and first_bracket != -1:
        start_pos = min(first_brace, first_bracket)
    elif first_brace != -1:
        start_pos = first_brace
    else: 
        start_pos = first_bracket

    last_brace = text.rfind('}')
    last_bracket = text.rfind(']')
    end_pos = max(last_brace, last_bracket)

    if end_pos < start_pos:
        logger.warning("Could not find a valid JSON structure {{...}} or [...] in the AI response.")
        return '{"error": "No valid JSON structure found in response"}'

    text = text[start_pos : end_pos + 1]

    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)

    text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)

    if text.count('"') % 2 != 0:
        text += '"'

    open_braces, close_braces = text.count("{"), text.count("}")
    if open_braces > close_braces:
        text += "}" * (open_braces - close_braces)

    open_brackets, close_brackets = text.count("["), text.count("]")
    if open_brackets > close_brackets:
        text += "]" * (open_brackets - close_brackets)

    return text

def relaxed_json_loads(text: str) -> dict:
    """
    Parses JSON robustly by first calling a powerful sanitization function.
    """
    sanitized_text = sanitize_json_text(text)
    try:
        return json.loads(sanitized_text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed even after sanitization: {e}")
        logger.debug(f"--- Sanitized text that failed parsing ---\n{sanitized_text}\n-----------------------------")
        raise

def safe_join_texts(items: Any) -> str:
    """
    Join a list-like into a string; if not list-like, stringify safely.
    """
    if isinstance(items, list):
        return " ".join(safe_str(x) for x in items)
    return safe_str(items)

def is_present_str(s: Any) -> bool:
    """Check if a value indicates current/present in a forgiving way."""
    if s is None:
        return False
    return 'present' in safe_str(s).lower() or 'current' in safe_str(s).lower()

# ==============================================================================
#  PDF DOWNLOADING & EXTRACTION LOGIC
# ==============================================================================

def download_and_identify_file(file_url: str, output_path: str) -> Tuple[bool, str, str]:
    """
    Downloads a file from a URL, saves it, and identifies its type.
    """
    try:
        parsed_url = urlparse(file_url)
        is_google_doc = "docs.google.com" in parsed_url.netloc and "/document/" in parsed_url.path
        is_google_drive = "drive.google.com" in parsed_url.netloc and ("open" in parsed_url.path or "file" in parsed_url.path or "/d/" in parsed_url.path)

        REQUEST_TIMEOUT = 60
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

        if is_google_doc:
            doc_id = parsed_url.path.split('/d/')[1].split('/')[0]
            export_url_parts = list(parsed_url)
            export_url_parts[2] = f"/document/d/{doc_id}/export"
            export_url_parts[4] = "format=pdf"
            pdf_url = urlunparse(export_url_parts)
            logger.info(f"Google Docs URL detected, exporting as PDF from: {pdf_url}")
            
            response = requests.get(pdf_url, headers=headers, stream=True, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        elif is_google_drive:
            file_id = None
            if "id=" in parsed_url.query:
                file_id = parsed_url.query.split("id=")[1].split("&")[0]
            elif "/d/" in parsed_url.path:
                file_id = parsed_url.path.split("/d/")[1].split('/')[0]
            
            if not file_id:
                raise ValueError("Could not extract file ID from Google Drive URL")

            export_url = f"https://docs.google.com/document/d/{file_id}/export?format=pdf"
            try:
                response = requests.get(export_url, stream=True, timeout=REQUEST_TIMEOUT, headers=headers)
                response.raise_for_status()
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Verify PDF header
                with open(output_path, 'rb') as f_check:
                    if not f_check.read(5).startswith(b'%PDF'):
                        raise ValueError("On-the-fly conversion did not result in a valid PDF.")
                logger.info("Successfully converted/downloaded file as PDF from Google Drive.")
            
            except (requests.exceptions.RequestException, ValueError) as e:
                logger.warning(f"On-the-fly GDrive conversion failed ({e}). Falling back to direct download...")
                session = requests.Session()
                session.headers.update(headers)
                URL = "https://docs.google.com/uc?export=download"
                response = session.get(URL, params={'id': file_id}, stream=True, timeout=REQUEST_TIMEOUT)
                token = None
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        token = value
                        break
                if token:
                    params = {'id': file_id, 'confirm': token}
                    response = session.get(URL, params=params, stream=True, timeout=REQUEST_TIMEOUT)
                
                response.raise_for_status()
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
        
        else: # Standard URL download
            response = requests.get(file_url, headers=headers, stream=True, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise ValueError("Downloaded file is empty or does not exist.")

        file_type = 'unsupported'
        with open(output_path, 'rb') as f:
            header = f.read(8)
            if header.startswith(b'%PDF'):
                file_type = 'pdf'
            elif header.startswith(b'\x89PNG\r\n\x1a\n'):
                file_type = 'png'
            elif header.startswith(b'\xff\xd8\xff'):
                file_type = 'jpeg'
        
        logger.info(f"File downloaded successfully to {output_path} and identified as type: {file_type}")
        return True, output_path, file_type

    except requests.exceptions.RequestException as e:
        error_msg = f"Download failed: {str(e)}"
        logger.error(f"Failed to download from {file_url}: {error_msg}")
        return False, error_msg, 'error'
    except ValueError as e:
        error_msg = f"Invalid file or URL: {str(e)}"
        logger.error(f"Error processing {file_url}: {error_msg}")
        return False, error_msg, 'error'
    except Exception as e:
        error_msg = f"An unexpected error occurred during download: {str(e)}"
        logger.error(f"Unexpected error for {file_url}: {error_msg}")
        return False, error_msg, 'error'

def extract_urls_from_pdf_annotations(pdf_path: str) -> List[str]:
    """Extracts hyperlinks from PDF annotations using PyMuPDF."""
    urls = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            links = page.get_links()
            for link in links:
                if link.get("uri"):
                    urls.append(link["uri"])
        doc.close()
    except Exception as e:
        logger.warning(f"Error extracting hyperlinks from {pdf_path}: {e}")
    return urls

def extract_text_and_urls_from_pdf(pdf_path: str) -> Tuple[str, List[str]]:
    """
    Extracts text from a PDF using a robust fallback strategy (pdfplumber -> fitz -> OCR)
    and also extracts embedded hyperlinks.
    """
    text_content = ""
    extracted_urls = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n"
        logger.info(f"Extraction with pdfplumber successful for {pdf_path}")
    except Exception as e:
        logger.warning(f"pdfplumber failed for {pdf_path}: {e}. Trying next method.")

    if not text_content.strip():
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text_content += page.get_text() + "\n"
            doc.close()
            logger.info(f"Extraction with fitz successful for {pdf_path}")
        except Exception as e:
            logger.warning(f"fitz failed for {pdf_path}: {e}. Trying OCR.")

    if not text_content.strip() and st.session_state.get('enable_ocr', True):
        logger.info(f"Falling back to OCR for PDF file {pdf_path}")
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text_content += pytesseract.image_to_string(img) + "\n"
            doc.close()
            logger.info(f"OCR extraction completed for PDF {pdf_path}")
        except Exception as e:
            logger.error(f"OCR extraction failed for PDF {pdf_path}: {e}")

    extracted_urls = extract_urls_from_pdf_annotations(pdf_path)

    return text_content, list(set(extracted_urls))

def extract_text_from_image(image_path: str) -> Tuple[str, List[str]]:
    """
    Extracts text from an image file (PNG, JPEG, etc.) using OCR.
    """
    try:
        if not st.session_state.get('enable_ocr', True):
            logger.warning("OCR is disabled. Cannot process image file.")
            return "OCR is disabled in settings.", []
            
        logger.info(f"Extracting text from image file {image_path} using OCR.")
        img = Image.open(image_path)
        text_content = pytesseract.image_to_string(img)
        logger.info(f"OCR on image file {image_path} completed.")
        return text_content, []
    except Exception as e:
        logger.error(f"OCR extraction failed for image {image_path}: {e}")
        return f"Error during image processing: {e}", []

# ==============================================================================
#  CORE AI & API FUNCTIONS (Mistral)
# ==============================================================================

def analyze_text_with_mistral(prompt: str, api_key: str) -> str:
    """
    Calls Mistral La Plateforme chat completions API.
    Includes JITTER and INCREASED BACKOFF to prevent 429 Rate Limits.
    """
    if not api_key:
        return json.dumps({"error": "Missing Mistral API key for this request."})

    # Create a masked identifier for the key
    key_id = f"...{api_key[-4:]}" if len(api_key) > 4 else "ShortKey"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert resume parser and data analyst. Always return STRICT JSON only—"
                    "no markdown fences, no commentary. Fill missing values with empty strings "
                    '"" or [] where appropriate.'
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }

    attempts = 5
    # INCREASED BACKOFF: Start with 5 seconds wait, not 2
    initial_backoff = 5.0 

    for attempt in range(attempts):
        try:
            # 1. ADD JITTER: Random sleep to desynchronize threads
            time.sleep(uniform(0.5, 2.5))

            resp = requests.post(MISTRAL_ENDPOINT, headers=headers, json=payload, timeout=120)

            if resp.status_code == 200:
                data = resp.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                return content.strip()
            
            elif resp.status_code == 429:
                # 2. RATE LIMIT HANDLING: Wait longer exponentially
                wait_s = (initial_backoff * (2 ** attempt)) + uniform(1, 3)
                logger.warning(
                    f"⚠️ RATE LIMIT (429) on Key {key_id}. "
                    f"Attempt {attempt + 1}/{attempts}. Pausing for {wait_s:.2f}s..."
                )
                time.sleep(wait_s)
                continue
                
            elif resp.status_code in (500, 502, 503, 504):
                wait_s = (initial_backoff * (2 ** attempt)) + uniform(1, 3)
                logger.warning(
                    f"Mistral Server Error {resp.status_code} on Key {key_id}. "
                    f"Attempt {attempt + 1}/{attempts}. Retrying in {wait_s:.2f}s..."
                )
                time.sleep(wait_s)
                continue
            else:
                logger.error(f"Mistral API Error on Key {key_id}: {resp.status_code} - {resp.text}")
                error_payload = {"error": f"API Error Status {resp.status_code}: {resp.text}"}
                return json.dumps(error_payload)
                
        except requests.exceptions.RequestException as e:
            wait_s = (initial_backoff * (2 ** attempt)) + uniform(1, 3)
            logger.warning(
                f"RequestException on Key {key_id} (attempt {attempt + 1}/{attempts}): {e}. "
                f"Retrying in {wait_s:.2f}s..."
            )
            time.sleep(wait_s)
            continue
    
    logger.error(f"Mistral request failed on Key {key_id} after all retries due to persistent errors.")
    final_error_payload = {"error": "API Rate Limit Exceeded. Failed after all retries."}
    return json.dumps(final_error_payload)

# ==============================================================================
#  HELPER & FORMATTING FUNCTIONS
# ==============================================================================

def get_github_repo_count(username):
    """Fetches GitHub public repo count and always returns it as a string."""
    if not username:
        return ""
    try:
        response = requests.get(f'https://api.github.com/users/{username}', timeout=10)
        if response.status_code == 200:
            return str(response.json().get('public_repos', ""))
        else:
            return ""
    except requests.exceptions.RequestException:
        return ""

def extract_github_username(url):
    match = re.search(r'github\.com/([^/]+)', url)
    return match.group(1) if match else None

def format_mobile_number(raw_number: str) -> str:
    """Cleans and formats a mobile number string to a fixed 10-digit format."""
    raw_number = safe_str(raw_number)
    if not raw_number.strip():
        return ""
    digits_only = re.sub(r'\D', '', raw_number)
    if len(digits_only) == 12 and digits_only.startswith('91'):
        return digits_only[2:]
    elif len(digits_only) == 11 and digits_only.startswith('0'):
        return digits_only[1:]
    elif len(digits_only) == 10:
        return digits_only
    else:
        return ""

def sort_links(links_list):
    linkedin_link, github_link = "", ""
    other_links = []
    for link in links_list:
        link = safe_str(link)
        if 'linkedin.com/in/' in link and not linkedin_link:
            linkedin_link = link
        elif 'github.com' in link and not github_link:
            github_link = link
        else:
            other_links.append(link)
    if github_link:
        username = extract_github_username(github_link)
        if username:
            github_link = f"https://github.com/{username}"
    return linkedin_link, github_link, other_links

def get_latest_experience(exp_list: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(exp_list, list) or not exp_list:
        return None
    for exp in exp_list:
        if not isinstance(exp, dict):
            continue
        end_date_str = safe_str(exp.get('endDate', ''))
        if is_present_str(end_date_str):
            return exp
    for exp in exp_list:
        if isinstance(exp, dict):
            return exp
    return None

def get_highest_education_institute(edu_data):
    if not isinstance(edu_data, dict):
        return ""
    if edu_data.get('masters_doctorate') and safe_str(edu_data['masters_doctorate'].get('collegeName')):
        return safe_str(edu_data['masters_doctorate']['collegeName'])
    if edu_data.get('bachelors') and safe_str(edu_data['bachelors'].get('collegeName')):
        return safe_str(edu_data['bachelors']['collegeName'])
    if edu_data.get('diploma') and safe_str(edu_data['diploma'].get('collegeName')):
        return safe_str(edu_data['diploma']['collegeName'])
    if edu_data.get('intermediate_puc_12th') and safe_str(edu_data['intermediate_puc_12th'].get('collegeName')):
        return safe_str(edu_data['intermediate_puc_12th']['collegeName'])
    if edu_data.get('ssc_10th') and safe_str(edu_data['ssc_10th'].get('collegeName')):
        return safe_str(edu_data['ssc_10th']['collegeName'])
    return ""

def check_tesseract_installation():
    try:
        pytesseract.get_tesseract_version()
        return True
    except pytesseract.TesseractNotFoundError:
        return False

def classify_and_format_projects_from_ai(projects: List[Any]) -> Dict[str, str]:
    """
    Reads the 'classification' key from AI output to sort projects.
    """
    internal_titles, internal_techs = [], []
    external_titles, external_techs = [], []

    if not isinstance(projects, list):
        projects = []

    for p in projects:
        if not isinstance(p, dict):
            continue

        title = safe_str(p.get('title', '')).strip()
        if not title:
            continue

        tech_stack_list = p.get('techStack', [])
        if isinstance(tech_stack_list, list):
            tech_stack_str = ", ".join(safe_str(tech) for tech in tech_stack_list)
        else:
            tech_stack_str = safe_str(tech_stack_list).strip()

        classification = safe_str(p.get('classification', 'External')).lower()

        if classification == 'internal':
            internal_titles.append(title)
            if tech_stack_str:
                internal_techs.append(tech_stack_str)
        else:
            external_titles.append(title)
            if tech_stack_str:
                external_techs.append(tech_stack_str)

    return {
        "Internal Project Title": "\n".join(internal_titles),
        "Internal Projects Techstacks": "\n".join(internal_techs),
        "External Project Title": "\n".join(external_titles),
        "External Projects Techstacks": "\n".join(external_techs)
    }

def assign_priority_band(probability: Any) -> str:
    """Assigns a priority band based on an overall probability score."""
    try:
        prob_numeric = float(probability)
    except (ValueError, TypeError):
        return 'Not Shortlisted'

    if prob_numeric >= 90: return 'P1'
    elif 75 <= prob_numeric < 90: return 'P2'
    elif 60 <= prob_numeric < 75: return 'P3'
    else: return 'Not Shortlisted'

def calculate_skill_probabilities(data):
    scores = {column: 0 for column in SKILL_COLUMNS}
    if not isinstance(data, dict):
        return scores

    skills_list = data.get('skills', []) if isinstance(data.get('skills', []), list) else []
    skills_text = " ".join(safe_str(x) for x in skills_list).lower()

    projects = data.get('projects', [])
    if isinstance(projects, list):
        projects_text = " ".join(safe_str(x) for x in projects).lower()
    else:
        projects_text = safe_str(projects).lower()

    certifications = data.get('certifications', [])
    certs_text = " ".join(safe_str(x) for x in certifications).lower() if isinstance(certifications, list) else safe_str(certifications).lower()

    experience_text = ""
    exp_list = data.get('experience', [])
    if isinstance(exp_list, list):
        exp_chunks = []
        for exp in exp_list:
            if isinstance(exp, dict):
                desc = exp.get('description', '')
                if isinstance(desc, list):
                    exp_chunks.append(" ".join(safe_str(x) for x in desc))
                else:
                    exp_chunks.append(safe_str(desc))
                exp_chunks.append(safe_str(exp.get('jobTitle', '')))
                exp_chunks.append(safe_str(exp.get('companyName', '')))
        experience_text = " ".join(exp_chunks).lower()
    else:
        experience_text = safe_str(exp_list).lower()

    education_text = safe_str(data.get('education', {})).lower()
    foundational_text = f"{skills_text} {projects_text} {experience_text} {education_text}"

    for skill in SKILLS_TO_ASSESS:
        score = 0
        skill_lower = skill.lower()
        if skill == '.Net':
            skill_pattern = r'(?i)(?<![a-zA-Z0-9])\.net(?![a-zA-Z0-9])'
        else:
            skill_pattern = r'\b' + re.escape(skill_lower) + r'\b'

        if re.search(skill_pattern, foundational_text):
            score += 10
        if re.search(skill_pattern, experience_text):
            score += 30
        if re.search(skill_pattern, projects_text):
            score += 20
        if re.search(skill_pattern, certs_text):
            score += 20
        scores[f'{skill}_Probability'] = score
    return scores

# ==============================================================================
#  MAIN WORKER FUNCTIONS
# ==============================================================================

# ==============================================================================
#  GITHUB TECHNICAL SCREENING MODULE
# ==============================================================================

def extract_github_info_via_ai(resume_text: str, clickable_links: List[str], api_key: str) -> Dict[str, Any]:
    """Uses AI to exhaustively find GitHub profiles from text, links, or portfolios."""
    prompt = f"""
    Analyze the following resume content and links to find the candidate's GitHub profile.
    Candidates might provide a username, a full URL, or a link hidden in a portfolio site.
    Ignore links ending in '.io' unless they are explicitly GitHub Pages (e.g., username.github.io).
    
    Resume Text: {resume_text}
    Detected Links: {clickable_links}
    
    Return ONLY a JSON object:
    {{
      "github_found": boolean,
      "github_url": "string or null",
      "github_username": "string or null",
      "reasoning": "string"
    }}
    """
    response = analyze_text_with_mistral(prompt, api_key)
    try:
        return relaxed_json_loads(response)
    except Exception as e:
        logger.error(f"Error parsing GitHub info from AI response: {e}")
        return {"github_found": False, "github_url": None, "github_username": None}

def analyze_github_repositories(username: str, required_tech: str) -> Dict[str, Any]:
    """
    ULTIMATE DEEP SCAN: Checks Languages (<1%), Topics, Descriptions, 
    and scans dependency files (requirements.txt, package.json) for frameworks.
    """
    if not username:
        return {"found": False, "matches": [], "error": "No username"}

    # Clean and split required tech (e.g., "Django, FastAPI" -> ['django', 'fastapi'])
    tech_list = [t.strip().lower() for t in re.split(r'[,\s/]+', required_tech) if t.strip()]
    if not tech_list:
        return {"found": False, "matches": [], "error": "No tech stack provided"}

    headers = {"Accept": "application/vnd.github.v3+json"}
    if "GITHUB_TOKEN" in st.secrets:
        headers["Authorization"] = f"token {st.secrets['GITHUB_TOKEN']}"
        
    try:
        # 1. Fetch all public repos
        repo_resp = requests.get(f"https://api.github.com/users/{username}/repos?per_page=100&sort=updated", headers=headers, timeout=15)
        if repo_resp.status_code != 200:
            return {"found": False, "error": f"GitHub API Error: {repo_resp.status_code}"}
        
        repos = repo_resp.json()
        matched_repos = []

        for repo in repos:
            repo_name = repo.get('name', '')
            owner = repo.get('owner', {}).get('login')
            description = (repo.get('description') or '').lower()
            topics = [t.lower() for t in repo.get('topics', [])]
            
            # 2. Languages API (Finds the 1% matches)
            lang_resp = requests.get(f"https://api.github.com/repos/{owner}/{repo_name}/languages", headers=headers, timeout=5)
            repo_langs = [l.lower() for l in lang_resp.json().keys()] if lang_resp.status_code == 200 else []
            
            # 3. Dependency File Scan (For Django, FastAPI, etc.)
            # We look for key files in the root to identify frameworks
            frameworks_detected = []
            manifest_files = ['requirements.txt', 'package.json', 'pyproject.toml', 'go.mod']
            
            # To save API calls, we only scan files if the base language matches (e.g. Python for Django)
            if any(lang in ['python', 'javascript', 'typescript', 'go'] for lang in repo_langs):
                for file_name in manifest_files:
                    file_url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{file_name}"
                    file_resp = requests.get(file_url, headers=headers, timeout=5)
                    if file_resp.status_code == 200:
                        # Extract text from the manifest file
                        import base64
                        try:
                            content = base64.b64decode(file_resp.json().get('content', '')).decode('utf-8').lower()
                            # Check if any of our required tech is inside the requirements/package file
                            for tech in tech_list:
                                if tech in content:
                                    frameworks_detected.append(tech)
                        except Exception:
                            pass

            # 4. Create the Search Blob
            search_blob = f"{repo_name} {description} {' '.join(topics)} {' '.join(repo_langs)} {' '.join(frameworks_detected)}".lower()

            # 5. Final Matching
            matches_for_this_repo = []
            for tech in tech_list:
                pattern = r'\b' + re.escape(tech) + r'\b'
                if re.search(pattern, search_blob):
                    matches_for_this_repo.append(tech)

            if matches_for_this_repo:
                # 6. Verify Ownership/Commits
                v_url = f"https://api.github.com/repos/{owner}/{repo_name}/commits?author={username}&per_page=1"
                v_resp = requests.get(v_url, headers=headers, timeout=5)
                is_verified = (v_resp.status_code == 200 and len(v_resp.json()) > 0) or (owner.lower() == username.lower())
                
                if is_verified:
                    matched_repos.append({
                        "name": repo_name,
                        "url": repo.get('html_url'),
                        "tech_detected": ", ".join(set(matches_for_this_repo)).title(),
                        "details": "Matched via " + ("Files" if frameworks_detected else "Metadata")
                    })

        return {
            "found": True,
            "matches": matched_repos,
            "match_count": len(matched_repos),
            "commits_verified": len(matched_repos) > 0
        }
    except Exception as e:
        return {"found": False, "error": str(e)}
    
def process_resume_for_shortlisting(row, resume_index, user_requirements, company_name, api_key, **kwargs):
    """
    Worker for shortlisting. 
    Includes STRICT Python-based Keyword Validation to prevent "JavaScript" == "Java" confusion.
    """
    user_id = row['user_id']
    resume_link = row['Resume link']

    result = {
        'User ID': user_id,
        'Resume Link': resume_link,
        'Company Name': company_name,
        'Overall Probability': 0, 'Overall Remarks': "Error processing",
        'Priority Band': "Not Shortlisted",
        'Projects Probability': 0, 'Projects Remarks': "",
        'Skills Probability': 0, 'Skills Remarks': "",
        'Experience Probability': 0, 'Experience Remarks': "",
        'Other Probability': 0, 'Other Remarks': "",
        'Internal Project Title': "", 'Internal Projects Techstacks': "",
        'External Project Title': "", 'External Projects Techstacks': "",
        'Total Projects Count': 0, 'Internal Projects Count': 0, 'External Projects Count': 0,
    }

    temp_file_path = None
    try:
        logger.info(f"Shortlisting resume #{resume_index + 1} for user {user_id} with Project Extraction.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
            temp_file_path = tmp.name

        download_success, msg_or_path, file_type = download_and_identify_file(resume_link, temp_file_path)

        if not download_success:
            raise ValueError(f"Download Error: {msg_or_path}")

        if file_type == 'pdf':
            resume_text, clickable_links = extract_text_and_urls_from_pdf(temp_file_path)
        elif file_type in ['png', 'jpeg']:
            resume_text, _ = extract_text_from_image(temp_file_path)
        else:
            raise ValueError(f"Unsupported file type.")

        if not resume_text.strip():
            raise ValueError("Could not extract any text from the file.")

        # ==============================================================================
        #  CRITICAL FIX: PYTHON-BASED "JAVA vs JAVASCRIPT" VALIDATION
        # ==============================================================================
        
        text_lower = resume_text.lower()
        reqs_lower = user_requirements.lower()
        system_warning = ""

        # Check for Java specifically using REGEX (Word Boundaries)
        if re.search(r'\bjava\b', reqs_lower):
            has_standalone_java = re.search(r'\bjava\b', text_lower)
            if not has_standalone_java:
                system_warning += "\n\n[SYSTEM WARNING]: The user explicitly requires 'Java' (the backend language). I have scanned the text and 'Java' appears to be MISSING as a standalone word. The text might contain 'JavaScript', but THAT IS NOT JAVA. Treat 'Java' as MISSING."

        # ==============================================================================

        project_instruction_block = ""
        if INTERNAL_PROJECTS_STRING:
            project_instruction_block = f"""
"projects": Analyze the resume for projects. For each project, extract its title and techStack. CRITICALLY, you must add a "classification" field. Classify a project as "Internal" if its title, description or context matches any project from the OFFICIAL INTERNAL PROJECTS LIST provided below. Otherwise, classify it as "External". Be flexible in your matching (e.g., 'Jobby App' in the list should match 'Jobby-app' or 'Jobby Application' in a resume).
OFFICIAL INTERNAL PROJECTS LIST: {INTERNAL_PROJECTS_STRING}
Example Project Entry: {{ "title": "Jobby App", "techStack": ["React", "JS"], "classification": "Internal" }}
"""
        else:
            project_instruction_block = f"""
"projects": [ {{ "title": "string", "techStack": ["list of tech keywords"], "classification": "External" }} ]
"""

        prompt = f"""
You are a Nuanced Technical Recruiter and Logic Engine.
Your goal is to categorize the candidate into Priority Bands (P1, P2, P3) based on strict keyword matching.

{system_warning}

**CRITICAL ANTI-HALLUCINATION RULES:**
1. **JAVA IS NOT JAVASCRIPT.** - If the resume contains "JavaScript", "ECMAScript", or "React.js", DO NOT count this as "Java".
   - "Java" is a standalone backend language. "JavaScript" is a frontend language.
   - If the candidate lists "JavaScript Essentials" certification, that is **NOT** Java.
   - If the resume text does not explicitly say "Java" as a separate word, count it as MISSING.

**SCORING GUIDELINES (STRICTLY FOLLOW THIS):**

**BAND P1 (Score 90 - 100): THE PERFECT MATCH**
- The candidate has **ALL** the specific technologies listed in the Required Criteria.
- Example: If user asks for "Java, Springboot, React", the resume MUST have ALL THREE to get > 90.
- If even ONE core skill (especially Java) is missing, DO NOT give a score above 90.

**BAND P2 (Score 75 - 89): THE STRONG CONTENDER (Missing 1-2 Skills)**
- The candidate matches **MOST** of the criteria but is missing a specific technology.
- Example: User asks for "Java, Springboot, React". Candidate has "React and Node" but NO "Java".
- **Action:** This is still a good profile. Do NOT give 0. Give a score between 75 and 89.
- **Remarks:** You must explicitly state: "Candidate fits P2. Good frontend skills, but missing required Java."

**BAND P3 (Score 60 - 74): THE PARTIAL MATCH**
- The candidate has relevant skills but is missing **MAJOR** parts of the stack.
- Example: User asks for "Full Stack Java". Candidate only knows "HTML and CSS".

**BAND F (Score 0 - 59): NO MATCH**
- The resume is completely unrelated to the job description.

Return your answer as a **single, pure JSON object**.

**REQUIRED JSON STRUCTURE:**
{{
  "projects_probability": "integer (0–100)",
  "projects_remarks": "string",
  "skills_probability": "integer (0–100)",
  "skills_remarks": "string",
  "experience_probability": "integer (0–100)",
  "experience_remarks": "string",
  "other_probability": "integer (0–100)",
  "other_remarks": "string",
  "overall_probability": "integer (0–100)",
  "overall_remarks": "string",
  {project_instruction_block}
}}

---
**Required Criteria:**
{user_requirements}
---
**Resume Text:**
{resume_text}
---
"""
        mistral_response_text = analyze_text_with_mistral(prompt, api_key=api_key)

        data = {}
        try:
            data = relaxed_json_loads(mistral_response_text)
            if not isinstance(data, dict):
                raise ValueError(f"AI returned data that is not a JSON object. Type: {type(data)}")
            if "error" in data:
                raise ValueError(data["error"])
        except Exception as e:
            raise ValueError(f"Error parsing AI response: {e}")

        result.update({
            'Overall Probability': data.get('overall_probability', 0),
            'Projects Probability': data.get('projects_probability', 0),
            'Skills Probability': data.get('skills_probability', 0),
            'Experience Probability': data.get('experience_probability', 0),
            'Other Probability': data.get('other_probability', 0),
            'Overall Remarks': data.get('overall_remarks', 'N/A'),
            'Projects Remarks': data.get('projects_remarks', 'N/A'),
            'Skills Remarks': data.get('skills_remarks', 'N/A'),
            'Experience Remarks': data.get('experience_remarks', 'N/A'),
            'Other Remarks': data.get('other_remarks', 'N/A'),
        })

        projects_data = data.get('projects', [])
        classified_projects = classify_and_format_projects_from_ai(projects_data)
        result.update(classified_projects)

        # --- ENHANCED GITHUB SCREENING INTEGRATION ---
        github_skills_input = kwargs.get('github_skills', '')
        gh_info = extract_github_info_via_ai(resume_text, clickable_links, api_key)
        
        # Initialize default value for the new column
        result['GitHub Probability'] = 0
        
        if gh_info.get("github_found") and gh_info.get("github_username"):
            analysis_criteria = github_skills_input if github_skills_input.strip() else user_requirements
            gh_analysis = analyze_github_repositories(gh_info["github_username"], analysis_criteria)
            
            # --- UPDATED PROBABILITY LOGIC ---
            # If any verified matches exist, score is 100
            if gh_analysis.get("commits_verified"):
                result['GitHub Probability'] = 100
                result['Commit Ownership Verified'] = "Yes — commits made by the user"
            else:
                result['GitHub Probability'] = 0
                result['Commit Ownership Verified'] = "No — no commits by the user"

            result['GitHub Screening Outcome'] = "Verified Matches Found" if gh_analysis.get("match_count", 0) > 0 else "No Matching Tech Found"
            result['Verified GitHub Repos'] = "\n".join([f"{r['name']} ({r['tech_detected']})" for r in gh_analysis.get("matches", [])])
            result['GitHub Profile Used'] = gh_info.get("github_url")
        else:
            result['GitHub Screening Outcome'] = "Profile Not Found"
            result['Verified GitHub Repos'] = "N/A"
            result['GitHub Profile Used'] = "None"
            result['GitHub Probability'] = 0
            result['Commit Ownership Verified'] = "No — no profile found"
        # ----------------------------------------------

        internal_titles_str = classified_projects.get('Internal Project Title', '')
        external_titles_str = classified_projects.get('External Project Title', '')
        internal_count = len(internal_titles_str.splitlines()) if internal_titles_str else 0
        external_count = len(external_titles_str.splitlines()) if external_titles_str else 0
        result['Internal Projects Count'] = internal_count
        result['External Projects Count'] = external_count
        result['Total Projects Count'] = internal_count + external_count

    except Exception as e:
        logger.error(f"Failed shortlisting {user_id} ({resume_link}): {e}")
        error_msg = f"Error: {e}"
        result['Overall Remarks'] = error_msg

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError as e:
                logger.error(f"Error removing temporary file {temp_file_path}: {e}")

    return result

def process_resume_comprehensively(row, resume_index, analysis_type, company_name, api_key):
    """Worker for comprehensive data extraction with AI-driven project classification."""
    user_id = row['user_id']
    resume_link = row['Resume link']

    if analysis_type == "Internal Projects Matching":
        result_cols = [
            'User ID', 'Resume Link', 'Total Projects Count', 'Internal Projects Count', 'External Projects Count',
            'Internal Project Titles', 'Internal Project Techstacks',
            'External Project Titles', 'External Project Techstacks'
        ]
        result = {col: "" for col in result_cols}
    else:
        base_columns = [
            'User ID', 'Resume Link', 'Full Name', 'Mobile Number', 'Email ID', 'LinkedIn Link', 'GitHub Link',
            'Other Links', 'Skills', 'Internal Project Title', 'Internal Projects Techstacks',
            'External Project Title', 'External Projects Techstacks', 'Latest Experience Company Name',
            'Latest Experience Job Title', 'Latest Experience Start Date', 'Latest Experience End Date',
            'Currently Working? (Yes/No)', 'Years of IT Experience', 'Years of Non-IT Experience', 'City', 'State',
            'Certifications', 'Awards', 'Achievements', 'GitHub Repo Count', 'Highest Education Institute Name'
        ]
        education_cols = [
            f"{level} {field}"
            for level in ['Masters/Doctorate', 'Bachelors', 'Diploma', 'Intermediate / PUC / 12th', 'SSC / 10th']
            for field in ['Course Name', 'College Name', 'Department Name', 'Year of Completion', 'Percentage']
        ]
        education_cols = [c.replace("Intermediate / PUC / 12th Course Name", "Intermediate / PUC / 12th Name") for c in education_cols]
        education_cols = [c.replace("SSC / 10th Course Name", "SSC / 10th Name") for c in education_cols]
        education_cols = [c for c in education_cols if "Masters/Doctorate Name" not in c and "Bachelors Name" not in c and "Diploma Name" not in c]

        all_columns = base_columns + SKILL_COLUMNS + education_cols
        result = {col: "" for col in all_columns}

    result['User ID'] = user_id
    result['Resume Link'] = resume_link
    result['Company Name'] = company_name

    temp_file_path = None
    mistral_response_text_for_logging = ""
    try:
        logger.info(f"Processing resume #{resume_index + 1} for user {user_id} with analysis type: {analysis_type}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
            temp_file_path = tmp.name

        download_success, msg_or_path, file_type = download_and_identify_file(resume_link, temp_file_path)

        if not download_success:
            raise ValueError(f"Download Error: {msg_or_path}")

        if file_type == 'pdf':
            resume_text, clickable_links = extract_text_and_urls_from_pdf(temp_file_path)
        elif file_type in ['png', 'jpeg']:
            resume_text, clickable_links = extract_text_from_image(temp_file_path)
        else:
            raise ValueError(f"Unsupported file type: The file is not a valid PDF or image.")

        if not resume_text.strip():
            raise ValueError("Could not extract any text from the file.")

        project_instruction_block = ""
        if INTERNAL_PROJECTS_STRING and analysis_type != "Personal Details":
            project_instruction_block = f"""
"projects": Analyze the resume for projects. For each project, extract its title and techStack. CRITICALLY, you must add a "classification" field. Classify a project as "Internal" if its title, description or context matches any project from the OFFICIAL INTERNAL PROJECTS LIST provided below. Otherwise, classify it as "External". Be flexible in your matching (e.g., 'Jobby App' in the list should match 'Jobby-app' or 'Jobby Application' in a resume).
OFFICIAL INTERNAL PROJECTS LIST: {INTERNAL_PROJECTS_STRING}
Example Project Entry: {{ "title": "Jobby App", "techStack": ["React", "JS"], "classification": "Internal" }}
"""
        else:
            project_instruction_block = f"""
"projects": [ {{ "title": "string", "techStack": ["list of tech keywords"], "classification": "External" }} ]
"""

        prompt = ""
        if analysis_type == "Internal Projects Matching":
            prompt = f"""
You are a project classification expert. Analyze the provided resume text and perform this CRITICAL task:
1. Extract all projects mentioned in the resume.
2. For each project, determine if it is an "Internal" or "External" project by comparing it against the provided OFFICIAL INTERNAL PROJECTS LIST. Your matching should be smart and flexible (e.g., 'Jobby App' in the list should match 'Jobby-app' in the resume).
3. Return ONLY a pure JSON object with the results.

OFFICIAL INTERNAL PROJECTS LIST:
---
{INTERNAL_PROJECTS_STRING}
---

REQUIRED JSON Structure:
{{
  "projects": [
    {{
      "title": "string",
      "techStack": ["list", "of", "technologies"],
      "classification": "Internal" or "External"
    }}
  ]
}}

Resume Text:
---
{resume_text}
---
"""
        elif analysis_type == "All Data":
            prompt = f"""
You are a machine that strictly outputs a single, valid JSON object. Analyze the resume text provided below to populate the specified JSON structure.

**JSON STRUCTURE AND INSTRUCTIONS:**
{{
  "fullName": "string", "mobileNumber": "string", "email": "string",
  "address": {{"city": "string", "state": "string"}}, "textLinks": ["list of all URLs found"],
  "skills": ["list of strings"], "certifications": ["list of strings"], "awards": ["list of strings"],
  "achievements": ["list of strings"], "yearsITExperience": "float or string", "yearsNonITExperience": "float or string",
  {project_instruction_block}
  "education": {{
    "masters_doctorate": {{"courseName": "string", "departmentName": "string", "completionYear": "string", "percentage": "string", "collegeName": "string"}},
    "bachelors": {{"courseName": "string", "departmentName": "string", "completionYear": "string", "percentage": "string", "collegeName": "string"}},
    "diploma": {{"courseName": "string", "departmentName": "string", "completionYear": "string", "percentage": "string", "collegeName": "string"}},
    "intermediate_puc_12th": {{"schoolName": "string", "departmentName": "string", "completionYear": "string", "percentage": "string", "collegeName": "string"}},
    "ssc_10th": {{"schoolName": "string", "completionYear": "string", "percentage": "string", "collegeName": "string"}}
  }},
  "experience": [ {{ "companyName": "string", "jobTitle": "string", "startDate": "string", "endDate": "string", "description": "string or list of strings" }} ]
}}

**CRITICAL NOTE ON PROJECTS:**
Refer to the "projects" instruction above. You MUST use the OFFICIAL INTERNAL PROJECTS LIST to classify projects correctly.

Resume Text:
---
{resume_text}
---
"""
        elif analysis_type == "Personal Details":
            prompt = f"""
Analyze the provided resume text and extract ONLY the personal details into a pure JSON object.
The entire response MUST be ONLY the JSON object.
JSON Structure: {{"fullName": "string", "mobileNumber": "string", "email": "string", "address": {{"city": "string", "state": "string"}}, "textLinks": ["list of strings"]}}
Resume Text: --- {resume_text} ---
"""
        elif analysis_type == "Skills & Projects":
            prompt = f"""
You are an expert data extractor. Analyze the resume and produce a single JSON object.

**JSON STRUCTURE AND INSTRUCTIONS:**
{{
  "skills": ["list of strings"],
  "certifications": ["list of strings"],
  "awards": ["list of strings"],
  "achievements": ["list of strings"],
  {project_instruction_block}
  "experience": [{{ "companyName": "string", "jobTitle": "string", "startDate": "string", "endDate": "string", "description": "string"}}]
}}

**CRITICAL NOTE ON PROJECTS:**
Refer to the "projects" instruction above. You MUST use the OFFICIAL INTERNAL PROJECTS LIST to classify projects correctly.

Resume Text:
---
{resume_text}
---
"""

        mistral_response_text = analyze_text_with_mistral(prompt, api_key=api_key)
        mistral_response_text_for_logging = mistral_response_text
        data = relaxed_json_loads(mistral_response_text)
        if not isinstance(data, dict): raise ValueError(f"AI returned non-dict data. Type: {type(data)}")
        if "error" in data: raise ValueError(data["error"])

        projects_data = data.get('projects', [])
        classified_projects = classify_and_format_projects_from_ai(projects_data)

        if analysis_type == "Internal Projects Matching":
            internal_titles_str = classified_projects.get('Internal Project Title', '')
            external_titles_str = classified_projects.get('External Project Title', '')
            internal_count = len(internal_titles_str.splitlines()) if internal_titles_str else 0
            external_count = len(external_titles_str.splitlines()) if external_titles_str else 0

            result.update({
                'Total Projects Count': internal_count + external_count,
                'Internal Projects Count': internal_count,
                'External Projects Count': external_count,
                'Internal Project Titles': internal_titles_str,
                'Internal Project Techstacks': classified_projects.get('Internal Projects Techstacks', ''),
                'External Project Titles': external_titles_str,
                'External Project Techstacks': classified_projects.get('External Projects Techstacks', '')
            })
        else:
            result.update(classified_projects)

            if analysis_type in ["All Data", "Personal Details"]:
                addr = data.get('address', {}) if isinstance(data.get('address', {}), dict) else {}
                result.update({
                    'Full Name': safe_str(data.get('fullName', "")), 'Mobile Number': format_mobile_number(data.get('mobileNumber', "")),
                    'Email ID': safe_str(data.get('email', "")), 'City': safe_str(addr.get('city', "")), 'State': safe_str(addr.get('state', "")),
                })
                text_links = data.get('textLinks', [])
                if not isinstance(text_links, list): text_links = [safe_str(text_links)] if text_links else []
                all_links = sorted(list(set([safe_str(x) for x in text_links] + clickable_links)))
                linkedin, github, others = sort_links(all_links)
                result.update({'LinkedIn Link': linkedin, 'GitHub Link': github, 'Other Links': "\n".join(others)})
                if github: result['GitHub Repo Count'] = get_github_repo_count(extract_github_username(github))

            if analysis_type in ["All Data", "Skills & Projects"]:
                result.update(calculate_skill_probabilities(data))
                skills_data = data.get('skills', [])
                if isinstance(skills_data, list):
                    cleaned_skills = [safe_str(s) for s in skills_data]
                    unique_skills = sorted(list(set(s for s in cleaned_skills if s)))
                    result['Skills'] = ", ".join(unique_skills)

                certs_data = data.get('certifications', [])
                if isinstance(certs_data, list):
                    cleaned_certs = [safe_str(c) for c in certs_data]
                    result['Certifications'] = "\n".join(c for c in cleaned_certs if c)

                awards_data = data.get('awards', [])
                if isinstance(awards_data, list):
                    cleaned_awards = [safe_str(a) for a in awards_data]
                    result['Awards'] = "\n".join(a for a in cleaned_awards if a)

                achievements_data = data.get('achievements', [])
                if isinstance(achievements_data, list):
                    cleaned_achievements = [safe_str(a) for a in achievements_data]
                    result['Achievements'] = "\n".join(a for a in cleaned_achievements if a)

                latest_exp = get_latest_experience(data.get('experience', []))
                if latest_exp:
                    end_date = latest_exp.get('endDate', "")
                    result.update({
                        'Latest Experience Company Name': safe_str(latest_exp.get('companyName', "")),
                        'Latest Experience Job Title': safe_str(latest_exp.get('jobTitle', "")),
                        'Latest Experience Start Date': safe_str(latest_exp.get('startDate', "")),
                        'Latest Experience End Date': safe_str(end_date), 'Currently Working? (Yes/No)': "Yes" if is_present_str(end_date) else "No"
                    })

            if analysis_type == "All Data":
                result.update({'Years of IT Experience': safe_str(data.get('yearsITExperience', "")), 'Years of Non-IT Experience': safe_str(data.get('yearsNonITExperience', ""))})

                edu = data.get('education', {}) if isinstance(data.get('education'), dict) else {}

                result['Highest Education Institute Name'] = get_highest_education_institute(edu)
                edu_levels = {
                    'masters_doctorate': ('Masters/Doctorate', 'courseName'), 'bachelors': ('Bachelors', 'courseName'),
                    'diploma': ('Diploma', 'courseName'), 'intermediate_puc_12th': ('Intermediate / PUC / 12th', 'schoolName'),
                    'ssc_10th': ('SSC / 10th', 'schoolName')
                }
                for key, (prefix, name_key) in edu_levels.items():
                    level_data = edu.get(key, {}) if isinstance(edu.get(key, {}), dict) else {}
                    if key in ['intermediate_puc_12th', 'ssc_10th']:
                        result[f'{prefix} Name'] = safe_str(level_data.get(name_key, ''))
                    result[f'{prefix} Course Name'] = safe_str(level_data.get('courseName', ''))
                    result[f'{prefix} College Name'] = safe_str(level_data.get('collegeName', ''))
                    result[f'{prefix} Department Name'] = safe_str(level_data.get('departmentName', ''))
                    result[f'{prefix} Year of Completion'] = safe_str(level_data.get('completionYear', ''))
                    result[f'{prefix} Percentage'] = safe_str(level_data.get('percentage', ''))

    except json.JSONDecodeError as e:
        error_msg = "Error: AI returned a malformed response that could not be parsed."
        logger.error(f"Failed processing {user_id} ({resume_link}): JSONDecodeError - {e}")
        logger.debug(f"--- Full Malformed Response ---\n{mistral_response_text_for_logging}\n-----------------------------")
        if analysis_type == "Internal Projects Matching":
            result['Total Projects Count'] = error_msg
        else:
            result['Full Name'] = error_msg
    except Exception as e:
        logger.error(f"Failed processing {user_id} ({resume_link}): {e}", exc_info=True)
        error_msg_display = f"Error: {e}"
        if analysis_type == "Internal Projects Matching":
            result['Total Projects Count'] = error_msg_display
        else:
            result['Full Name'] = error_msg_display

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try: os.remove(temp_file_path)
            except OSError as e: logger.error(f"Error removing temp file {temp_file_path}: {e}")

    return result

# ==============================================================================
#  BATCH PROCESSING & UI
# ==============================================================================

def process_resumes_in_batches_live(df, batch_size, worker_function, display_columns, **kwargs):
    st.session_state.comprehensive_results = []

    progress_text = st.empty()
    progress_bar = st.progress(0)
    results_placeholder = st.empty()

    gspread_client = get_gspread_client()
    worksheet = None
    if gspread_client:
        analysis_mode = st.session_state.get('last_analysis_mode', 'default')
        subsheet_name = ""

        if analysis_mode == 'shortlisting':
            shortlisting_type = st.session_state.get('shortlisting_mode', "Probability Wise (Default)")
            subsheet_name = "Priority_Wise_Results" if shortlisting_type == "Priority Wise (P1 / P2 / P3 Bands)" else "Probability_Wise_Results"
        else:
            subsheet_name = analysis_mode.replace(" ", "_")
        worksheet = get_or_create_worksheet(gspread_client, GSHEET_NAME, subsheet_name)

    num_resumes = len(df)
    # Check key availability
    if not MISTRAL_API_KEYS:
        st.error("No API Keys found! Cannot proceed.")
        return

    num_keys = len(MISTRAL_API_KEYS)
    logger.info(f"Using {num_keys} API keys for {num_resumes} resumes.")

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {}
        for i, (df_index, row) in enumerate(df.iterrows()):
            key_index = i % num_keys
            assigned_key = MISTRAL_API_KEYS[key_index]
            
            future = executor.submit(worker_function, row, df_index, **kwargs, api_key=assigned_key)
            futures[future] = df_index

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()

            if (st.session_state.get('last_analysis_mode') == 'shortlisting' and
                st.session_state.get('shortlisting_mode') == "Priority Wise (P1 / P2 / P3 Bands)"):
                result['Priority Band'] = assign_priority_band(result.get('Overall Probability', 0))

            result['Analysis Datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            st.session_state.comprehensive_results.append(result)
            progress = (i + 1) / len(df)
            progress_text.markdown(f"**Processing... {i+1}/{len(df)} resumes completed.**")
            progress_bar.progress(progress)

            temp_df = pd.DataFrame(st.session_state.comprehensive_results)
            numeric_cols = ['Total Projects Count', 'Internal Projects Count', 'External Projects Count']
            for col in numeric_cols:
                if col in temp_df.columns:
                    temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0).astype(int)
            if "All Data" in st.session_state.get('last_analysis_mode', ''):
                temp_df = coerce_probability_columns(temp_df)
            cols_to_show = [col for col in display_columns if col in temp_df.columns]
            results_placeholder.dataframe(temp_df[cols_to_show], height=400)

    if worksheet and st.session_state.comprehensive_results:
        try:
            logger.info(f"Preparing to batch-write {len(st.session_state.comprehensive_results)} rows to Google Sheets.")
            header = worksheet.row_values(1)
            
            all_result_keys = set()
            for res_dict in st.session_state.comprehensive_results:
                all_result_keys.update(res_dict.keys())

            final_header = header.copy()
            new_cols_to_add = [key for key in all_result_keys if key not in final_header]

            if not header:
                final_header = sorted(list(all_result_keys))
                worksheet.append_row(final_header, value_input_option='USER_ENTERED')
            elif new_cols_to_add: 
                final_header.extend(new_cols_to_add)
                worksheet.update('A1', [final_header])

            rows_to_append = []
            for result_dict in st.session_state.comprehensive_results:
                row_values = [result_dict.get(col, "") for col in final_header]
                rows_to_append.append(row_values)
            
            if rows_to_append:
                worksheet.append_rows(rows_to_append, value_input_option='USER_ENTERED')
                logger.info("Successfully batch-wrote all rows to Google Sheets.")

        except Exception as e:
            logger.error(f"Failed to batch-write to Google Sheets: {e}")
            st.toast("⚠️ Could not batch-write results to Google Sheet.", icon="📄")

    progress_text.success(f"**✅ Analysis Complete! {len(df)}/{len(df)} resumes processed. Results saved to Google Sheets.**")

# ==============================================================================
#  MAIN STREAMLIT APPLICATION
# ==============================================================================
def main():
    st.set_page_config(page_title="AI Resume Analyzer", layout="wide", page_icon="📄")

    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Determine number of keys loaded
        num_keys_loaded = len(MISTRAL_API_KEYS) if MISTRAL_API_KEYS else 1
        
        st.session_state.batch_size = st.slider(
            "Concurrency",
            min_value=1, 
            max_value=20, 
            value=num_keys_loaded,
            help=f"Loaded {num_keys_loaded} keys. It is recommended to keep this slider at {num_keys_loaded} to avoid Rate Limits."
        )
        st.session_state.enable_ocr = st.checkbox(
            "Enable OCR for PDFs & Images",
            value=True,
            help="Required to read text from scanned PDFs and image files (PNG, JPG)."
        )
        if not check_tesseract_installation() and st.session_state.get('enable_ocr'):
            st.error("Tesseract is not installed or not in your PATH. OCR will not function.")

    st.subheader("Step 1: Provide Resume Data")

    input_method = st.radio("Choose input method:", ["Upload CSV", "Paste Text"], horizontal=True, label_visibility="collapsed", index=1)
    df_input = None

    if input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV file with 'user_id' and 'Resume link' columns.", type="csv", label_visibility="collapsed")
        if uploaded_file:
            try: df_input = pd.read_csv(uploaded_file, dtype=str).fillna("")
            except Exception as e: st.error(f"Error reading CSV file: {e}")
    else:
        text_data = st.text_area("Paste data here (user_id [Tab] resume_link)", height=150, label_visibility="collapsed", placeholder="user1\thttp://example.com/resume.pdf\nuser2\thttps://example.com/resume.png")
        if text_data:
            try: df_input = pd.read_csv(StringIO(text_data), header=None, names=['user_id', 'Resume link'], dtype=str, sep='\t').fillna("")
            except Exception as e: st.error(f"Could not parse text. Error: {e}")

    if df_input is not None:
        df_input.dropna(subset=['user_id', 'Resume link'], inplace=True)
        df_input = df_input[df_input['Resume link'].str.strip() != ''].reset_index(drop=True)
        if not all(col in df_input.columns for col in ['user_id', 'Resume link']):
            st.error("Input data must contain 'user_id' and 'Resume link' columns.")
        else:
            st.success(f"Successfully loaded {len(df_input)} resume entries.")
            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Step 2: Priority-Based Shortlisting")
                user_requirements = st.text_area(
                    "Enter Job Description or Requirements for Shortlisting",
                    placeholder="e.g., 'Seeking a senior Python developer with 5+ years of experience...' or 'Java Certification'",
                    height=150,
                    help="Paste any text here: a full job description, a list of keywords, or a detailed prompt. The analysis will focus on your query's intent."
                )

                company_name = st.text_input(
                    "Enter Company Name (Required)",
                    placeholder="e.g., Acme Corporation",
                    help="This name is required to identify the analysis batch and will be saved with the results."
                )
                # NEW DYNAMIC GITHUB SKILLS INPUT
                github_skills = st.text_input(
                    "Enter Required Skills for GitHub Analysis",
                    placeholder="e.g., Python, TensorFlow, Docker",
                    help="GitHub repositories will be evaluated specifically against these technologies. If left blank, the main Job Description will be used."
                )

                if user_requirements.strip():
                    st.info("**Mode:** Priority-Based Shortlisting (Focused Analysis)")
                else:
                    st.info(f"**Mode:** Comprehensive Extraction")

            with col2:
                st.subheader("Step 3: Comprehensive Data Extraction")
                analysis_type = st.selectbox(
                    "Choose data to extract (used if shortlisting is empty):",
                    ("All Data", "Personal Details", "Skills & Projects", "Internal Projects Matching"),
                    help="This analysis runs only if the 'Job Description' box is left empty."
                )

                st.write("")
                st.subheader("Step 4: Start Analysis")
                
                shortlisting_options = ["Probability Wise (Default)", "Priority Wise (P1 / P2 / P3 Bands)"]
                shortlisting_default_index = 1
                
                if user_requirements.strip():
                    shortlisting_mode = st.selectbox(
                        "Choose Shortlisting Mode",
                        options=shortlisting_options,
                        index=shortlisting_default_index 
                    )
                    button_text = f"🚀 Start Shortlisting for {len(df_input)} Resumes"
                else:
                    shortlisting_mode = "N/A"
                    button_text = f"🚀 Start '{analysis_type}' Extraction for {len(df_input)} Resumes"

                start_button = st.button(
                    button_text,
                    type="primary",
                    disabled=not company_name.strip(),
                    help="Please enter a Company Name to start the analysis." if not company_name.strip() else ""
                )

                if not company_name.strip() and df_input is not None and not start_button:
                     st.warning("Company Name is a required field.", icon="⚠️")

            live_results_container = st.container()

            if start_button:
                st.session_state.analysis_running = True
                with live_results_container:
                    if user_requirements.strip():
                        st.session_state.last_analysis_mode = "shortlisting"
                        st.session_state.shortlisting_mode = shortlisting_mode
                        
                        display_columns = [
                            'User ID', 'Resume Link', 'Overall Probability', 'Overall Remarks',
    'GitHub Screening Outcome', 'GitHub Profile Used', 
    'Commit Ownership Verified', # NEW COLUMN
    'GitHub Probability', 
    'Verified GitHub Repos',
    'Projects Probability', 'Projects Remarks', 'Skills Probability', 'Skills Remarks',
    'Experience Probability', 'Experience Remarks', 'Other Probability', 'Other Remarks',
    'Total Projects Count', 'Internal Projects Count', 'External Projects Count',
    'Internal Project Title', 'Internal Projects Techstacks',
    'External Project Title', 'External Projects Techstacks'
                        ]

                        if st.session_state.shortlisting_mode == "Priority Wise (P1 / P2 / P3 Bands)":
                            display_columns.insert(3, 'Priority Band')

                        process_resumes_in_batches_live(
                            df=df_input, batch_size=st.session_state.batch_size, worker_function=process_resume_for_shortlisting,
                            display_columns=display_columns, user_requirements=user_requirements.strip(), company_name=company_name.strip(),
                            github_skills=github_skills.strip() # ADDED THIS LINE
                        )
                    else:
                        st.session_state.last_analysis_mode = analysis_type

                        if analysis_type == "Internal Projects Matching":
                            display_columns = [
                                'User ID', 'Resume Link', 'Total Projects Count', 'Internal Projects Count', 'External Projects Count',
                                'Internal Project Titles', 'Internal Project Techstacks',
                                'External Project Titles', 'External Project Techstacks'
                            ]
                        else:
                            all_extraction_columns = [
                                'User ID', 'Resume Link', 'Full Name', 'Mobile Number', 'Email ID',
                                'LinkedIn Link', 'GitHub Link', 'GitHub Repo Count', 'Other Links', 'City', 'State',
                                'Years of IT Experience', 'Years of Non-IT Experience',
                                'Highest Education Institute Name', 'Skills'
                            ] + SKILL_COLUMNS + [
                                'Total Projects Count', 'Internal Projects Count', 'External Projects Count',
                                'Internal Project Title', 'Internal Projects Techstacks',
                                'External Project Title', 'External Projects Techstacks',
                                'Latest Experience Company Name', 'Latest Experience Job Title',
                                'Latest Experience Start Date', 'Latest Experience End Date', 'Currently Working? (Yes/No)',
                                'Certifications', 'Awards', 'Achievements',
                            ]
                            display_columns = all_extraction_columns

                        process_resumes_in_batches_live(
                            df=df_input, batch_size=st.session_state.batch_size,
                            worker_function=process_resume_comprehensively,
                            display_columns=display_columns,
                            analysis_type=analysis_type,
                            company_name=company_name.strip()
                        )
                st.session_state.analysis_running = False

    if st.session_state.comprehensive_results:
        st.markdown("---")
        
        final_df = pd.DataFrame(st.session_state.comprehensive_results).fillna("")
        
        if st.session_state.last_analysis_mode == "shortlisting":
            prob_cols = ['Overall Probability', 'Projects Probability', 'Skills Probability', 'Experience Probability', 'Other Probability']
            numeric_cols = ['Total Projects Count', 'Internal Projects Count', 'External Projects Count']
            for col in prob_cols:
                if col in final_df.columns:
                    final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0)
            for col in numeric_cols:
                 if col in final_df.columns:
                    final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0).astype(int)
            
            base_display_cols = [
                'User ID', 'Resume Link', 'Overall Probability', 'Overall Remarks',
    'Projects Probability', 'Projects Remarks', 'Skills Probability', 'Skills Remarks',
    'Experience Probability', 'Experience Remarks', 'Other Probability', 'Other Remarks',
    'GitHub Profile Used', 
    'Commit Ownership Verified', # NEW COLUMN
    'GitHub Probability', 
    'Total Projects Count', 'Internal Projects Count', 'External Projects Count',
    'Internal Project Title', 'Internal Projects Techstacks',
    'External Project Title', 'External Projects Techstacks'
            ]

            if st.session_state.get('shortlisting_mode') == "Priority Wise (P1 / P2 / P3 Bands)":
                band_order = ['P1', 'P2', 'P3', 'Not Shortlisted']
                final_df['Priority Band'] = pd.Categorical(final_df['Priority Band'], categories=band_order, ordered=True)
                
                display_cols = base_display_cols.copy()
                display_cols.insert(3, 'Priority Band')
                display_cols.extend(['Company Name', 'Analysis Datetime'])

                final_df_ordered = final_df.sort_values(by=['Priority Band', 'Overall Probability'], ascending=[True, False])
            else:
                display_cols = base_display_cols.copy()
                display_cols.extend(['Company Name', 'Analysis Datetime'])
                
                final_df_ordered = final_df.sort_values(by='Overall Probability', ascending=False)
            
            final_df_ordered = final_df_ordered.reindex(columns=[col for col in display_cols if col in final_df_ordered.columns], fill_value='')
            file_name = f"resume_shortlist_{st.session_state.shortlisting_mode.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
        elif st.session_state.last_analysis_mode == "Internal Projects Matching":
            numeric_cols = ['Total Projects Count', 'Internal Projects Count', 'External Projects Count']
            for col in numeric_cols:
                if col in final_df.columns:
                    final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0).astype(int)
            final_column_order = [
                'User ID', 'Resume Link', 'Total Projects Count', 'Internal Projects Count', 'External Projects Count',
                'Internal Project Titles', 'Internal Project Techstacks', 'External Project Titles', 'External Project Techstacks',
                'Company Name', 'Analysis Datetime'
            ]
            final_df_ordered = final_df.reindex(columns=[col for col in final_column_order if col in final_df.columns], fill_value='')
            file_name = f"resume_analysis_{st.session_state.last_analysis_mode.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        else: # Comprehensive Modes
            final_df = coerce_probability_columns(final_df)
            base_cols = ['User ID', 'Resume Link', 'Full Name', 'Mobile Number', 'Email ID', 'LinkedIn Link', 'GitHub Link', 'GitHub Repo Count', 'Other Links', 'City', 'State', 'Years of IT Experience', 'Years of Non-IT Experience', 'Highest Education Institute Name', 'Skills']
            project_cols = ['Total Projects Count', 'Internal Projects Count', 'External Projects Count','Internal Project Title', 'Internal Projects Techstacks', 'External Project Title', 'External Projects Techstacks']
            exp_cols = ['Latest Experience Company Name', 'Latest Experience Job Title', 'Latest Experience Start Date', 'Latest Experience End Date', 'Currently Working? (Yes/No)']
            other_cols = ['Certifications', 'Awards', 'Achievements']
            edu_levels = ['Masters/Doctorate', 'Bachelors', 'Diploma', 'Intermediate / PUC / 12th', 'SSC / 10th']
            edu_fields = ['Name','Course Name', 'College Name', 'Department Name', 'Year of Completion', 'Percentage']
            edu_cols = [f"{level} {field}" for level in edu_levels for field in edu_fields]
            final_column_order = base_cols + SKILL_COLUMNS + project_cols + exp_cols + other_cols + edu_cols + ['Company Name', 'Analysis Datetime']
            final_column_order_filtered = [col for col in final_column_order if col in final_df.columns]
            final_df_ordered = final_df.reindex(columns=final_column_order_filtered, fill_value='')
            file_name = f"resume_analysis_{st.session_state.last_analysis_mode.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        st.subheader("Step 5: Filter, Review & Download Results")
        
        filtered_df = final_df_ordered.copy()

        with st.expander("🔍 Show Interactive Filters", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'Priority Band' in filtered_df.columns:
                    unique_bands = sorted(final_df_ordered['Priority Band'].cat.categories.tolist())
                    default_band = ['P1'] if 'P1' in unique_bands else []
                    selected_bands = st.multiselect('Filter by Priority Band:', options=unique_bands, default=default_band)
                    if selected_bands:
                        filtered_df = filtered_df[filtered_df['Priority Band'].isin(selected_bands)]
                    else:
                        filtered_df = filtered_df[filtered_df['Priority Band'].isin([])]
                
                searchable_cols = [col for col in ['Skills', 'Overall Remarks', 'Internal Project Title', 'External Project Title'] if col in filtered_df.columns]
                if searchable_cols:
                    search_term = st.text_input('Search text in key fields:', key='search_box')
                    if search_term:
                        search_series = filtered_df[searchable_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
                        filtered_df = filtered_df[search_series.str.contains(search_term, case=False, na=False)]

            with col2:
                if 'Overall Probability' in filtered_df.columns:
                    prob_range = st.slider('Filter by Overall Probability:', 0, 100, (0, 100))
                    if prob_range != (0, 100):
                        filtered_df = filtered_df[(filtered_df['Overall Probability'] >= prob_range[0]) & (filtered_df['Overall Probability'] <= prob_range[1])]
                
                if 'Skills Probability' in filtered_df.columns:
                    skills_prob_range = st.slider('Filter by Skills Probability:', 0, 100, (0, 100))
                    if skills_prob_range != (0, 100):
                        filtered_df = filtered_df[(filtered_df['Skills Probability'] >= skills_prob_range[0]) & (filtered_df['Skills Probability'] <= skills_prob_range[1])]

            with col3:
                st.write("Project Filters:")
                if 'Internal Projects Count' in filtered_df.columns:
                    has_internal = st.checkbox('Show only with Internal Projects')
                    if has_internal:
                        filtered_df = filtered_df[filtered_df['Internal Projects Count'] > 0]
                if 'External Projects Count' in filtered_df.columns:
                    has_external = st.checkbox('Show only with External Projects')
                    if has_external:
                        filtered_df = filtered_df[filtered_df['External Projects Count'] > 0]
                
                st.write("") 

                if 'Experience Probability' in filtered_df.columns:
                    exp_prob_range = st.slider('Filter by Experience Probability:', 0, 100, (0, 100))
                    if exp_prob_range != (0, 100):
                        filtered_df = filtered_df[(filtered_df['Experience Probability'] >= exp_prob_range[0]) & (filtered_df['Experience Probability'] <= exp_prob_range[1])]

            with col4:
                if 'Projects Probability' in filtered_df.columns:
                    proj_prob_range = st.slider('Filter by Projects Probability:', 0, 100, (0, 100))
                    if proj_prob_range != (0, 100):
                        filtered_df = filtered_df[(filtered_df['Projects Probability'] >= proj_prob_range[0]) & (filtered_df['Projects Probability'] <= proj_prob_range[1])]

                if 'Other Probability' in filtered_df.columns:
                    other_prob_range = st.slider('Filter by Other Probability:', 0, 100, (0, 100))
                    if other_prob_range != (0, 100):
                        filtered_df = filtered_df[(filtered_df['Other Probability'] >= other_prob_range[0]) & (filtered_df['Other Probability'] <= other_prob_range[1])]
            
            st.markdown("---")
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                if 'Total Projects Count' in filtered_df.columns:
                    min_total = st.number_input('Minimum Total Projects:', 0, 100, 0, 1)
                    if min_total > 0:
                        filtered_df = filtered_df[filtered_df['Total Projects Count'] >= min_total]
            with p_col2:
                 if 'Internal Projects Count' in filtered_df.columns:
                    min_internal = st.number_input('Minimum Internal Projects:', 0, 100, 0, 1)
                    if min_internal > 0:
                        filtered_df = filtered_df[filtered_df['Internal Projects Count'] >= min_internal]
            with p_col3:
                if 'External Projects Count' in filtered_df.columns:
                    min_external = st.number_input('Minimum External Projects:', 0, 100, 0, 1)
                    if min_external > 0:
                        filtered_df = filtered_df[filtered_df['External Projects Count'] >= min_external]

        st.info(f"Displaying **{len(filtered_df)}** of **{len(final_df_ordered)}** candidates.")
        
        st.dataframe(filtered_df)
        
        csv_buffer = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download {len(filtered_df)} Filtered Results as CSV", 
            data=csv_buffer, 
            file_name=f"filtered_{file_name}", 
            mime="text/csv"
        )

if __name__ == "__main__":
    if not MISTRAL_API_KEYS:
        st.error("Missing Mistral API keys. Add 'MISTRAL_API_KEY_1', 'MISTRAL_API_KEY_2', etc. to st.secrets.", icon="🚨")
    else:
        main()