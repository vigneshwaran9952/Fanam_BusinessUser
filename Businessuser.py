import os
import json
import time
import pandas as pd
from datetime import datetime
from random import uniform
import streamlit as st
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
import yagmail
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import bcrypt
import hashlib

# --- NEW IMPORTS FOR LIVE SEARCH ---
import asyncio
import httpx
import re
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
# -----------------------------------

# -------- Config --------
# Load environment variables (ensure .env file is present with keys)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY") # NEW
YAGMAIL_USER = os.getenv("YAGMAIL_USER")
YAGMAIL_PASS = os.getenv("YAGMAIL_PASS")

# --- Initialize Gemini for Live Search (Fast Init) ---
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_live_search = genai.GenerativeModel("gemini-2.0-flash-lite")
# ----------------------------------------

FULL_CONTENT_JSON = "merged_content.json"
SUMMARY_FAISS_PATH = "faiss_index_summary"
CONTENT_FAISS_PATH = "faiss_index_content"
LANGUAGE_CSV = "language_code.csv"
USER_FILE = "users.json"
NOTIFY_STATE_FILE = "last_data_state.txt"
HISTORY_FOLDER = "user_histories" # Folder for individual chat histories
USER_SETTINGS_FOLDER = "user_settings"

SIMILARITY_THRESHOLD = 0.75
MAX_CONTEXT_CHARS = 16000 
ALLOWED_COUNTRIES = {"India", "China", "US", "UAE"}
ALLOWED_SECTORS = {"Banking", "Finance", "Healthcare", "Environment"}

fallback_phrases = [
    "sorry, no detailed content found",
    "sorry, no relevant information",
    "no relevant information is available",
    "provided context does not contain",
    "i am sorry",
    "no relevant data found",
    "no detailed content found",
    "no information found"
]


# -------- Utility Functions --------

def clean_text(text: str) -> str:
    """Cleans up text by normalizing whitespace."""
    return ' '.join(text.split())

@st.cache_data(show_spinner=False)
def get_language_code(language):
    """Retrieves the language code for a given language name."""
    if not os.path.exists(LANGUAGE_CSV):
        return None
    df = pd.read_csv(LANGUAGE_CSV)
    if "Language" not in df.columns or "Code" not in df.columns:
        return None
    result = df[df["Language"].str.lower() == language.lower()]
    if result.empty:
        return None
    return result["Code"].values[0]

# --- MODIFIED: Save history now saves to a new file per session/topic ---
def save_history(entry, owner, timestamp):
    """Saves a single chat entry to a session-based history file."""
    if not os.path.exists(HISTORY_FOLDER):
        os.makedirs(HISTORY_FOLDER)
    
    # Use the provided timestamp (which could be the session ID or a history file ID)
    history_file = os.path.join(HISTORY_FOLDER, f"{owner}_{timestamp}.json")
    
    history_data = []
    # If the file exists, load it (for continuous conversation)
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            try:
                history_data = json.load(f)
            except:
                history_data = []
    
    history_data.append(entry)
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=4, ensure_ascii=False)

# --- NEW: Function to load a specific history file ---
def load_specific_history(filename):
    """Loads the entire chat history from a specific file."""
    history_file = os.path.join(HISTORY_FOLDER, filename)
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except:
                return []
    return []

# --- NEW: Function to list all history files for a user ---
def list_user_history_files(owner):
    """Lists all history files for a given user, sorted by date."""
    if not os.path.exists(HISTORY_FOLDER):
        return []
    
    # Files are named: username_YYYYMMDD_HHMMSS.json
    files = [f for f in os.listdir(HISTORY_FOLDER) if f.startswith(f"{owner}_") and f.endswith(".json")]
    
    # Sort files based on the timestamp in the filename (newest first)
    files.sort(key=lambda x: x.split('_')[1].split('.')[0], reverse=True)
    return files


# Optimized TF-IDF filtering function
def filter_content_by_summary_tfidf(top5_sum_docs, top5_cont_docs, top_n=5):
    """Ranks full content documents based on their similarity to the top summary documents using TF-IDF."""
    if not top5_sum_docs or not top5_cont_docs:
        return []
    
    sum_texts = [d.page_content for d in top5_sum_docs]
    cont_texts = [d.page_content for d in top5_cont_docs]
    
    vect = TfidfVectorizer().fit(sum_texts + cont_texts)
    sum_tfidf = vect.transform(sum_texts)
    cont_tfidf = vect.transform(cont_texts)
    
    sim_matrix = cosine_similarity(sum_tfidf, cont_tfidf)
    cont_scores = sim_matrix.mean(axis=0)
    
    scored = list(zip(top5_cont_docs, cont_scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    top = [doc for doc, score in scored[:top_n]]
    return top

def hash_password(password: str) -> str:
    """Hashes a password using bcrypt."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode(), salt)
    return hashed.decode()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a hashed password."""
    try:
        return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())
    except Exception:
        return False

def save_user_settings(username, country, sector, language):
    """Saves the current user settings to a dedicated file."""
    if not os.path.exists(USER_SETTINGS_FOLDER):
        os.makedirs(USER_SETTINGS_FOLDER)
    settings_file = os.path.join(USER_SETTINGS_FOLDER, f"{username}.json")
    data = {"country": country, "sector": sector, "language": language}
    with open(settings_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_user_settings(username):
    """Loads the current user settings."""
    settings_file = os.path.join(USER_SETTINGS_FOLDER, f"{username}.json")
    if os.path.exists(settings_file):
        with open(settings_file, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except:
                return None
    return None

def file_md5(filepath):
    """Calculates the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except FileNotFoundError:
        return None

def read_last_state():
    """Reads the last known state (MD5 hash) of the content files."""
    state = {}
    if not os.path.isfile(NOTIFY_STATE_FILE):
        return state
    with open(NOTIFY_STATE_FILE, "r") as f:
        for line in f:
            if ":" in line:
                key, value = line.strip().split(":", 1)
                state[key.strip()] = value.strip()
    return state

def write_last_state(state_dict):
    """Writes the current state (MD5 hash) of the content files."""
    with open(NOTIFY_STATE_FILE, "w") as f:
        for key, value in state_dict.items():
            f.write(f"{key}: {value}\n")

def send_email_notification(users_emails, country_sector_topics):
    """Sends email notification about data updates."""
    if not YAGMAIL_USER or not YAGMAIL_PASS:
        print("Yagmail credentials not configured")
        return
    try:
        yag = yagmail.SMTP(YAGMAIL_USER, YAGMAIL_PASS)
        subject = "Fanam Guard: New Compliance Data Added"
        countries = sorted({country.title() for country, _, _ in country_sector_topics})
        sectors = sorted({sector.title() for _, sector, _ in country_sector_topics})
        content = f"""
Hello,
New compliance data has been added to the Fanam Guard system.
Countries: {', '.join(countries)}
Sectors: {', '.join(sectors)}
Visit the app to explore the latest updates:
https://www.rbi.org.in/
Regards,
Fanam Guard Notification System
"""
        for user_email in users_emails: 
            if "@" in user_email and "." in user_email: 
                yag.send(to=user_email, subject=subject, contents=[content])
                print(f"Notification email sent to: {user_email}")
            else:
                 print(f"Skipping email for non-email-like username: {user_email}")

    except Exception as e:
        print(f"Failed to send email: {e}")

# -------- Cached Resources & Initialization (Moved below for better control) --------

@st.cache_resource
def load_llm():
    """Loads the Google Generative AI LLM."""
    # LLM is fast to initialize and used everywhere, so we keep it cached early.
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=GOOGLE_API_KEY, temperature=0.4)

@st.cache_resource
def load_embeddings():
    """Loads the HuggingFace Embeddings model."""
    # This is slow, will be called only after login.
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

# Helper function to load or build FAISS, used inside load_data_and_faiss
def load_or_build_faiss(path, docs, embedding):
    """Loads a FAISS index from disk or builds and saves a new one."""
    if os.path.exists(path):
        return FAISS.load_local(path, embedding, allow_dangerous_deserialization=True)
    if docs:
        vectordb = FAISS.from_documents(docs, embedding)
        vectordb.save_local(path)
        return vectordb
    return None # Return None if no documents

@st.cache_resource
def load_data_and_faiss(_embedding_model): 
    """Loads content data and initializes/loads all FAISS indices."""
    
    # --- 1. Load Merged Content (Filtered RAG) ---
    full_data = []
    summary_docs = []
    content_docs = []
    country_sector_topics = set()
    try:
        with open(FULL_CONTENT_JSON, "r", encoding="utf-8") as f:
            full_data = json.load(f)
    except FileNotFoundError:
        print(f"{FULL_CONTENT_JSON} not found. Filtered RAG will be empty.")
    except json.JSONDecodeError:
        print(f"Error decoding {FULL_CONTENT_JSON}. Filtered RAG will be empty.")
        
    for idx, item in enumerate(full_data):
        country = item.get("country", "").lower()
        sector = item.get("sector", "").lower()
        topics = item.get("topics", [])
        summary_docs.append(Document(page_content=clean_text(item.get("summary", "")),
                                     metadata={"index": idx, "country": country, "sector": sector}))
        content_docs.append(Document(page_content=clean_text(item.get("content", "")),
                                     metadata={"index": idx, "country": country, "sector": sector}))
        country_sector_topics.add((country, sector, tuple(topics)))

    # --- 2. Initialize/Load FAISS Indices ---
    summary_vectordb = load_or_build_faiss(SUMMARY_FAISS_PATH, summary_docs, _embedding_model) 
    content_vectordb = load_or_build_faiss(CONTENT_FAISS_PATH, content_docs, _embedding_model)

    return (full_data, summary_docs, content_docs, country_sector_topics, 
            summary_vectordb, content_vectordb) 

# --- GLOBAL VARIABLES (Only the fast ones remain here) ---
llm = load_llm() # Load LLM early as it's quick and needed for some core logic

# --- Prompt Setup ---

prompt_template = """
You are a professional Compliance Knowledge Assistant with expertise across India, the United States, the United Arab Emirates, and China.
You provide insights into compliance, regulations, legal frameworks, governance, and policy for the sectors: Banking, Finance, Healthcare, and Environment.

### Response Guidelines:
- Write in a ChatGPT-style tone — natural, confident, and clear.
- Use short, structured paragraphs or bullet points.
- Never mention or refer to any “document,” “context,” or “source text.”
- Expand acronyms on first use.
- Be factual, concise, and engaging.

{response_detail_instruction}

### Details:
- **Country:** {country}
- **Sector:** {sector}
- **Question:** {question}
- **Relevant Information:**
{context}

**Answer:**
"""
prompt = PromptTemplate(input_variables=["context", "question", "country", "sector", "response_detail_instruction"], template=prompt_template)

def safe_llm_invoke(prompt_text):
    """Invokes the LLM with a retry mechanism."""
    for _ in range(3):
        try:
            result = llm.invoke(prompt_text)
            if result and result.content.strip():
                return result.content
        except Exception:
            time.sleep(uniform(1.0, 2.5))
    return "No detailed content found to answer your query."


# --- User Management (Fast) ---
def load_users():
    """Loads all registered users from the user file."""
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}

def save_users(users):
    """Saves the user data to the user file."""
    with open(USER_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4, ensure_ascii=False)

def register_user(username, password, users):
    """Registers a new user."""
    if username in users:
        return False, "Username already exists"
    hashed_pass = hash_password(password)
    users[username] = {"password": hashed_pass} 
    save_users(users)
    return True, "Registration successful! Please login."
    
def login_user(username, password, users):
    """Logs in an existing user."""
    if username not in users:
        return False, "Invalid username or password"
    hashed_pass = users[username]["password"]
    if verify_password(password, hashed_pass):
        return True, f"Welcome {username}!"
    else:
        return False, "Invalid username or password"

users = load_users()

# ----------------------------------------------------
# --- Sector Research Agent (Live Search Logic) ---
# ----------------------------------------------------

# Helper: Gemini with Retry (Specific for the Live Search Agent)
def generate_with_retry_live(prompt, retries=5, initial_wait=2):
    wait = initial_wait
    for attempt in range(retries):
        try:
            response = gemini_live_search.generate_content(prompt)
            return response.text.strip()
        except ResourceExhausted:
            if attempt < retries - 1:
                # print(f"[WARNING] Resource exhausted. Retrying in {wait} seconds...")
                time.sleep(wait)
                wait *= 2
            else:
                return "API quota exceeded. Try again later."
        except Exception as e:
            # print("[ERROR]", e)
            return f"Error: {e}"

class SectorResearchAgent:

    def extract_urls(self, text: str) -> list:
        pattern = re.compile(r"http[s]?://[^\s'\"]+")
        return list(set(pattern.findall(text)))

    def search_google(self, query: str) -> list:
        if not SERPAPI_KEY:
            return []
        try:
            search = GoogleSearch({"q": query, "api_key": SERPAPI_KEY, "num": 5})
            results = search.get_dict()
            return [r["link"] for r in results.get("organic_results", [])[:5]]
        except Exception as e:
            print("[SEARCH ERROR]", e)
            return []

    async def scrape_urls(self, urls: list) -> str:
        headers = {"User-Agent": "Mozilla/5.0"}
        text_dump = ""
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            tasks = []
            for url in urls:
                tasks.append(self._fetch_and_parse(client, url, headers))
            
            results = await asyncio.gather(*tasks)
            text_dump = "\n".join(results)
        return text_dump

    async def _fetch_and_parse(self, client, url, headers):
        try:
            r = await client.get(url, headers=headers)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "iframe"]):
                    tag.decompose()
                text = soup.get_text(" ", strip=True)
                # Keep up to 15000 characters for chunking, minimum 500 characters
                if len(text) > 500:
                    return f"\n--- SOURCE: {url} ---\n{text[:15000]}"
        except Exception as e:
            # print(f"[SCRAPE ERROR] {url}: {e}")
            pass
        return ""

    async def reframe_question(self, country: str, sector: str, question: str) -> str:
        prompt = f"""
Reframe this question so it is explicitly related to BOTH the country and the sector.

Country: {country}
Sector: {sector}
Question: {question}

Return ONLY the rewritten question.
"""
        return generate_with_retry_live(prompt)

    async def summarize(self, raw_text_chunks: list, country: str, sector: str) -> str:
        combined_summaries = "\n---\n".join(raw_text_chunks)
        prompt = f"""
You are an expert in the sector and country mentioned.

Country: {country}
Sector: {sector}

Combine and refine the following scraped content into a single, structured, factual, clear answer. Structure the answer with bullet points or short paragraphs. Do not mention "scraped content" or "sources".

{combined_summaries}
"""
        return generate_with_retry_live(prompt)

    async def process(self, country: str, sector: str, question: str, lang_code: str):
        reframed = await self.reframe_question(country, sector, question)
        urls = self.search_google(reframed)
        scraped_text = await self.scrape_urls(urls)

        if not scraped_text or scraped_text.startswith("Error:"):
            return urls, "No live compliance data found or an error occurred during scraping."

        # Chunking scraped text for Gemini limits
        max_chunk_size = 15000
        chunks = [scraped_text[i:i+max_chunk_size] for i in range(0, len(scraped_text), max_chunk_size)]

        summary = await self.summarize(chunks, country, sector)
        
        if summary.startswith("API quota exceeded.") or summary.startswith("Error:"):
            return urls, summary

        # Translate summary to selected language
        if lang_code != "en":
            try:
                summary_translated = GoogleTranslator(source='auto', target=lang_code).translate(summary)
            except Exception as e:
                print("[TRANSLATE ERROR]", e)
                summary_translated = summary
        else:
            summary_translated = summary

        return urls, summary_translated
# ----------------------------------------------------
# --- END Sector Research Agent ---
# ----------------------------------------------------


# -------- Streamlit UI and Logic --------
st.set_page_config(page_title="Business user", layout="centered", page_icon="images/AI_1.png")
st.markdown("""
        <style>
        .dark-green-text {
            text-align: center;
            font-size: 48px;
            font-weight: bold;
            color: #00325b;
        }
        </style>
        <h1 class="dark-green-text">Fanam Guard</h1>
        """, unsafe_allow_html=True)

# --- Session State Initialization ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "selected_country" not in st.session_state:
    st.session_state.selected_country = None
if "selected_sector" not in st.session_state:
    st.session_state.selected_sector = None
if "language" not in st.session_state:
    st.session_state.language = None
if "settings_saved" not in st.session_state:
    st.session_state.settings_saved = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_session_id" not in st.session_state: # Unique ID for the current chat session
    st.session_state.chat_session_id = datetime.now().strftime("%Y%m%d_%H%M%S") 
if "temp_sum_idx" not in st.session_state:
    st.session_state.temp_sum_idx = None
if "temp_cont_idx" not in st.session_state:
    st.session_state.temp_cont_idx = None
if "current_faiss_key" not in st.session_state:
    st.session_state.current_faiss_key = None
if "enhance_mode" not in st.session_state:
    st.session_state.enhance_mode = False
if "live_search_mode" not in st.session_state: # Live Search Mode Toggle
    st.session_state.live_search_mode = False
if "current_history_file" not in st.session_state: # To track selected history in sidebar
    st.session_state.current_history_file = None

# --- RAG DATA SESSION STATE ---
if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = False
if "embedding" not in st.session_state:
    st.session_state.embedding = None
if "summary_docs" not in st.session_state:
    st.session_state.summary_docs = None
if "content_docs" not in st.session_state:
    st.session_state.content_docs = None
if "country_sector_topics" not in st.session_state:
    st.session_state.country_sector_topics = None
if "summary_vectordb" not in st.session_state:
    st.session_state.summary_vectordb = None
if "content_vectordb" not in st.session_state:
    st.session_state.content_vectordb = None


# --- Authentication Logic ---

# MODIFIED to use session state variables for RAG data
def update_filtered_faiss(country, sector):
    """Builds and caches the country/sector specific FAISS indices."""
    if not st.session_state.rag_initialized:
        return False # Cannot filter if RAG is not loaded

    country = country.lower()
    sector = sector.lower()
    faiss_key = f"{country}_{sector}"
    
    # Check if the desired index is already in the cache
    if st.session_state.current_faiss_key == faiss_key and st.session_state.temp_sum_idx and st.session_state.temp_cont_idx:
        return True

    # Filter documents
    summary_docs = st.session_state.summary_docs
    content_docs = st.session_state.content_docs
    
    filtered_summaries = [d for d in summary_docs if d.metadata.get("country", "") == country and d.metadata.get("sector", "") == sector]
    filtered_contents = [d for d in content_docs if d.metadata.get("country", "") == country and d.metadata.get("sector", "") == sector]

    if filtered_summaries and filtered_contents:
        # Build temporary FAISS indices for the filtered subset 
        # Use the embedding model loaded into session state
        embedding = st.session_state.embedding 
        st.session_state.temp_sum_idx = FAISS.from_documents(filtered_summaries, embedding)
        st.session_state.temp_cont_idx = FAISS.from_documents(filtered_contents, embedding)
        st.session_state.current_faiss_key = faiss_key
        return True
    
    # Clear cache if no data found for the selection
    st.session_state.temp_sum_idx = None
    st.session_state.temp_cont_idx = None
    st.session_state.current_faiss_key = None
    return False

def load_settings_on_login(username):
    # RAG components must be loaded before calling update_filtered_faiss
    if not st.session_state.rag_initialized:
        return 

    settings = load_user_settings(username)
    if settings:
        st.session_state.selected_country = settings.get("country", None)
        st.session_state.selected_sector = settings.get("sector", None)
        st.session_state.language = settings.get("language", None)
        st.session_state.settings_saved = True
        if settings.get("country") and settings.get("sector"):
            update_filtered_faiss(settings["country"], settings["sector"])
    else:
        # Set defaults if no settings found
        st.session_state.selected_country = list(ALLOWED_COUNTRIES)[0]
        st.session_state.selected_sector = list(ALLOWED_SECTORS)[0]
        st.session_state.language = "English"
        st.session_state.settings_saved = False


# --- Authentication UI (Loads immediately) ---
if not st.session_state.authenticated:
    if "login_info_shown" not in st.session_state or not st.session_state.login_info_shown:
        st.toast("Please Login to continue...", icon="ℹ️", duration="short")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_btn = st.form_submit_button("Login")
        if login_btn:
            success, msg = login_user(username, password, users)
            if success:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.toast(msg, icon="✅")
                st.session_state.login_info_shown = False
                st.session_state.chat_history = [] 
                st.session_state.chat_session_id = datetime.now().strftime("%Y%m%d_%H%M%S") 
                st.session_state.current_history_file = None 
                
                # NOTE: Settings will be loaded *after* RAG initialization in the next rerun
                st.rerun()
            else:
                st.error(msg)
    with tab2:
        with st.form("register_form"):
            reg_user = st.text_input("New Username")
            reg_pass = st.text_input("New Password", type="password")
            register_btn = st.form_submit_button("Register")
        if register_btn:
            if not reg_user or not reg_pass:
                st.warning("Please fill all fields")
            else:
                success, msg = register_user(reg_user, reg_pass, users)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
    st.stop()


# --- Main App Content (After Authentication) ---

# ***************************************************************
# --- RAG INITIALIZATION (Delayed until after successful login) ---
# ***************************************************************

if not st.session_state.rag_initialized:
    # This block now runs only once, after login, and delays the UI render
    with st.spinner("Loading compliance knowledge base (This may take a moment)..."):
        try:
            # 1. Load Embeddings (The slowest single step)
            embedding_model = load_embeddings()
            st.session_state.embedding = embedding_model
            
            # 2. Load Data and FAISS Indices
            (full_data, summary_docs, content_docs, country_sector_topics, 
             summary_vectordb, content_vectordb) = load_data_and_faiss(embedding_model)
            
            # 3. Store in Session State
            st.session_state.full_data = full_data
            st.session_state.summary_docs = summary_docs
            st.session_state.content_docs = content_docs
            st.session_state.country_sector_topics = country_sector_topics
            st.session_state.summary_vectordb = summary_vectordb
            st.session_state.content_vectordb = content_vectordb
            st.session_state.rag_initialized = True
            
            # 4. Load User Settings now that RAG components are available
            load_settings_on_login(st.session_state.username)
            
            st.rerun() # Rerun to display the main interface
            
        except Exception as e:
            st.error(f"Failed to initialize RAG components: {e}")
            st.stop()
            
# If RAG is initialized, we can access the components from session state:
embedding = st.session_state.embedding
summary_docs = st.session_state.summary_docs
content_docs = st.session_state.content_docs
country_sector_topics = st.session_state.country_sector_topics
summary_vectordb = st.session_state.summary_vectordb
content_vectordb = st.session_state.content_vectordb

# ***************************************************************
# --- END RAG Initialization ---
# ***************************************************************

st.markdown("""
    <style>
            /* Set the sidebar width to be visually appealing like the video */
        [data-testid="stSidebar"] {
            width: 300px !important;
        }
        /* Style for the history link buttons in the sidebar */
        .history-link {
            text-align: left;
            margin-bottom: 5px;
            padding: 8px 10px;
            border-radius: 5px;
            cursor: pointer;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-size: 14px;
            font-weight: 500;
            color: #333333; /* Dark text for better readability */
            background-color: #f0f2f6; /* Light gray background */
            border: 1px solid #e0e0e0;
            transition: background-color 0.2s;
        }
        .history-link:hover {
            background-color: #e0e0e0; /* Slightly darker on hover */
        }
        .history-link.active {
            background-color: #00325b; /* Dark green/blue for active */
            color: white;
            font-weight: bold;
        }
        /* Custom box for Live Search Mode */
        .live-search-box {
            background-color: #ffe4e1; /* Light red/pink for alert */
            border: 2px solid #ff4b4b;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 15px;
            font-weight: bold;
            color: #8b0000;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    # Function to handle New Chat button click
    def start_new_chat():
        st.session_state.chat_history = []
        st.session_state.enhance_mode = False # Reset enhance mode on new chat
        st.session_state.live_search_mode = False # Reset live search mode on new chat
        st.session_state.chat_session_id = datetime.now().strftime("%Y%m%d_%H%M%S") # NEW ID
        st.session_state.current_history_file = None # Ensure history view is cleared

    # Function to handle history item click
    def load_old_chat(filename):
        # 1. Load the history from the file
        history_list = load_specific_history(filename)
        
        # 2. Update session state for chat display and session ID
        st.session_state.current_history_file = filename
        
        # Convert the dictionary history format into the (role, message) tuple format for display
        display_history = []
        for h in history_list:
            if "question" in h and "answer" in h:
                display_history.append(("user", h["question"]))
                display_history.append(("assistant", h["answer"]))
            
        st.session_state.chat_history = display_history
        
        # 3. Set the current chat session ID to the filename's timestamp 
        try:
            timestamp_part = filename.split('_')[-1].replace('.json', '')
            st.session_state.chat_session_id = timestamp_part
        except:
            st.session_state.chat_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    
    # Function to handle logout
    def logout_user():
        for key in list(st.session_state.keys()):
            if key not in ['authenticated', 'username', 'login_info_shown', 'rag_initialized', 'embedding', 'summary_docs', 'content_docs', 'country_sector_topics', 'summary_vectordb', 'content_vectordb', 'full_data', 'llm']: 
                del st.session_state[key]
        st.session_state.authenticated = False
        st.session_state.username = None

    # 1. Controls (New Chat, Logout)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("New Chat", on_click=start_new_chat, use_container_width=True):
            st.rerun() # Trigger the rerun immediately
    
    with col2:
        if st.button("Logout", use_container_width=True, on_click=logout_user):
            st.rerun() # Trigger the rerun immediately
            
    st.markdown("---") # Visual separator

    # 2. Mode Toggles 
    # Only show toggles if settings are saved AND we are in a NEW chat (not viewing old history)
    if st.session_state.settings_saved and st.session_state.current_history_file is None:
        
        # --- Enhance Mode Toggle ---
        def toggle_enhance_mode_callback():
            """Callback function for the st.toggle, handles toasts."""
            if st.session_state.enhance_mode:
                st.toast("Enhance Mode Activated (Responses will be more detailed)", icon="🔍")
            else:
                st.toast("Enhance Mode DeActivated (Responses will be short and concise)", icon="⚠️")
        
        st.toggle(
            "Enhance Mode",
            key="enhance_mode",
            value=st.session_state.enhance_mode,
            on_change=toggle_enhance_mode_callback,
        )
        
        # --- Live Search Mode Toggle (NEW) ---
        def toggle_live_search_callback():
            """Callback function for the st.toggle, handles toasts."""
            if st.session_state.live_search_mode:
                st.toast("Deep reSearch Mode Activated ", icon="📡")
            else:
                st.toast("Deep research Mode DeActivated ", icon="💾")
        
        st.toggle(
            "Deep Research", # The new toggle box
            key="live_search_mode",
            value=st.session_state.live_search_mode,
            on_change=toggle_live_search_callback,
        )
    
    st.markdown("---") # Visual separator
    
    # 3. Settings Expander
    with st.expander("Settings", expanded=True):
        with st.form("settings"):
            country_list = sorted(list(ALLOWED_COUNTRIES))
            sector_list = sorted(list(ALLOWED_SECTORS))
            
            country_index = country_list.index(st.session_state.selected_country) if st.session_state.selected_country in country_list and st.session_state.selected_country is not None else 0
            selected_country = st.selectbox("Country", country_list, index=country_index)
            
            sector_index = sector_list.index(st.session_state.selected_sector) if st.session_state.selected_sector in sector_list and st.session_state.selected_sector is not None else 0
            selected_sector = st.selectbox("Sector", sector_list, index=sector_index)
            
            if os.path.exists(LANGUAGE_CSV):
                lang_df = pd.read_csv(LANGUAGE_CSV)
                languages = lang_df["Language"].dropna().unique().tolist()
            else:
                languages = ["English"]
            
            language_index = languages.index(st.session_state.language) if st.session_state.language in languages and st.session_state.language is not None else 0
            language = st.selectbox("Select Language:", languages, index=language_index)
            
            save_button = st.form_submit_button("Save")
            
            if save_button:
                st.session_state.selected_country = selected_country
                st.session_state.selected_sector = selected_sector
                st.session_state.language = language
                
                if st.session_state.rag_initialized and update_filtered_faiss(selected_country, selected_sector):
                    st.session_state.settings_saved = True
                    save_user_settings(st.session_state.username, selected_country, selected_sector, language)
                    st.toast("Settings saved! You can now chat.", icon="✅")
                else:
                    st.session_state.settings_saved = False
                    st.warning("No data found for this Country/Sector combination. Please select a different one.")
                st.rerun() 

    st.markdown("---") # Visual separator

    # 4. History List
    with st.expander("History ", expanded=False):
        if st.session_state.username:
            history_files = list_user_history_files(st.session_state.username)
            
            if not history_files:
                st.markdown("<p style='color:gray; font-size:12px;'>No past conversations.</p>", unsafe_allow_html=True)
            else:
                # Display only the last 50 files/conversations
                for file in history_files[:50]:
                    
                    # Get a title from the conversation file
                    history_data = load_specific_history(file)
                    if history_data:
                        # Use the first question as the title
                        title = history_data[0].get("question", "Untitled Conversation")
                    else:
                        # Fallback title with timestamp
                        time_str = file.split('_')[-1].replace('.json', '')
                        title = f"Chat ({time_str})"

                    is_active = file == st.session_state.current_history_file
                    
                    # Use a standard button for the history link
                    if st.button(title, key=f"history-link-{file}", on_click=load_old_chat, args=(file,), use_container_width=True):
                        # Force a rerun to load the chat data immediately
                        st.rerun()


# --- Content Update Check and Rebuild Logic (MODIFIED) ---
# Only run this check if RAG components are loaded
if st.session_state.rag_initialized:
    last_state = read_last_state()
    current_state = {}
    needs_rebuild = False
    
    # Check FULL_CONTENT_JSON
    md5_full = file_md5(FULL_CONTENT_JSON)
    if md5_full:
        current_state[FULL_CONTENT_JSON] = md5_full
        if current_state[FULL_CONTENT_JSON] != last_state.get(FULL_CONTENT_JSON):
            needs_rebuild = True
    
    if needs_rebuild:
        st.toast("Compliance data updated. Rebuilding indices...", icon="🔄")
        
        # Invalidate cache for content/FAISS and reload them
        load_data_and_faiss.clear()
        
        # Invalidate RAG initialization state to force a reload of all RAG components
        st.session_state.rag_initialized = False 
        
        # Notify users about updates
        emails = [u for u in users.keys()] 
        send_email_notification(emails, st.session_state.country_sector_topics) # Use the old topics before reload
        write_last_state(current_state)
        
        st.rerun() 

# --- Chat Display / History View ---

# Display the conversation history
for role, message in st.session_state.chat_history:
    avatar_icon = "images/AI_1.png" if role == "assistant" else "user"
    with st.chat_message(role, avatar=avatar_icon):
        st.markdown(message, unsafe_allow_html=True)

# --- Chat Input (RAG or Live Search) ---
if not st.session_state.settings_saved:
    st.info("Please select your **Country**, **Sector**, and **Language** in **Settings** and click **Save** before chatting.")
    st.chat_input("Ask anything (Settings not saved)", disabled=True) 
else:
    
    # Display Live Search warning box if activated
    if st.session_state.live_search_mode:
        st.markdown('<div class="live-search-box">📡 **Deep research Mode is ON:** Answers are generated from live web data. This may be slower than internal search.</div>', unsafe_allow_html=True)

    user_input = st.chat_input("Ask anything ", key="main_chat_input")
    
    if user_input:
        current_question = user_input
        
        if not current_question.strip():
            st.stop()
        
        # Append user message to display
        st.session_state.chat_history.append(("user", current_question))
        with st.chat_message("user"):
            st.markdown(current_question)

        with st.spinner("Thinking..."):
            
            # --- Initialization ---
            english_answer = ""
            final_answer = ""
            question_en = current_question # Default English question
            urls = [] # For saving URLs from live search
            
            # --- Define lang_code and settings ---
            lang_code = get_language_code(st.session_state.language) if st.session_state.language else None
            country = st.session_state.selected_country.lower() if st.session_state.selected_country else ""
            sector = st.session_state.selected_sector.lower() if st.session_state.selected_sector else ""
            
            country_display = st.session_state.selected_country.upper()
            sector_display = st.session_state.selected_sector.title()

            
            # --- Determine Search Mode ---
            if st.session_state.live_search_mode:
                
                # ----------------------------------------------------
                # --- LIVE SEARCH MODE (SectorResearchAgent) ---
                # ----------------------------------------------------
                if not SERPAPI_KEY or not GOOGLE_API_KEY:
                    final_answer = "Live Search Mode requires SERPAPI_KEY and GOOGLE_API_KEY in the environment variables."
                    urls = []
                else:
                    agent = SectorResearchAgent()
                    urls, english_answer = asyncio.run(agent.process(
                        st.session_state.selected_country, 
                        st.session_state.selected_sector, 
                        current_question, 
                        lang_code
                    ))
                    search_type = "Live Search"
                    
                    # In Live Search Mode, the agent handles the final translation
                    final_answer = english_answer 
                    is_fallback = "no live compliance data found" in final_answer.lower() or "error:" in final_answer.lower()
                
            else:
                
                # ----------------------------------------------------
                # --- FILTERED RAG MODE (Existing Logic) ---
                # ----------------------------------------------------
                
                # 1. Translate Question (if needed)
                try:
                    if st.session_state.language.lower() != "english":
                        lang_code_trans = get_language_code(st.session_state.language)
                        if lang_code_trans and lang_code_trans != "en":
                            question_en = GoogleTranslator(source="auto", target="en").translate(current_question)
                    else:
                        question_en = current_question 
                except Exception:
                    question_en = current_question

                # 2. Search Logic (Filtered RAG)
                top_docs = []
                search_type = "Filtered RAG (Country/Sector)"
                temp_sum_idx = st.session_state.temp_sum_idx
                temp_cont_idx = st.session_state.temp_cont_idx

                if not temp_sum_idx or not temp_cont_idx:
                    english_answer = f"Internal content indices for {country_display} - {sector_display} are not initialized. Please re-save your settings."
                else:
                    # Search the cached FAISS indices
                    retriever = temp_sum_idx.as_retriever(search_kwargs={"k": 5})
                    top5_sum_docs = retriever.invoke(question_en)

                    retriever_cont = temp_cont_idx.as_retriever(search_kwargs={"k": 5})
                    top5_cont_docs = retriever_cont.invoke(question_en)

                    # Refine content documents based on summary relevance
                    top_docs = filter_content_by_summary_tfidf(top5_sum_docs, top5_cont_docs, top_n=5)
                
                # --- Response Instruction based on Enhance Mode ---
                if st.session_state.enhance_mode:
                    response_instruction = "- Provide a **comprehensive and detailed** answer, summarizing the most important points from the Relevant Information. Keep the answer to a **medium-to-long length**."
                else:
                    response_instruction = "- Provide a **very concise and brief** answer, focusing only on the primary point. Keep the answer to a **short length**."
                    
                # 3. Process Search Results and Invoke LLM
                if not english_answer and top_docs:
                    context = "\n\n".join(d.page_content for d in top_docs)
                    
                    # 4. Format and Invoke LLM 
                    full_prompt = prompt.format(
                        context=context, 
                        question=question_en,
                        country=country_display,
                        sector=sector_display,
                        response_detail_instruction=response_instruction,
                    )
                    english_answer = safe_llm_invoke(full_prompt)
                elif not english_answer:
                    english_answer = f"No relevant compliance data found using {search_type} for your query."

                # 5. Check for fallback and apply RAG guardrail
                is_fallback = any(phrase in (english_answer or "").lower() for phrase in fallback_phrases)

                if is_fallback:
                    # Guardrail: Ask the user to refine their query based on selected settings
                    final_answer = (
                        f"I couldn't find specific compliance details for your question based on the selected **Country** "
                        f"({country_display}) and **Sector** ({sector_display}). "
                        f"Please ask a question more closely related to **{sector_display}** regulations in **{country_display}**."
                    )
                else:
                    # 6. Translate Answer (if needed for RAG)
                    if english_answer and lang_code != "en":
                        try:
                            final_answer = GoogleTranslator(source="en", target=lang_code).translate(english_answer)
                        except Exception:
                            final_answer = english_answer
                    else:
                        final_answer = english_answer
                
            # --- Shared Steps (Save History & Display) ---
            if not is_fallback:
                # 7. Save History (Use the current session ID)
                if st.session_state.username:
                    history_entry = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "country": country_display,
                        "sector": sector_display,
                        "language": st.session_state.language,
                        "question": current_question,
                        "translated_question": question_en,
                        "answer": final_answer,
                        "search_type": search_type,
                        "urls": urls # Save URLs if Live Search was used
                    }
                    # Save the entry to the file referenced by st.session_state.chat_session_id
                    save_history(history_entry, st.session_state.username, st.session_state.chat_session_id)


            # 8. Display Final Answer
            with st.chat_message("assistant", avatar="images/AI_1.png"):
                
                if urls: # Display sources if live search was used
                    url_list = "\n".join([f"1. [{u.split('/')[2]}]({u})" for u in urls if u])
                    st.markdown(f"### 🌐 Sources ({search_type}):\n{url_list}\n\n---")
                    
                st.markdown(final_answer or "No response generated.", unsafe_allow_html=True)
                
            # 9. Update Session History (Append assistant message for display)
            st.session_state.chat_history.append(("assistant", final_answer or "No response generated."))