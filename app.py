"""
VibeVisualizer - Movie Discovery Engine (Full Code with Vector DB & Web Fallback Cascade)
"""

import os

# ---------------------------------------------------------
# PRE-IMPORT THREAD & CPU OPTIMIZATION
# ---------------------------------------------------------
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

# Limit CPU threads to prevent CPU thrashing during initial load
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import re
import json
import urllib.parse
import requests
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

# ---------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
load_dotenv()

@st.cache_resource(show_spinner="⚡ Initializing Search Engine...")
def load_embedding_model():
    import torch
    torch.set_num_threads(2)
    torch.set_num_threads(2)  # Adjust CPU core usage
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


get_embedding_model = load_embedding_model

@st.cache_resource(show_spinner="📂 Loading 10k Movie Vector Index...")
def load_vector_db():
    index = None
    metadata = []
    
    if os.path.exists("movie_index.faiss"):
        index = faiss.read_index("movie_index.faiss")
        
    if os.path.exists("movie_metadata.json"):
        with open("movie_metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
    return index, metadata

# Initialize cached references (Loaded into RAM once, shared across reloads)
EMBEDDING_MODEL = load_embedding_model()
FAISS_INDEX, MOVIE_METADATA = load_vector_db()

def get_secret(key: str, default=None):
    return os.getenv(key) or (st.secrets.get(key) if hasattr(st, "secrets") else None) or default

GROQ_API_KEY = get_secret("GROQ_API_KEY")
TMDB_API_KEY = get_secret("TMDB_API_KEY")
GOOGLE_SEARCH_API_KEY = get_secret("GOOGLE_SEARCH_API_KEY")
GOOGLE_CSE_ID = get_secret("GOOGLE_CSE_ID")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"
MEMORY_FILE = "user_vibe_memory.json"
INDEX_FILE = "movie_index.faiss"
METADATA_FILE = "movie_metadata.json"

# Strict L2 similarity threshold (0.35 L2 = ~0.94 Cosine Similarity)
DISTANCE_THRESHOLD = 0.35  

# Thread-safe writeback lock
INDEX_MUTEX = threading.Lock()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json"
}

@st.cache_resource
def get_http_session():
    """Creates and caches a persistent HTTP session with connection pooling."""
    sess = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
    sess.mount("https://", adapter)
    sess.headers.update(HEADERS)
    return sess

session = get_http_session()

st.set_page_config(page_title="VibeVisualizer", page_icon="🎬", layout="wide")

# Custom UI Styling
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa !important;
    }
    
    .hero-container {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }

    .vibe-badge {
        background-color: #e50914 !important;
        color: #ffffff !important;
        font-weight: bold;
        font-size: 11px;
        padding: 3px 8px;
        border-radius: 4px;
        display: inline-block;
        margin-bottom: 6px;
    }

    .genre-badge {
        background-color: #f1f5f9 !important;
        color: #334155 !important;
        font-size: 11px;
        font-weight: 600;
        padding: 3px 7px;
        border-radius: 4px;
        display: inline-block;
        margin-right: 4px;
        margin-bottom: 4px;
        border: 1px solid #cbd5e1;
    }

    .card-title {
        font-size: 14px;
        font-weight: bold;
        height: 40px;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        margin-top: 6px;
        margin-bottom: 4px;
    }

    .card-meta {
        font-size: 12px;
        color: #475569 !important;
        margin-top: 4px;
        margin-bottom: 6px;
        line-height: 1.4;
        height: 36px;
    }

    .card-desc {
        font-size: 12px;
        color: #334155 !important;
        background-color: #f8fafc !important;
        border: 1px solid #e2e8f0;
        padding: 8px;
        border-radius: 6px;
        margin-top: 6px;
        margin-bottom: 8px;
        line-height: 1.35;
        height: 65px;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_groq_client():
    if not GROQ_API_KEY:
        return None
    try:
        return Groq(api_key=GROQ_API_KEY)
    except Exception:
        return None

# ---------------------------------------------------------
# 1.5 FAISS VECTOR DB & EMBEDDINGS INIT
# ---------------------------------------------------------


EMBEDDING_MODEL = load_embedding_model()

@st.cache_resource
def load_vector_assets():
    """Loads FAISS index and metadata into memory or initializes new ones."""
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        try:
            index = faiss.read_index(INDEX_FILE)
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            return index, metadata
        except Exception:
            pass
    # Initialize 384-dim normalized L2 index if missing/corrupt
    index = faiss.IndexFlatL2(384)
    return index, {}

FAISS_INDEX, VECTOR_METADATA = load_vector_assets()

def append_to_vector_db_mutex(movie_obj: dict):
    """Safely adds a movie embedding to RAM index and persists to disk under lock."""
    if not movie_obj or not isinstance(movie_obj, dict) or "id" not in movie_obj:
        return

    m_id_str = str(movie_obj["id"])
    if m_id_str in VECTOR_METADATA.values():
        return

    vibe_str = movie_obj.get('vibe_reason', '')
    doc = f"{movie_obj.get('title', '')} {movie_obj.get('genres', '')} {movie_obj.get('overview', '')} {vibe_str}".strip()
    vec = EMBEDDING_MODEL.encode([doc], convert_to_numpy=True)
    faiss.normalize_L2(vec)

    with INDEX_MUTEX:
        new_faiss_id = FAISS_INDEX.ntotal
        FAISS_INDEX.add(vec)
        VECTOR_METADATA[str(new_faiss_id)] = movie_obj

        try:
            faiss.write_index(FAISS_INDEX, INDEX_FILE)
            with open(METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump(VECTOR_METADATA, f, indent=2)
        except Exception:
            pass


# ---------------------------------------------------------
# 2. TMDB CACHE & FORMATTING
# ---------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def get_tmdb_genre_map():
    if not TMDB_API_KEY:
        return {}
    try:
        res = session.get(
            f"{TMDB_BASE}/genre/movie/list",
            params={"api_key": TMDB_API_KEY},
            timeout=5.0
        )
        if res.status_code == 200:
            return {item["id"]: item["name"] for item in res.json().get("genres", [])}
    except Exception:
        pass
    return {}

GENRE_MAP = get_tmdb_genre_map()

LANGUAGE_OPTIONS = [
    "All Languages", "Telugu", "Hindi", "Tamil", "Malayalam", "Kannada", "English", "Korean", "Japanese"
]

INDUSTRY_OPTIONS = [
    "🌟 All Cinema Regions",
    "🎬 Hollywood (Western Cinema)",
    "🇮🇳 Indian Cinema (All Regional)",
    "💥 Bollywood (Hindi Cinema)",
    "🔥 South Indian Cinema",
    "🌏 East Asian Cinema",
    "🌐 World Cinema"
]

# Session State Initialization
if "search_results" not in st.session_state:
    st.session_state["search_results"] = None
if "searched_movie_info" not in st.session_state:
    st.session_state["searched_movie_info"] = None
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""
if "selected_movie" not in st.session_state:
    st.session_state["selected_movie"] = None

# Memory Helper Functions
def init_memory():
    if "search_history" not in st.session_state:
        st.session_state["search_history"] = []
    if "favorite_movies" not in st.session_state:
        st.session_state["favorite_movies"] = []
    if "vibe_preference" not in st.session_state:
        st.session_state["vibe_preference"] = "All Vibes"

    if os.path.exists(MEMORY_FILE) and not st.session_state["search_history"]:
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                st.session_state["search_history"] = json.load(f)
        except Exception:
            st.session_state["search_history"] = []

def save_search_to_memory(query: str):
    if query and query.strip():
        q_clean = query.strip()
        if not st.session_state["search_history"] or st.session_state["search_history"][0] != q_clean:
            st.session_state["search_history"].insert(0, q_clean)
            st.session_state["search_history"] = st.session_state["search_history"][:8]
            try:
                with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                    json.dump(st.session_state["search_history"], f, indent=2)
            except Exception:
                pass

def toggle_favorite(movie: dict):
    favs = st.session_state["favorite_movies"]
    existing_ids = [m["id"] for m in favs if isinstance(m, dict) and "id" in m]
    
    if movie["id"] in existing_ids:
        st.session_state["favorite_movies"] = [m for m in favs if m.get("id") != movie["id"]]
    else:
        st.session_state["favorite_movies"].append(movie)

def clear_search_history():
    st.session_state["search_history"] = []
    if os.path.exists(MEMORY_FILE):
        try:
            os.remove(MEMORY_FILE)
        except Exception:
            pass

def clear_favorites():
    st.session_state["favorite_movies"] = []

def clean_movie_title(raw_title: str) -> str:
    """Removes years like (2016), quotes, and extra noise from title strings."""
    if not raw_title:
        return ""
    cleaned = re.sub(r'[\"\']', '', raw_title)
    cleaned = re.sub(r'\(\s*\d{4}\s*\)', '', cleaned)
    return cleaned.strip()

def format_tmdb_movie_item(item: dict, target_language: str = "All Languages") -> dict:
    if not item or not isinstance(item, dict) or not item.get("id"):
        return None

    # Drop movie if TMDB has no poster available
    poster_path = item.get("poster_path")
    if not poster_path:
        return None

    orig_lang = (item.get("original_language") or "").lower()
    movie_id = item.get("id")
    release_year = (item.get("release_date") or "")[:4] or "N/A"

    genre_ids = item.get("genre_ids", [])
    genre_names = [GENRE_MAP.get(gid) for gid in genre_ids if gid in GENRE_MAP][:2]
    genres_str = " • ".join(genre_names) if genre_names else "Drama • Action"

    lang_names = {
        "te": "Telugu", "hi": "Hindi", "ta": "Tamil", 
        "ml": "Malayalam", "kn": "Kannada", "en": "English", 
        "ko": "Korean", "ja": "Japanese"
    }
    audio_str = f"{lang_names.get(orig_lang, orig_lang.upper() if orig_lang else 'Unknown')} (Original)"

    if target_language != "All Languages" and target_language.lower() not in audio_str.lower():
        audio_str += f", {target_language} Dub"

    overview_text = item.get("overview", "")
    if not overview_text or len(overview_text) < 10:
        overview_text = "A dramatic cinematic feature."

    title_name = item.get("title") or item.get("original_title") or "Untitled"

    return {
        "id": movie_id,
        "title": title_name,
        "year": release_year,
        "rating": round(item.get("vote_average", 0.0), 1),
        "poster": f"{TMDB_IMG}{poster_path}",
        "overview": overview_text,
        "audio": audio_str,
        "subtitles": "English, Multi-Sub",
        "genres": genres_str,
        "orig_lang": orig_lang,
        "popularity": item.get("popularity", 0.0)
    }

# ---------------------------------------------------------
# 3. TMDB SEARCH & RECOMMENDATIONS
# ---------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_single_movie_cached(title: str, target_language: str):
    """Sanitizes query and fetches accurate TMDB movie metadata."""
    clean_q = clean_movie_title(title)
    if not TMDB_API_KEY or not clean_q:
        return None

    try:
        res = session.get(
            f"{TMDB_BASE}/search/movie",
            params={"api_key": TMDB_API_KEY, "query": clean_q, "include_adult": False},
            timeout=5.0
        )
        if res.status_code == 200:
            results = res.json().get("results", [])
            if results:
                q_lower = clean_q.lower()
                exact = next((m for m in results if (m.get("title") or "").lower() == q_lower or (m.get("original_title") or "").lower() == q_lower), None)
                chosen = exact if exact else max(results[:3], key=lambda x: x.get("popularity", 0))
                return format_tmdb_movie_item(chosen, target_language)
    except Exception:
        pass

    return None

@st.cache_data(ttl=86400, show_spinner=False)
def get_movie_trailer_url(movie_id: int) -> str:
    """Broader video key extraction for high trailer match rates."""
    if TMDB_API_KEY and movie_id and isinstance(movie_id, int):
        try:
            res = session.get(
                f"{TMDB_BASE}/movie/{movie_id}/videos",
                params={"api_key": TMDB_API_KEY},
                timeout=5.0
            )
            if res.status_code == 200:
                videos = res.json().get("results", [])
                
                # Priority 1: Official Trailer
                for v in videos:
                    if v.get("site") == "YouTube" and v.get("type") == "Trailer":
                        return f"https://www.youtube.com/watch?v={v.get('key')}"
                        
                # Priority 2: Teasers or Clips
                for v in videos:
                    if v.get("site") == "YouTube" and v.get("type") in ["Teaser", "Clip", "Featurette"]:
                        return f"https://www.youtube.com/watch?v={v.get('key')}"
                        
                # Priority 3: Any YouTube Video Key
                for v in videos:
                    if v.get("site") == "YouTube" and v.get("key"):
                        return f"https://www.youtube.com/watch?v={v.get('key')}"
        except Exception:
            pass
    return None

# ---------------------------------------------------------
# STEP 4: FUNCTION CALLING / TOOLS DEFINITIONS & HANDLERS
# ---------------------------------------------------------

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_movie_trailer",
            "description": "Fetch official YouTube trailer URL for a movie title or TMDB movie ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "movie_id": {
                        "type": "integer",
                        "description": "The TMDB numerical ID of the movie."
                    },
                    "movie_title": {
                        "type": "string",
                        "description": "The title of the movie."
                    }
                },
                "required": ["movie_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filter_movies_by_criteria",
            "description": "Filter candidate movies strictly by minimum rating or release year cutoff.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_rating": {
                        "type": "number",
                        "description": "Minimum TMDB vote rating out of 10 (e.g. 7.5)."
                    },
                    "min_year": {
                        "type": "integer",
                        "description": "Minimum release year cutoff (e.g. 2010)."
                    }
                }
            }
        }
    }
]

def execute_tool_call(tool_name: str, arguments: dict, candidate_pool: list) -> dict:
    """Executes local Python function based on Groq tool decisions."""
    if tool_name == "get_movie_trailer":
        m_id = arguments.get("movie_id")
        trailer_url = get_movie_trailer_url(m_id)
        return {"movie_id": m_id, "trailer_url": trailer_url}

    elif tool_name == "filter_movies_by_criteria":
        min_rating = arguments.get("min_rating", 0.0)
        min_year = arguments.get("min_year", 0)
        
        filtered = []
        for m in candidate_pool:
            try:
                yr = int(m.get("year", 0))
            except Exception:
                yr = 0
            rt = float(m.get("rating", 0.0))
            
            if rt >= min_rating and yr >= min_year:
                filtered.append(m)
        return {"filtered_candidates": filtered}

    return {}

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_tmdb_direct_recommendations(movie_id: int, target_language: str):
    if not TMDB_API_KEY or not movie_id or not isinstance(movie_id, int):
        return []
    
    recs = []
    try:
        res_rec = session.get(f"{TMDB_BASE}/movie/{movie_id}/recommendations", params={"api_key": TMDB_API_KEY}, timeout=5.0)
        res_sim = session.get(f"{TMDB_BASE}/movie/{movie_id}/similar", params={"api_key": TMDB_API_KEY}, timeout=5.0)
        
        raw_rec = res_rec.json().get("results", []) if res_rec.status_code == 200 else []
        raw_sim = res_sim.json().get("results", []) if res_sim.status_code == 200 else []
        
        for item in (raw_rec + raw_sim):
            formatted = format_tmdb_movie_item(item, target_language)
            if formatted:
                recs.append(formatted)
    except Exception:
        pass
    return recs

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fallback_trending_movies(target_language: str = "All Languages"):
    if not TMDB_API_KEY:
        return []
    try:
        res = session.get(f"{TMDB_BASE}/trending/movie/week", params={"api_key": TMDB_API_KEY}, timeout=5.0)
        if res.status_code == 200:
            formatted_list = []
            for item in res.json().get("results", []):
                formatted = format_tmdb_movie_item(item, target_language)
                if formatted:
                    formatted_list.append(formatted)
            return formatted_list
    except Exception:
        pass
    return []

# ---------------------------------------------------------
# 4. PARALLEL VECTOR ENGINE & WEB FALLBACK CASCADE
# ---------------------------------------------------------

def safe_parse_groq_json(raw_text: str) -> list:
    """Extracts JSON arrays or movie titles robustly from LLM response strings."""
    if not raw_text:
        return []
    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            return data.get("movies", [])
        elif isinstance(data, list):
            return data
    except Exception:
        pass

    try:
        match = re.search(r'\{.*"movies"\s*:\s*(\[.*?\])\s*\}', raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except Exception:
        pass

    try:
        titles = re.findall(r'"([^"]+)"', raw_text)
        if titles:
            return [t for t in titles if len(t) > 2 and t.lower() != "movies"]
    except Exception:
        pass

    return []

def google_cse_search(query: str) -> list:
    """Executes site-restricted Google CSE search to pull exact movie title candidates."""
    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_CSE_ID:
        return []
    try:
        req_url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_SEARCH_API_KEY}&cx={GOOGLE_CSE_ID}&q={urllib.parse.quote(query + ' site:themoviedb.org/movie/')}"
        res = session.get(req_url, timeout=5.0)
        if res.status_code == 200:
            items = res.json().get("items", [])
            titles = []
            for it in items:
                raw_t = it.get("title", "")
                clean_t = raw_t.split("—")[0].split("-")[0].replace("TMDB", "").strip()
                if clean_t:
                    titles.append(clean_t)
            return titles
    except Exception:
        pass
    return []

def execute_web_fallback_cascade(vibe_prompt: str, selected_language: str, selected_industry: str) -> list:
    """3-Step Fallback Cascade: Google CSE -> Groq LLM Parsing -> TMDB Hydration."""
    # Step 1: Google CSE Search
    google_titles = google_cse_search(vibe_prompt)
    
    # Step 2: Groq LLM Extractor
    client = get_groq_client()
    groq_titles = []
    if client:
        industry_instruction = ""
        if "Hollywood" in selected_industry:
            industry_instruction = "CRITICAL MANDATE: Only return Western/American Cinema (Hollywood)."
        elif "Bollywood" in selected_industry:
            industry_instruction = "CRITICAL MANDATE: Only return Hindi-language Bollywood movies."
        elif "South Indian" in selected_industry:
            industry_instruction = "CRITICAL MANDATE: Only return South Indian movies."
        elif "Indian Cinema" in selected_industry:
            industry_instruction = "CRITICAL MANDATE: Only return Indian cinema movies."
        elif "East Asian" in selected_industry:
            industry_instruction = "CRITICAL MANDATE: Only return East Asian movies."

        sys_p = f"""Return output strictly in JSON. Provide 10-15 real movies matching vibe: "{vibe_prompt}".
Language: {selected_language} | Industry: {selected_industry} | {industry_instruction}
Format: {{"movies": ["Title 1", "Title 2"]}}"""

        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": sys_p}],
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            groq_titles = safe_parse_groq_json(resp.choices[0].message.content)
        except Exception:
            pass

    # Combine candidates
    combined_raw_titles = list(dict.fromkeys(google_titles + groq_titles))
    
    # Step 3: TMDB Hydration
    hydrated_movies = []
    def enrich(title):
        return fetch_single_movie_cached(title, target_language=selected_language)

    with ThreadPoolExecutor(max_workers=4) as exec_pool:
        results = list(exec_pool.map(enrich, combined_raw_titles))
        hydrated_movies = [m for m in results if m]

    return hydrated_movies

def query_local_index_stream(vibe_prompt: str, k: int = 15) -> list:
    """THREAD 1: Searches FAISS local index using L2 vector distance."""
    if FAISS_INDEX.ntotal == 0:
        return []
    
    q_vec = EMBEDDING_MODEL.encode([vibe_prompt], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    
    distances, indices = FAISS_INDEX.search(q_vec, min(k, FAISS_INDEX.ntotal))
    
    local_candidates = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx != -1:
            m_obj = VECTOR_METADATA.get(str(idx))
            if m_obj:
                local_candidates.append({
                    "movie": m_obj,
                    "distance": float(dist)
                })
    return local_candidates

def live_fetch_and_vectorize_stream(selected_language: str) -> list:
    """THREAD 2: Fetches fresh /now_playing TMDB movies, vectorizes & computes inline distance."""
    try:
        res = session.get(f"{TMDB_BASE}/movie/now_playing", params={"api_key": TMDB_API_KEY}, timeout=4.0)
        if res.status_code != 200:
            return []
        
        raw_items = res.json().get("results", [])
        existing_ids = {str(m["id"]) for m in VECTOR_METADATA.values() if isinstance(m, dict) and "id" in m}
        
        fresh_movies = []
        for item in raw_items:
            if str(item.get("id")) not in existing_ids:
                fmt = format_tmdb_movie_item(item, selected_language)
                if fmt:
                    fresh_movies.append(fmt)
        return fresh_movies
    except Exception:
        return []

def search_by_vibe_pipeline(vibe_prompt: str, selected_language: str, selected_industry: str, target_movie_obj=None, status_container=None):
    results = []
    
    # 1. TMDB Native Recommendations (if target movie exists)
    if target_movie_obj and isinstance(target_movie_obj.get("id"), int):
        if status_container:
            status_container.write(f"🎯 Fetching direct TMDB recommendations for '{target_movie_obj['title']}'...")
        tmdb_recs = fetch_tmdb_direct_recommendations(target_movie_obj["id"], selected_language)
        results.extend(tmdb_recs)

    # 2. RETRIEVAL STEP: Parallel Dual-Stream Vector Search (FAISS + Live Stream)
    if status_container:
        status_container.write("⚡ [RAG Step 1/3] Retrieving candidates from FAISS Vector Index...")

    local_candidates = []
    live_movies = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(query_local_index_stream, vibe_prompt, 15)
        f2 = executor.submit(live_fetch_and_vectorize_stream, selected_language)
        
        local_candidates = f1.result()
        live_movies = f2.result()

    best_distance = 999.0
    for cand in local_candidates:
        results.append(cand["movie"])
        if cand["distance"] < best_distance:
            best_distance = cand["distance"]

    if live_movies:
        q_vec = EMBEDDING_MODEL.encode([vibe_prompt], convert_to_numpy=True)
        faiss.normalize_L2(q_vec)
        
        for lm in live_movies:
            doc = f"{lm.get('title', '')} {lm.get('genres', '')} {lm.get('overview', '')}"
            m_vec = EMBEDDING_MODEL.encode([doc], convert_to_numpy=True)
            faiss.normalize_L2(m_vec)
            
            l2_dist = float(np.linalg.norm(q_vec - m_vec))
            results.append(lm)
            
            if l2_dist < best_distance:
                best_distance = l2_dist

    # 3. DISTANCE DECISION GATE (Web Fallback)
    if best_distance > DISTANCE_THRESHOLD or not results:
        if status_container:
            status_container.write(f"🌐 Distance Gate Triggered (Min Distance: {best_distance:.2f} > 0.35). Launching Web Fallback Cascade...")
        
        fallback_movies = execute_web_fallback_cascade(vibe_prompt, selected_language, selected_industry)
        results.extend(fallback_movies)

    # 4. Deduplication & Cleaning Candidate Pool
    unique_candidates = []
    seen_ids = set()
    seen_titles = set()

    if target_movie_obj and isinstance(target_movie_obj.get("id"), int):
        unique_candidates.append(target_movie_obj)
        seen_ids.add(target_movie_obj["id"])
        seen_titles.add(str(target_movie_obj["title"]).strip().lower())

    for item in results:
        if item and isinstance(item, dict) and item.get("id"):
            norm_title = str(item["title"]).strip().lower()
            m_id = item.get("id")
            if m_id not in seen_ids and norm_title not in seen_titles:
                seen_ids.add(m_id)
                seen_titles.add(norm_title)
                unique_candidates.append(item)

    if not unique_candidates:
        if status_container:
            status_container.write("🍿 Fetching top trending movies...")
        unique_candidates = fetch_fallback_trending_movies(selected_language)

    # Limit context window to top 10 candidates for the LLM
    candidate_pool = unique_candidates[:10]

    # 5. AUGMENTATION & GENERATION STEP: Function Calling & Groq LLM
    if status_container:
        status_container.write("🧠 [RAG Step 2/3 & 3/3] Executing Tool Calls & Generating Explanations...")

    context_str = "\n".join([
        f"- ID: {m['id']} | Title: {m['title']} ({m.get('year', 'N/A')}) | Rating: {m.get('rating', 'N/A')} | Overview: {m.get('overview', '')[:120]}..."
        for m in candidate_pool
    ])

    system_prompt = f"""You are an expert movie discovery AI with function-calling capabilities.
User Query / Vibe: "{vibe_prompt}"
Selected Region: {selected_industry} | Language: {selected_language}

RETRIEVED CANDIDATE MOVIES FROM VECTOR DATABASE:
{context_str}

TASK:
1. If the user specifies numeric constraints (e.g. rating > 7.5 or released after 2015), call `filter_movies_by_criteria`.
2. Select the top movies matching the vibe from the candidates list.
3. For each selected movie, provide a concise 1-2 sentence "vibe_reason".

Return strictly valid JSON format:
{{
  "recommendations": [
    {{
      "id": 12345,
      "title": "Movie Title",
      "vibe_reason": "Why this matches the vibe..."
    }}
  ]
}}"""

    final_rag_results = []
    client = get_groq_client()

    if client:
        try:
            messages = [{"role": "system", "content": system_prompt}]
            
            # Initial Groq call supplying tools schema
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            
            response_msg = resp.choices[0].message
            
            # Execute tool if requested by the LLM
            if hasattr(response_msg, "tool_calls") and response_msg.tool_calls:
                for tool_call in response_msg.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)
                    tool_result = execute_tool_call(fn_name, fn_args, candidate_pool)
                    
                    if fn_name == "filter_movies_by_criteria" and tool_result.get("filtered_candidates"):
                        candidate_pool = tool_result["filtered_candidates"]
                    
                    messages.append(response_msg)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result)
                    })
                
                # Second Groq call after returning tool results
                second_resp = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=messages,
                    temperature=0.2,
                    max_tokens=800,
                    response_format={"type": "json_object"}
                )
                llm_json = json.loads(second_resp.choices[0].message.content)
            else:
                llm_json = json.loads(response_msg.content)

            selected_items = llm_json.get("recommendations", [])
            candidate_dict = {m["id"]: m for m in candidate_pool}
            for item in selected_items:
                m_id = item.get("id")
                if m_id in candidate_dict:
                    movie_data = candidate_dict[m_id].copy()
                    movie_data["vibe_reason"] = item.get("vibe_reason", movie_data.get("overview", ""))
                    final_rag_results.append(movie_data)

        except Exception:
            final_rag_results = candidate_pool
    else:
        final_rag_results = candidate_pool

    if not final_rag_results:
        final_rag_results = candidate_pool

    # 6. MUTEX WRITE-BACK TO FAISS PERSISTENT STORE
    def background_persist():
        for m in final_rag_results:
            append_to_vector_db_mutex(m)

    threading.Thread(target=background_persist, daemon=True).start()

    return final_rag_results

# ---------------------------------------------------------
# 5. STREAMLIT UI - SEARCH & CONTROLS
# ---------------------------------------------------------

st.title("🎬 VibeVisualizer")
st.caption("High-Yield Vibe Discovery Engine across Indian, Hollywood & World Cinema.")
# --- SIDEBAR MEMORY CONTROL PANEL ---
init_memory()

with st.sidebar:
    st.title("🧠 Vibe Memory")
    
    st.session_state["vibe_preference"] = st.selectbox(
        "⚡ Preferred Vibe",
        ["All Vibes", "Dark & Moody", "Cyberpunk / Synthwave", "Action & Thriller", "Feel Good / Cozy"],
        index=0
    )
    
    st.markdown("---")
    
    tab_favs, tab_hist = st.tabs(["⭐ Favorites", "📜 History"])
    
    with tab_favs:
        if st.session_state["favorite_movies"]:
            for fav in st.session_state["favorite_movies"]:
                col_title, col_play, col_del = st.columns([3, 1, 1])
                with col_title:
                    st.caption(f"🎬 **{fav['title']}**")
                with col_play:
                    if st.button("▶", key=f"tab_play_{fav['id']}"):
                        st.session_state["selected_movie"] = fav
                        st.rerun()
                with col_del:
                    if st.button("❌", key=f"tab_del_fav_{fav['id']}"):
                        toggle_favorite(fav)
                        st.rerun()
            st.divider()
            if st.button("🗑️ Clear All Favorites", key="btn_clear_favs", use_container_width=True):
                clear_favorites()
                st.rerun()
        else:
            st.info("No saved movies yet.")

    with tab_hist:
        if st.session_state["search_history"]:
            for idx, past_query in enumerate(list(st.session_state["search_history"])):
                col_q, col_qdel = st.columns([4, 1])
                with col_q:
                    st.caption(f"🔍 {past_query}")
                with col_qdel:
                    if st.button("❌", key=f"tab_del_hist_{idx}"):
                        st.session_state["search_history"].pop(idx)
                        try:
                            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                                json.dump(st.session_state["search_history"], f, indent=2)
                        except Exception:
                            pass
                        st.rerun()
            st.divider()
            if st.button("🗑️ Clear All History", key="btn_clear_hist", use_container_width=True):
                clear_search_history()
                st.rerun()
        else:
            st.info("No search history yet.")

if not GROQ_API_KEY or not TMDB_API_KEY:
    st.error("⚠️ Missing API Keys! Please check your .env file or Streamlit secrets for GROQ_API_KEY and TMDB_API_KEY.")
    st.stop()

col_input, col_lang, col_ind = st.columns([3.5, 1, 1.2])

with col_input:
    user_query = st.text_input(
        "Describe your vibe OR enter a movie name:", 
        placeholder="e.g., 'Casablanca', 'RRR', 'high octane action', 'John Wick'"
    )

with col_lang:
    selected_lang = st.selectbox("🗣️ Language:", LANGUAGE_OPTIONS, index=0)

with col_ind:
    selected_industry = st.selectbox("🏢 Cinema Region Focus:", INDUSTRY_OPTIONS, index=0)

search_mode = st.radio(
    "Search Mode:", 
    ["Search by Vibe / Mood", "Search by Specific Movie Name (Vibe Recommendations)"], 
    horizontal=True
)

if st.button("Explore Movies 🚀", type="primary"):
    if not user_query.strip():
        st.info("Please enter a vibe or movie title above!")
    else:
        save_search_to_memory(user_query)
        st.session_state["selected_movie"] = None
        st.session_state["last_query"] = user_query
        
        status_box = st.status("Searching catalog & resolving movies...", expanded=True)
        
        if search_mode == "Search by Specific Movie Name (Vibe Recommendations)":
            status_box.write(f"🎯 Locating target movie '{user_query}'...")
            target_movie = fetch_single_movie_cached(user_query, target_language=selected_lang)
            st.session_state["searched_movie_info"] = target_movie
            
            vibe_q = f"{user_query} epic drama action" if not target_movie else f"{target_movie['title']} {target_movie['genres']} epic vibe"
            
            raw_results = search_by_vibe_pipeline(
                vibe_prompt=vibe_q,
                selected_language=selected_lang,
                selected_industry=selected_industry,
                target_movie_obj=target_movie,
                status_container=status_box
            )
            
            # DEDUPLICATION FIX: Exclude searched movie from recommendation results
            if target_movie and "id" in target_movie and raw_results:
                target_id = target_movie["id"]
                st.session_state["search_results"] = [
                    m for m in raw_results if m.get("id") != target_id
                ]
            else:
                st.session_state["search_results"] = raw_results

        else:
            st.session_state["searched_movie_info"] = None
            st.session_state["search_results"] = search_by_vibe_pipeline(
                vibe_prompt=user_query,
                selected_language=selected_lang,
                selected_industry=selected_industry,
                target_movie_obj=None,
                status_container=status_box
            )
            
        status_box.update(label="Vibe Discovery Complete!", state="complete", expanded=False)

st.markdown("---")

# Default Home View
if st.session_state["search_results"] is None and not st.session_state["selected_movie"]:
    st.subheader("🔥 Popular Trending Movies")
    st.session_state["search_results"] = fetch_fallback_trending_movies(selected_lang)

# ---------------------------------------------------------
# 6. MOVIE DETAIL VIEW
# ---------------------------------------------------------

if st.session_state["selected_movie"]:
    movie = st.session_state["selected_movie"]
    
    if st.button("🔙 Back to All Recommendations"):
        st.session_state["selected_movie"] = None
        st.rerun()

    st.markdown(f"## 🎬 {movie['title']} ({movie['year']})")
    
    col_hero1, col_hero2 = st.columns([1, 2])
    
    with col_hero1:
        st.image(movie["poster"], use_container_width=True)
        
    with col_hero2:
        st.markdown("### **Overview**")
        st.write(movie["overview"])
        st.markdown(f"⭐ **TMDB Rating:** {movie['rating']}/10")
        st.markdown(f"🏷️ **Genres:** `{movie['genres']}`")
        st.markdown(f"🗣️ **Audio Tracks:** `{movie['audio']}`")
        st.markdown(f"💬 **Subtitles:** `{movie['subtitles']}`")

        st.markdown("### 🍿 Official Trailer")
        t_url = get_movie_trailer_url(movie["id"])
        if t_url:
            st.video(t_url)
        else:
            search_query = urllib.parse.quote(f"{movie['title']} {movie['year']} official trailer")
            st.markdown(f"🔗 [Watch Trailer directly on YouTube](https://www.youtube.com/results?search_query={search_query})")

    st.markdown("---")
    st.subheader(f"🔥 More Movies Like '{movie['title']}'")

    if "recs_cache" not in st.session_state or st.session_state.get("recs_movie_id") != movie["id"]:
        st.session_state["recs_cache"] = search_by_vibe_pipeline(
            vibe_prompt=f"Movies matching vibe of {movie['title']}", 
            selected_language=selected_lang, 
            selected_industry=selected_industry,
            target_movie_obj=movie
        )
        st.session_state["recs_movie_id"] = movie["id"]

    rec_movies = st.session_state["recs_cache"][:8]
    
    if rec_movies:
        for i in range(0, len(rec_movies), 4):
            r_cols = st.columns(4)
            r_row_movies = rec_movies[i:i+4]
            for idx, r_movie in enumerate(r_row_movies):
                with r_cols[idx]:
                    st.image(r_movie["poster"], use_container_width=True)
                    desc_text = movie.get('vibe_reason', movie.get('overview', ''))
                    st.markdown(f"<div class='card-desc'>{desc_text}</div>", unsafe_allow_html=True)
                    st.caption(f"⭐ {r_movie['rating']}/10")
                    if st.button("Watch Details ▶", key=f"rec_btn_{r_movie['id']}_{i}_{idx}"):
                        st.session_state["selected_movie"] = r_movie
                        st.rerun()

# ---------------------------------------------------------
# 7. ORDERED ROW-BY-ROW GRID DISPLAY
# ---------------------------------------------------------

else:
    # A. Target Searched Movie Hero Banner
    if st.session_state["searched_movie_info"]:
        s_movie = st.session_state["searched_movie_info"]
        
        st.markdown("<div class='hero-container'>", unsafe_allow_html=True)
        st.subheader(f"🎯 Target Movie Searched: {s_movie['title']} ({s_movie['year']})")
        
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            st.image(s_movie["poster"], use_container_width=True)
        with col_s2:
            st.markdown(f"⭐ **TMDB Rating:** {s_movie['rating']}/10 | 🏷️ **Genres:** `{s_movie['genres']}`")
            st.markdown(f"🗣️ **Audio:** `{s_movie['audio']}` | 💬 **Subs:** `{s_movie['subtitles']}`")
            st.write(f"📖 **pip install streamlit sentence-transformers faiss-cpu requests python-dotenv groq numpyOverview:** {s_movie['overview']}")
            
            st.markdown("#### 🍿 Official Trailer")
            s_trailer = get_movie_trailer_url(s_movie["id"])
            if s_trailer:
                st.video(s_trailer)
            else:
                s_search_q = urllib.parse.quote(f"{s_movie['title']} {s_movie['year']} official trailer")
                st.markdown(f"🔗 [Watch Trailer directly on YouTube](https://www.youtube.com/results?search_query={s_search_q})")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")

    # B. Guaranteed Row-by-Row Grid
    if st.session_state["search_results"]:
        movies = st.session_state["search_results"]
        
        if st.session_state["searched_movie_info"]:
            st.subheader(f"🍿 Found {len(movies)} Movies Matching Vibe of '{st.session_state['searched_movie_info']['title']}':")
        elif st.session_state["last_query"]:
            st.subheader(f"🍿 Found {len(movies)} Movies Matching Vibe: '{st.session_state['last_query']}'")

        for i in range(0, len(movies), 4):
            row_cols = st.columns(4)
            row_movies = movies[i:i+4]
            
            for idx, movie in enumerate(row_movies):
                with row_cols[idx]:
                    st.image(movie["poster"], use_container_width=True)
                    st.markdown(f"<div class='card-title'><b>{movie['title']}</b> ({movie['year']})</div>", unsafe_allow_html=True)
                    st.markdown(f"<span class='vibe-badge'>🔥 Match</span> <span class='genre-badge'>{movie['genres']}</span>", unsafe_allow_html=True)
                    st.markdown(f"<div class='card-meta'>⭐ <b>Rating:</b> {movie['rating']}/10<br>🗣️ <b>Audio:</b> {movie['audio']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='card-desc'>{movie['overview']}</div>", unsafe_allow_html=True)
                    
                    col_b1, col_b2 = st.columns(2)

                    with col_b1:
                        if st.button("🎬 Details", key=f"card_btn_{movie['id']}_{i}_{idx}"):
                            st.session_state["selected_movie"] = movie
                            st.rerun()

                    with col_b2:
                        is_saved = any(m.get("id") == movie["id"] for m in st.session_state["favorite_movies"])
                        fav_label = "❤️ Saved" if is_saved else "🤍 Save"
                        if st.button(fav_label, key=f"card_fav_{movie['id']}_{i}_{idx}"):
                            toggle_favorite(movie)
                            st.rerun()