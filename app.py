"""
VibeVisualizer - Movie Discovery Engine (Full Code with Surgical Fixes)
"""

import os
import re
import json
import urllib.parse
import requests
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
load_dotenv()

def get_secret(key: str):
    return os.getenv(key) or st.secrets.get(key, None)

GROQ_API_KEY = get_secret("GROQ_API_KEY")
TMDB_API_KEY = get_secret("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"

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
# 4. GROQ LLM & HYBRID PIPELINE
# ---------------------------------------------------------

def safe_parse_groq_json(raw_text: str) -> list:
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

def extract_vibe_movies_fast(vibe_prompt: str, selected_language: str, selected_industry: str) -> list:
    client = get_groq_client()
    if not client:
        return []

    # CORRECTION 1: Explicit hard rules for Cinema Region Focus
    industry_instruction = ""
    if "Hollywood" in selected_industry:
        industry_instruction = "CRITICAL MANDATE: You MUST ONLY return movies from Western/American Cinema (Hollywood). Do NOT include Indian, Korean, or foreign movies."
    elif "Bollywood" in selected_industry:
        industry_instruction = "CRITICAL MANDATE: You MUST ONLY return Hindi-language Bollywood movies. Do NOT include Hollywood or other regional movies."
    elif "South Indian" in selected_industry:
        industry_instruction = "CRITICAL MANDATE: You MUST ONLY return South Indian movies (Telugu, Tamil, Malayalam, Kannada cinema)."
    elif "Indian Cinema" in selected_industry:
        industry_instruction = "CRITICAL MANDATE: You MUST ONLY return Indian cinema movies (Bollywood or Indian regional languages)."
    elif "East Asian" in selected_industry:
        industry_instruction = "CRITICAL MANDATE: You MUST ONLY return East Asian movies (Korean, Japanese, Chinese cinema)."
    elif "World Cinema" in selected_industry:
        industry_instruction = "CRITICAL MANDATE: You MUST ONLY return international or world cinema movies outside Hollywood and India."

    system_prompt = f"""
    You are an expert movie recommendation engine. Return output strictly in JSON format.
    Provide 15-20 distinct, real, existing movie titles that match the vibe: "{vibe_prompt}".
    Do NOT add years in parentheses (e.g., return "Sultan" NOT "Sultan (2016)").
    
    Language Filter: {selected_language}
    Industry Filter: {selected_industry}
    {industry_instruction}
    
    Return JSON as:
    {{
      "movies": [
        "Movie Title 1",
        "Movie Title 2"
      ]
    }}
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.3,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        return safe_parse_groq_json(response.choices[0].message.content)
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

    # 2. AI Vibe Search
    if status_container:
        status_container.write("🧠 Finding matching vibe titles...")
    
    groq_titles = extract_vibe_movies_fast(vibe_prompt, selected_language, selected_industry)

    if groq_titles:
        if status_container:
            status_container.write(f"⚡ Fetching verified TMDB details for {len(groq_titles)} titles...")

        def enrich(title):
            t_str = title.get("title", "") if isinstance(title, dict) else str(title)
            return fetch_single_movie_cached(t_str, target_language=selected_language)

        with ThreadPoolExecutor(max_workers=3) as executor:
            groq_results = list(executor.map(enrich, groq_titles))
            results.extend([m for m in groq_results if m])

    # 3. Deduplication and Cleaning
    unique_movies = []
    seen_ids = set()
    seen_titles = set()

    # CORRECTION 2: Prepend searched movie directly as Card #1 instead of pre-blocking it
    if target_movie_obj and isinstance(target_movie_obj.get("id"), int):
        unique_movies.append(target_movie_obj)
        seen_ids.add(target_movie_obj["id"])
        seen_titles.add(str(target_movie_obj["title"]).strip().lower())

    for item in results:
        if item and isinstance(item, dict) and item.get("id"):
            norm_title = str(item["title"]).strip().lower()
            m_id = item.get("id")
            if m_id not in seen_ids and norm_title not in seen_titles:
                seen_ids.add(m_id)
                seen_titles.add(norm_title)
                unique_movies.append(item)

    # 4. Trending Fallback
    if not unique_movies:
        if status_container:
            status_container.write("🍿 Fetching top trending movies...")
        unique_movies = fetch_fallback_trending_movies(selected_language)

    # Keep target movie as Card #1 while sorting all remaining vibe recommendations by popularity/rating
    if target_movie_obj and isinstance(target_movie_obj.get("id"), int) and len(unique_movies) > 1:
        target_card = unique_movies[0]
        recs_cards = unique_movies[1:]
        recs_cards.sort(key=lambda x: (x.get("rating", 0), x.get("popularity", 0)), reverse=True)
        unique_movies = [target_card] + recs_cards
    else:
        unique_movies.sort(key=lambda x: (x.get("rating", 0), x.get("popularity", 0)), reverse=True)

    return unique_movies

# ---------------------------------------------------------
# 5. STREAMLIT UI - SEARCH & CONTROLS
# ---------------------------------------------------------

st.title("🎬 VibeVisualizer")
st.caption("High-Yield Vibe Discovery Engine across Indian, Hollywood & World Cinema.")

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
        st.session_state["selected_movie"] = None
        st.session_state["last_query"] = user_query
        
        status_box = st.status("Searching catalog & resolving movies...", expanded=True)
        
        if search_mode == "Search by Specific Movie Name (Vibe Recommendations)":
            status_box.write(f"🎯 Locating target movie '{user_query}'...")
            target_movie = fetch_single_movie_cached(user_query, target_language=selected_lang)
            st.session_state["searched_movie_info"] = target_movie
            
            vibe_q = f"{user_query} epic drama action" if not target_movie else f"{target_movie['title']} {target_movie['genres']} epic vibe"
            st.session_state["search_results"] = search_by_vibe_pipeline(
                vibe_prompt=vibe_q,
                selected_language=selected_lang,
                selected_industry=selected_industry,
                target_movie_obj=target_movie,
                status_container=status_box
            )
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
                    st.markdown(f"<div class='card-title'><b>{r_movie['title']}</b> ({r_movie['year']})</div>", unsafe_allow_html=True)
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
            st.write(f"📖 **Overview:** {s_movie['overview']}")
            
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
                    
                    if st.button("🎬 Watch & Details", key=f"card_btn_{movie['id']}_{i}_{idx}"):
                        st.session_state["selected_movie"] = movie
                        st.rerun()