"""
VibeVisualizer - AI Film Discovery Agent
"""

import os
import requests
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# ---------------------------------------------------------
# 1. SETUP & SECRETS
# ---------------------------------------------------------
load_dotenv()

def get_secret(key: str):
    return os.getenv(key) or st.secrets.get(key, None)

GROQ_API_KEY = get_secret("GROQ_API_KEY")
TMDB_API_KEY = get_secret("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json"
}

session = requests.Session()
session.headers.update(HEADERS)

st.set_page_config(page_title="VibeVisualizer", page_icon="🎬", layout="wide")

@st.cache_resource
def get_groq_client():
    return Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------

def fetch_single_movie_details(title: str):
    """Searches TMDb for a movie title and returns its card details."""
    try:
        res = session.get(
            f"{TMDB_BASE}/search/movie",
            params={"api_key": TMDB_API_KEY, "query": title.strip()},
            timeout=8
        ).json()
        
        results = res.get("results", [])
        if results:
            item = results[0]
            return {
                "title": item.get("title"),
                "year": (item.get("release_date") or "")[:4],
                "rating": round(item.get("vote_average", 0), 1),
                "poster": f"{TMDB_IMG}{item.get('poster_path')}" if item.get("poster_path") else "https://via.placeholder.com/500x750?text=No+Poster",
                "overview": item.get("overview", "")
            }
    except Exception:
        pass
    return None

def search_by_movie_title(title: str, limit: int = 12):
    """Path A: Look up a target movie and fetch direct TMDb recommendations."""
    try:
        s_res = session.get(
            f"{TMDB_BASE}/search/movie", 
            params={"api_key": TMDB_API_KEY, "query": title}, 
            timeout=10
        ).json()
        
        results = s_res.get("results", [])
        if not results:
            return None, []
        
        main_item = results[0]
        main_movie = {
            "title": main_item.get("title"),
            "year": (main_item.get("release_date") or "")[:4],
            "rating": round(main_item.get("vote_average", 0), 1),
            "poster": f"{TMDB_IMG}{main_item.get('poster_path')}" if main_item.get("poster_path") else "https://via.placeholder.com/500x750?text=No+Poster",
            "overview": main_item.get("overview", "")
        }
        
        rec_res = session.get(
            f"{TMDB_BASE}/movie/{main_item['id']}/recommendations", 
            params={"api_key": TMDB_API_KEY}, 
            timeout=10
        ).json()
        
        recommendations = []
        for item in rec_res.get("results", [])[:limit]:
            recommendations.append({
                "title": item.get("title"),
                "year": (item.get("release_date") or "")[:4],
                "rating": round(item.get("vote_average", 0), 1),
                "poster": f"{TMDB_IMG}{item.get('poster_path')}" if item.get("poster_path") else "https://via.placeholder.com/500x750?text=No+Poster",
                "overview": item.get("overview", "")
            })
            
        return main_movie, recommendations
    except Exception as e:
        st.error(f"Error fetching movie recommendations: {e}")
        return None, []

def search_by_vibe(vibe_prompt: str, limit: int = 12):
    """Path B: AI maps vibe to actual movie titles, then fetches live metadata from TMDb."""
    try:
        client = get_groq_client()
        prompt = (
            f"You are a film expert. Suggest {limit} real movie titles (including latest and classic releases) "
            f"that match this exact mood/vibe: '{vibe_prompt}'.\n"
            f"Format output ONLY as a simple comma-separated list of titles. Example: Inception, Dune, RRR, Interstellar"
        )
        
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )
        
        raw_titles = resp.choices[0].message.content.strip().split(",")
        clean_titles = [t.strip(' "\'\n') for t in raw_titles if t.strip()]
        
        cards = []
        for t in clean_titles[:limit]:
            card = fetch_single_movie_details(t)
            if card:
                cards.append(card)
                
        return cards
    except Exception as e:
        st.error(f"Error generating vibe recommendations: {e}")
        return []

# ---------------------------------------------------------
# 3. STREAMLIT UI
# ---------------------------------------------------------
st.title("🎬 VibeVisualizer")
st.caption("Describe your vibe — unlock movies, music, and style.")

if not GROQ_API_KEY or not TMDB_API_KEY:
    st.warning("Please configure GROQ_API_KEY and TMDB_API_KEY in your .env file!")
    st.stop()

with st.sidebar:
    st.header("Settings")
    num_results = st.slider("Number of recommendations:", min_value=4, max_value=20, value=12, step=4)

search_mode = st.radio("Choose Search Type:", ["Search by Specific Movie Name", "Search by Vibe / Mood"], horizontal=True)
user_query = st.text_input("Enter a movie title or describe your vibe:", placeholder="e.g., 'RRR' or 'high energy hero revenge action'")

if st.button("Find Movies 🚀", type="primary"):
    if not user_query.strip():
        st.info("Please type something first!")
    else:
        with st.spinner("Agent is searching..."):
            if search_mode == "Search by Specific Movie Name":
                main_movie, recommendations = search_by_movie_title(user_query, limit=num_results)
                
                if main_movie:
                    st.subheader("🎯 Movie You Searched:")
                    m_col1, m_col2 = st.columns([1, 4])
                    with m_col1:
                        st.image(main_movie["poster"], use_container_width=True)
                    with m_col2:
                        st.markdown(f"### **{main_movie['title']}** ({main_movie['year']})")
                        st.markdown(f"⭐ **Rating:** {main_movie['rating']}/10")
                        st.write(main_movie["overview"])
                    
                    st.divider()
                    st.subheader(f"🍿 Movies Similar to '{main_movie['title']}':")
                    
                if recommendations:
                    cols = st.columns(4)
                    for idx, item in enumerate(recommendations):
                        with cols[idx % 4]:
                            st.image(item["poster"], use_container_width=True)
                            st.markdown(f"**{item['title']}** ({item['year']})")
                            st.caption(f"⭐ Rating: {item['rating']}/10")
                else:
                    st.warning("No recommendations found.")
                    
            else:
                results = search_by_vibe(user_query, limit=num_results)
                if results:
                    st.subheader(f"🍿 Movies Matching Vibe: '{user_query}'")
                    cols = st.columns(4)
                    for idx, item in enumerate(results):
                        with cols[idx % 4]:
                            st.image(item["poster"], use_container_width=True)
                            st.markdown(f"**{item['title']}** ({item['year']})")
                            st.caption(f"⭐ Rating: {item['rating']}/10")
                else:
                    st.error("No movies found for this vibe. Try rephrasing!")