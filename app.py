import streamlit as st
import requests
import os
import urllib.parse
from dotenv import load_dotenv
from groq import Groq 

# Load credentials securely from the environment
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Application Page Setup with wide grid layout
st.set_page_config(page_title="VibeVisualizer Minimalist", layout="wide", initial_sidebar_state="expanded")

# Minimalist High-Contrast White Theme UI Custom CSS Styles
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Background - Pristine White & Off-White Accents */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Inter', sans-serif;
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    
    /* Sidebar Background Overrides to clean light grey */
    [data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    
    /* Typography Style overrides */
    h1, h2, h3, h4 {
        color: #0f172a !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    /* Minimalist Card Design */
    .movie-card {
        background: #ffffff;
        border: 1px solid #e2e8f0; 
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease-in-out;
        margin-bottom: 25px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .movie-card:hover {
        transform: translateY(-4px);
        border-color: #94a3b8; 
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }
    
    .movie-title {
        color: #0f172a !important;
        font-size: 18px !important;
        margin-top: 12px !important;
        margin-bottom: 4px !important;
        font-weight: 600 !important;
    }
    
    .actress-label {
        color: #64748b; 
        font-size: 13px;
        font-weight: 500;
        margin-bottom: 8px;
    }
    
    .movie-review {
        font-size: 13px;
        color: #334155;
        line-height: 1.5;
        margin-top: 8px;
        border-top: 1px solid #f1f5f9;
        padding-top: 8px;
    }
    
    /* Clean, high-contrast button */
    .trailer-btn {
        display: block;
        background-color: #0f172a;
        color: #ffffff !important;
        font-weight: 500;
        font-size: 13px;
        text-decoration: none;
        padding: 10px 14px;
        border-radius: 6px;
        text-align: center;
        margin-top: 14px;
        transition: background-color 0.2s;
    }
    .trailer-btn:hover {
        background-color: #334155;
    }
    
    /* Global form input fields style alignment */
    input, textarea, select {
        background-color: #ffffff !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- ADVANCED MOVIE DATA EXTRACTION FUNCTIONS ---
def fetch_movie_extra_details(movie_id, movie_title):
    fallback_trailer = f"https://www.youtube.com/results?search_query={urllib.parse.quote(movie_title + ' trailer')}"
    if not TMDB_API_KEY:
        return "Not Available", fallback_trailer
    try:
        credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits"
        credits_res = requests.get(credits_url, params={"api_key": TMDB_API_KEY}).json()
        cast = credits_res.get("cast", [])
        
        lead_actress = "Not Found"
        for member in cast[:8]:
            if member.get("gender") == 1:
                lead_actress = member.get("name")
                break
                
        videos_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos"
        videos_res = requests.get(videos_url, params={"api_key": TMDB_API_KEY}).json()
        videos = videos_res.get("results", [])
        
        trailer_url = fallback_trailer
        for video in videos:
            if video.get("type") == "Trailer" and video.get("site") == "YouTube":
                trailer_url = f"https://www.youtube.com/watch?v={video.get('key')}"
                break
        return lead_actress, trailer_url
    except Exception:
        return "Error Fetching", fallback_trailer

def fetch_movie_details_by_title(movie_title):
    if not TMDB_API_KEY:
        return None, "Unknown", f"https://www.youtube.com/results?search_query={urllib.parse.quote(movie_title + ' trailer')}"
    search_url = "https://api.themoviedb.org/3/search/movie"
    try:
        res = requests.get(search_url, params={"api_key": TMDB_API_KEY, "query": movie_title}).json()
        results = res.get("results", [])
        if results:
            movie_id = results[0]["id"]
            poster_path = results[0].get("poster_path")
            p_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
            actress, t_url = fetch_movie_extra_details(movie_id, movie_title)
            return p_url, actress, t_url
    except Exception:
        pass
    return None, "Not Available", f"https://www.youtube.com/results?search_query={urllib.parse.quote(movie_title + ' trailer')}"

def fetch_live_collections(endpoint_type):
    if not TMDB_API_KEY:
        st.error("❌ Missing TMDB_API_KEY in your .env file!")
        return []
    url_map = {
        "Trending Now": "https://api.themoviedb.org/3/trending/movie/day",
        "Top Rated Classics": "https://api.themoviedb.org/3/movie/top_rated",
        "Now Playing in Theaters": "https://api.themoviedb.org/3/movie/now_playing"
    }
    try:
        response = requests.get(url_map[endpoint_type], params={"api_key": TMDB_API_KEY})
        # If the API key is wrong, TMDB will return an error status code
        if response.status_code != 200:
            st.error(f"❌ TMDB API Error: {response.json().get('status_message', 'Invalid API key')}")
            return []
            
        results = response.json().get("results", [])[:9]
        live_movies = []
        for item in results:
            m_title = item.get("title", "Unknown Title")
            m_id = item.get("id")
            poster_path = item.get("poster_path")
            p_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=500"
            actress, trailer = fetch_movie_extra_details(m_id, m_title)
            overview = item.get("overview", "No description available.")
            live_movies.append({
                "title": m_title, "actress": actress,
                "review": overview[:140] + "..." if len(overview) > 140 else overview,
                "poster": p_url, "trailer": trailer
            })
        return live_movies
    except Exception as e:
        st.error(f"❌ Connection Error: {e}")
        return []

# --- USER STORAGE DATABASE SIMULATION ---
if "user_db" not in st.session_state:
    st.session_state.user_db = {
        "admin": {"password": "123", "favorites": [], "history": [], "name": "Chief Critic"},
        "guest": {"password": "guest", "favorites": [], "history": [], "name": "Guest Cinephile"}
    }
if "current_user" not in st.session_state:
    st.session_state.current_user = None

# --- PORTAL LOGIN PAGE ---
if st.session_state.current_user is None:
    st.markdown("<h1 style='text-align: center; margin-top: 50px;'>🎬 VibeVisualizer Portal</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_register = st.tabs(["🔒 Account Login", "📝 Register New Profile"])
        
        with tab_login:
            with st.form("auth_login"):
                user_in = st.text_input("Username").strip()
                pass_in = st.text_input("Password", type="password").strip()
                btn_login = st.form_submit_button("Authenticate")
                if btn_login:
                    if user_in in st.session_state.user_db and st.session_state.user_db[user_in]["password"] == pass_in:
                        st.session_state.current_user = user_in
                        st.rerun()
                    else:
                        st.error("Invalid Username or Password token.")
                        
        with tab_register:
            with st.form("auth_reg"):
                new_user = st.text_input("Choose Username").strip()
                new_name = st.text_input("Display Name (e.g. Alex)").strip()
                new_pass = st.text_input("Set Password", type="password").strip()
                btn_reg = st.form_submit_button("Create Profile")
                if btn_reg:
                    if new_user in st.session_state.user_db:
                        st.error("Username already taken.")
                    elif new_user == "" or new_pass == "":
                        st.warning("Fields cannot remain empty.")
                    else:
                        st.session_state.user_db[new_user] = {"password": new_pass, "favorites": [], "history": [], "name": new_name}
                        st.success("Registration successful! Switch to the login tab.")
    st.stop()

# --- AUTHENTICATED ACTIVE SESSION ---
user_key = st.session_state.current_user
user_data = st.session_state.user_db[user_key]

# Navigation Control Menu in Sidebar Container
with st.sidebar:
    st.markdown(f"<h3>👤 Welcome, <b>{user_data['name']}</b></h3>", unsafe_allow_html=True)
    st.markdown("---")
    app_mode = st.radio("Navigate Workspace:", ["🎬 Movie Core Engine", "⭐ Saved Favorites Portfolio", "⚙️ Manage Profile"])
    st.markdown("---")
    if st.button("🚪 Log Out"):
        st.session_state.current_user = None
        st.rerun()

# --- NAVIGATION ROUTING PANEL RULES ---
if app_mode == "⚙️ Manage Profile":
    st.markdown(f"<h2>⚙️ Profile Settings: <small style='color:#64748b;'>{user_key}</small></h2>", unsafe_allow_html=True)
    with st.form("update_profile_form"):
        updated_name = st.text_input("Modify Display Name", value=user_data["name"])
        updated_password = st.text_input("Change Password String", value=user_data["password"], type="password")
        if st.form_submit_button("Save Alterations"):
            st.session_state.user_db[user_key]["name"] = updated_name
            st.session_state.user_db[user_key]["password"] = updated_password
            st.success("Profile alterations successfully saved.")
            st.rerun()

elif app_mode == "⭐ Saved Favorites Portfolio":
    st.markdown("<h2>⭐ Your Handpicked <span style='color:#64748b;'>Favorites</span></h2>", unsafe_allow_html=True)
    favs = user_data["favorites"]
    if not favs:
        st.info("You haven't added any movies to your favorites portfolio yet.")
    else:
        for i in range(0, len(favs), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(favs):
                    m = favs[i + j]
                    with cols[j]:
                        st.markdown(f"""
                        <div class="movie-card">
                            <div>
                                <img src="{m['poster']}" style="width:100%; border-radius:8px;" />
                                <h3 class="movie-title">{m['title']}</h3>
                                <div class="actress-label">💃 Lead Actress: {m['actress']}</div>
                                <p class="movie-review">"{m['review']}"</p>
                            </div>
                            <a class="trailer-btn" href="{m['trailer']}" target="_blank">▶️ Watch Trailer</a>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button(f"🗑️ Remove", key=f"del_{i+j}"):
                            user_data["favorites"].pop(i + j)
                            st.rerun()

elif app_mode == "🎬 Movie Core Engine":
    st.markdown("<h1 style='color: #0f172a; margin-bottom: 0;'>🎬 VibeVisualizer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 15px; margin-top: 4px;'>Explore real-time live collections or generate recommendations matching your mood.</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color: #e2e8f0; margin-bottom: 30px;'>", unsafe_allow_html=True)

    col_input, col_display = st.columns([1, 3])
    with col_input:
        st.markdown("<h4>📡 Live Global Feeds</h4>", unsafe_allow_html=True)
        live_target = st.selectbox("Choose Live Stream:", ["Select Live Feed", "Trending Now", "Top Rated Classics", "Now Playing in Theaters"])
        
        st.markdown("<div style='text-align:center; color:#94a3b8; margin: 15px 0; font-size: 13px;'>— OR —</div>", unsafe_allow_html=True)
        
        st.markdown("<h4>✨ Custom Vibe Engine</h4>", unsafe_allow_html=True)
        mood_click = st.selectbox("Quick Select Vibe:", [
            "Select an Option", "🔥 Powerful Mass Hero / High Adrenaline", 
            "🧠 Psychological Mind-Bending Intellect", "🌧️ Deep Melancholic Emotion", "🦅 Royal / Historical Emperor Dynamics"
        ])
        user_description = st.text_area("Refine details:", placeholder="e.g. Serious cinematic background music, full action attitude...", height=100)
        submit_btn = st.button("Generate Dashboard", use_container_width=True)

    # API Processing Matrix Loops
    movies_output = None
    display_title = ""

    if live_target != "Select Live Feed":
        with st.spinner("Fetching live collection data..."):
            movies_output = fetch_live_collections(live_target)
            display_title = f"Live Update: {live_target}"
            if movies_output:
                user_data["history"].append({"prompt": display_title, "movies": movies_output})
                
    elif submit_btn and (mood_click != "Select an Option" or user_description.strip() != ""):
        final_prompt = user_description if mood_click == "Select an Option" else f"Vibe: {mood_click}. Details: {user_description}"
        display_title = f"Vibe Match: \"{final_prompt[:30]}...\""
        
        with st.spinner("Analyzing parameters via AI Core..."):
            try:
                client = Groq(api_key=GROQ_API_KEY)
                system_instructions = (
                    "You are an elite cinematic engine. Recommend exactly 9 fitting movies that align with the prompt. "
                    "Structure your response precisely like this for every single movie, separated by '---':\n"
                    "TITLE: [Movie Title]\n"
                    "ACTRESS: [Main lead actress name]\n"
                    "REVIEW: [A brief description explaining why it matches the dynamic mood]\n"
                    "---"
                )
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "system", "content": system_instructions}, {"role": "user", "content": final_prompt}]
                )
                ai_text = completion.choices[0].message.content
                
                parsed_movies = []
                for block in ai_text.split("---"):
                    if "TITLE:" in block:
                        lines = block.strip().split("\n")
                        t, act, rev = "Unknown", "Unknown", ""
                        for line in lines:
                            if line.startswith("TITLE:"): t = line.replace("TITLE:", "").strip().replace("[", "").replace("]", "")
                            elif line.startswith("ACTRESS:"): act = line.replace("ACTRESS:", "").strip().replace("[", "").replace("]", "")
                            elif line.startswith("REVIEW:"): rev = line.replace("REVIEW:", "").strip()
                        
                        p_url, fb_actress, t_url = fetch_movie_details_by_title(t)
                        parsed_movies.append({
                            "title": t, "actress": act if act != "Unknown" else fb_actress,
                            "review": rev, "poster": p_url if p_url else "https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=500",
                            "trailer": t_url
                        })
                if parsed_movies:
                    movies_output = parsed_movies[:9]
                    user_data["history"].append({"prompt": final_prompt, "movies": movies_output})
            except Exception as e:
                st.error(f"Engine Core Communication Fault: {e}")

    # RENDER ENGINE MATRIX SHOWCASE AREA
    with col_display:
        if user_data["history"]:
            current_session = user_data["history"][-1]
            st.markdown(f"<h5>🎯 Results for: <i>{current_session['prompt']}</i></h5>", unsafe_allow_html=True)
            movies_list = current_session["movies"]
            
            for row_idx in range(3):
                grid_cols = st.columns(3)
                for col_idx in range(3):
                    movie_index = row_idx * 3 + col_idx
                    if movie_index < len(movies_list):
                        movie = movies_list[movie_index]
                        with grid_cols[col_idx]:
                            st.markdown(f"""
                            <div class="movie-card">
                                <div>
                                    <img src="{movie['poster']}" style="width:100%; border-radius:8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);" />
                                    <h3 class="movie-title">{movie['title']}</h3>
                                    <div class="actress-label">💃 Lead Actress: {movie['actress']}</div>
                                    <p class="movie-review">"{movie['review']}"</p>
                                </div>
                                <a class="trailer-btn" href="{movie['trailer']}" target="_blank">▶️ Watch Trailer</a>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            is_fav = any(f["title"] == movie["title"] for f in user_data["favorites"])
                            if is_fav:
                                st.markdown("<p style='color: #10b981; font-size: 13px; font-weight: 500; margin-top:8px;'>❤️ Saved to Favorites</p>", unsafe_allow_html=True)
                            else:
                                if st.button(f"⭐ Add to Favorites", key=f"fav_btn_{movie_index}", use_container_width=True):
                                    user_data["favorites"].append(movie)
                                    st.rerun()
                                    
            # Profile Session History reload modules
            if len(user_data["history"]) > 1:
                st.markdown("<hr style='border-color: #e2e8f0;'>", unsafe_allow_html=True)
                st.markdown("<h3>📜 Personal Search History</h3>", unsafe_allow_html=True)
                for past_idx, past_data in enumerate(reversed(user_data["history"][:-1])):
                    if st.button(f"Reload Run {past_idx + 1}: {past_data['prompt'][:35]}", key=f"hist_{past_idx}"):
                        user_data["history"].append(past_data)
                        st.rerun()
        else:
            st.info("Please pick a live stream feed or define your custom criteria on the left panels to generate results.")