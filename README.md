# 🎬 VibeVisualizer — AI Film Discovery Agent

> **An AI-Powered Movie Search & Aesthetic Vibe Discovery Engine**

**VibeVisualizer** is an intelligent film recommendation platform built with **Streamlit**, **Groq AI (Llama 3.3)**, and **The Movie Database (TMDb) API**. It offers a dual-mode discovery system: perform direct title lookups for instant TMDb recommendations, or describe abstract mood vibes to let AI map out curated movie selections.

---

## ✨ Features

### 1. Dual Search Architecture (Mode Switcher)
Toggle seamlessly between two powerful search paths using the top menu:
* **🎯 Path A: Search by Specific Movie Name**
  * Direct lookup of any film using TMDb API.
  * Displays the main movie card (High-resolution poster, release year, community rating, audio/subtitle tracks, and plot overview).
  * Automatically fetches and displays direct similar movie recommendations powered by TMDb's recommendation engine.
* **✨ Path B: Search by Vibe / Mood**
  * Parses natural language queries (e.g., *"high energy hero revenge action"*, *"melancholic rainy night"*, *"90s nostalgic romance"*).
  * Uses **Groq (Llama-3.3-70b-versatile)** to dynamically map mood prompts to relevant movie titles.
  * Enriches AI results by pulling live metadata (posters, ratings, release years, official trailers) directly from TMDb.

### 2. Interactive UI & Custom Controls
* **Sidebar Controls:** Dynamic slider allowing users to customize the number of displayed recommendations (from 4 up to 20 films).
* **Responsive Grid Display:** Clean, multi-column visual card layout optimized for posters and quick details.
* **Robust Error Handling & Caching:** Uses Streamlit resource caching (`@st.cache_resource`) and custom HTTP session headers to maintain fast API connections.
* **User Authentication & Favorites:** Authenticates users via Firebase Auth REST APIs and allows saving favorite movies to a realtime Firebase database.

---

## 🛠️ Tech Stack

* **Frontend & Web Framework:** [Streamlit](https://streamlit.io/)
* **AI & LLM Provider:** [Groq API](https://groq.com/) (`llama-3.3-70b-versatile`)
* **Metadata & Recommendation Engine:** [The Movie Database (TMDb) API](https://www.themoviedb.org/)
* **Vector Indexing & ML:** FAISS (`faiss-cpu`), Sentence-Transformers (`all-MiniLM-L6-v2`)
* **Authentication & Database:** Firebase Auth REST API & Realtime Database
* **Environment & Security:** `python-dotenv` & Streamlit Secrets management

---

## 🚀 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/vibe-visualizer.git](https://github.com/your-username/vibe-visualizer.git)
cd vibe-visualizer
'''
### 2. Set Up Virtual Environment

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

'''
### 3. Install Dependencies

```bash
pip install -r requirements.txt

'''
### 4. Configure Environment Secrets

Create a .streamlit/secrets.toml file (or a .env file) in the root directory and add your credentials:

```bash

GROQ_API_KEY = "your_groq_api_key"
TMDB_API_KEY = "your_tmdb_api_key"
FIREBASE_API_KEY = "your_firebase_api_key"
FIREBASE_DATABASE_URL = "[https://your-firebase-database-default-rtdb.firebaseio.com](https://your-firebase-database-default-rtdb.firebaseio.com)"
'''
### 🏃 Running the Application
Launch the Streamlit app with:

```bash

streamlit run app.py
'''
### 📁 Project Structure

```bash

vibe-visualizer/
├── .streamlit/
│   └── secrets.toml        # Streamlit secret keys (GROQ_API_KEY, TMDB_API_KEY)
├── app.py                  # Core Streamlit application & logic
├── movie_embeddings.index  # Cached FAISS binary vector index
├── movie_metadata.json     # Metadata mapping store for indexed movies
├── .env                    # Environment variable backup
├── .gitignore              # Git exclusion rules for API key security
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation

'''
### 🗺️ Architectural Roadmap: Transitioning to Agentic AI
To evolve VibeVisualizer from a single-domain retrieval pipeline into an Autonomous Multi-Agent System, the following upgrades are planned:

[ ] Multi-Agent Orchestration (LangGraph / CrewAI): Transitioning to an Orchestrator-Worker pattern where dedicated agents handle distinct sub-tasks (Movie Agent, Music/Soundtrack Agent, Location/Vibe Agent) to build full experience itineraries.

[ ] Autonomous Tool Calling: Replacing static fallback logic with dynamic LLM tool selection using custom function-calling schemas.

[ ] Reflection & Self-Correction Loops: Adding a Critic Node that evaluates recommendation quality against user intent before rendering final outputs.

[ ] Persistent Memory Graphs: Implementing short-term context tracking and long-term user taste profiles stored in a vector-backed memory layer.
