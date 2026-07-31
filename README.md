# 🎬 VibeVisualizer — AI Film Discovery Agent

> **An AI-Powered Movie Search & Aesthetic Vibe Discovery Engine**

**VibeVisualizer** is an intelligent film recommendation platform built with **Streamlit**, **Groq AI (Llama 3.3)**, and **The Movie Database (TMDb) API**. It offers a dual-mode discovery system: perform direct title lookups for instant TMDb recommendations, or describe abstract mood vibes to let AI map out curated movie selections.

---

## ✨ Features

### 1. Dual Search Architecture (Mode Switcher)
Toggle seamlessly between two powerful search paths using the top menu:
* **🎯 Path A: Search by Specific Movie Name**
  * Direct lookup of any film using TMDb API.
  * Displays the main movie card (High-resolution poster, release year, community rating, and plot overview).
  * Automatically fetches and displays direct similar movie recommendations powered by TMDb's recommendation engine.
* **✨ Path B: Search by Vibe / Mood**
  * Parses natural language queries (e.g., *"high energy hero revenge action"*, *"melancholic rainy night"*, *"90s nostalgic romance"*).
  * Uses **Groq (Llama-3.3-70b-versatile)** to dynamically map mood prompts to relevant movie titles.
  * Enriches AI results by pulling live metadata (posters, ratings, release years) directly from TMDb.

### 2. Interactive UI & Custom Controls
* **Sidebar Controls:** Dynamic slider allowing users to customize the number of displayed recommendations (from 4 up to 20 films).
* **Responsive Grid Display:** Clean, 4-column visual card layout optimized for posters and quick details.
* **Robust Error Handling & Caching:** Uses Streamlit resource caching (`@st.cache_resource`) and custom HTTP session headers to maintain fast API connections.

---

## 🛠️ Tech Stack

* **Frontend & Web Framework:** [Streamlit](https://streamlit.io/)
* **AI & LLM Provider:** [Groq API](https://groq.com/) (`llama-3.3-70b-versatile`)
* **Metadata & Recommendation Engine:** [The Movie Database (TMDb) API](https://www.themoviedb.org/)
* **Environment & Security:** `python-dotenv` & Streamlit Secrets management

---

## 📁 Project Structure

```text
vibe-visualizer/
├── .streamlit/
│   └── secrets.toml       # Streamlit secret keys (GROQ_API_KEY, TMDB_API_KEY)
├── app.py                 # Core Streamlit application & logic
├── .env                   # Environment variable backup
├── .gitignore             # Git exclusion rules for API key security
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
