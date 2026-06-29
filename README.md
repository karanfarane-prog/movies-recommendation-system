# 🎬 Movie Recommendation System

A **content-based movie recommendation engine** built with Python, Streamlit, and scikit-learn. Find your next favorite movie based on genres, keywords, cast, director, and plot overview.

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---
## Live Demo
[https://movies-recommendation-system-jaqstj4zqx7mu5mk8pgdbu.streamlit.app/](https://movies-recommendation-system-jaqstj4zqx7mu5mk8pgdbu.streamlit.app/)

## GitHub Repository
[https://github.com/your-username/movies-recommendation-system](https://github.com/karanfarane-prog/movies-recommendation-system)

## 📌 Features

| Feature | Description |
|:---|:---|
| 🔍 **Search by Movie** | Enter a movie title and get 5 similar recommendations |
| 🎭 **Search by Genre** | Browse movies from your favorite genre |
| 🎯 **Movie + Genre Combo** | Narrow down recommendations by genre |
| ⭐ **User Ratings** | View TMDB crowd scores and top-rated movies |
| 🎬 **Movie Posters** | Visual recommendations powered by TMDB API |
| 📊 **Model Analysis** | Understand how the recommendation engine works |
| 🖱️ **Clickable Cards** | Click any movie card to view details on TMDB |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|:---|:---|
| **Python 3.8+** | Backend logic |
| **Streamlit** | Web interface |
| **pandas** | Data manipulation |
| **NumPy** | Numerical operations |
| **scikit-learn** | TF-IDF Vectorization & Cosine Similarity |
| **TMDB API** | Movie posters & details |
| **HTML/CSS** | Custom styling |

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Git
- TMDB API Key (free from [TMDB](https://www.themoviedb.org/signup))

### Setup Instructions
1. **Clone the repository**
git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system

2.**Create a virtual environment**
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

3.**Install dependencies**
pip install -r requirements.txt

4.**Add TMDB API Key**
Create a file at .streamlit/secrets.toml:
TMDB_API_KEY = "your_api_key_here"

5.**Run the app**
streamlit run app.py

**📊 How It Works**

**Recommendation Engine Pipeline**

1.Data Preprocessing – Merge movies and credits datasets

2.Feature Extraction – Extract genres, keywords, cast, director, overview

3.Tag Building – Combine all features into a single tag string

4.Vectorization – Convert tags to TF-IDF vectors (5000 features)

5.Similarity Calculation – Compute cosine similarity matrix (4806 x 4806)

6.Recommendation – Find top 5 most similar movies


