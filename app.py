import streamlit as st
import pickle
import pandas as pd
import numpy as np
from recommend import recommend, fetch_movie_details, get_api_key, PLACEHOLDER_POSTER
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def safe_literal_eval(text):
    """ast.literal_eval, but never raises — malformed/missing JSON-ish
    strings just become an empty list instead of crashing data loading."""
    if pd.isna(text):
        return []
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []

def convert(text):
    return [i['name'] for i in safe_literal_eval(text)]

def convert_cast(text):
    L = []
    for i in safe_literal_eval(text):
        if len(L) == 3:
            break
        L.append(i['name'])
    return L

def fetch_director(text):
    for i in safe_literal_eval(text):
        if i.get('job') == 'Director':
            return [i['name']]
    return []

def collapse(name):
    """'Robert Downey Jr.' -> 'RobertDowneyJr.' so the vectorizer treats a
    person's full name as one token instead of splitting on first/last
    name (which would falsely link unrelated movies sharing a common
    first name like 'James')."""
    return str(name).replace(" ", "")

# PAGE CONFIG
st.set_page_config(
    page_title="Movies Recommendation system",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# LOAD DATA

@st.cache_data
def load_data():

    try:
        movies = pd.read_csv("tmdb_5000_movies.csv")
        credits = pd.read_csv("tmdb_5000_credits.csv")
    except FileNotFoundError as e:
        st.error(
            f"Couldn't find a required data file ({e.filename}). "
            "Make sure tmdb_5000_movies.csv and tmdb_5000_credits.csv "
            "are in the same folder as app.py."
        )
        st.stop()

    movies = movies.merge(credits, on="title")

    # A handful of titles in this dataset repeat (e.g. re-releases). Keeping
    # duplicates breaks the title -> row-index lookup used everywhere else
    # (the second occurrence would silently overwrite the first), so we
    # de-dupe right after the merge, before anything else touches the index.
    movies = movies.drop_duplicates(subset="title", keep="first").reset_index(drop=True)

    # preprocessing
    movies["overview"] = movies["overview"].fillna("")
    movies["vote_average"] = pd.to_numeric(movies.get("vote_average"), errors="coerce").fillna(0)
    movies["vote_count"] = pd.to_numeric(movies.get("vote_count"), errors="coerce").fillna(0)

    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert_cast)
    movies["director"] = movies["crew"].apply(fetch_director)

    # Tag string used for similarity. Genres/keywords/cast/director are
    # collapsed to single tokens (see collapse()) so e.g. "Tom Hanks" isn't
    # diluted into the generic words "tom" and "hanks" by TF-IDF — cast and
    # director carry real recommendation signal now instead of being
    # computed and thrown away.
    movies["tags2"] = (
        movies["genres"].apply(lambda x: " ".join(collapse(g) for g in x))
        + " "
        + movies["keywords"].apply(lambda x: " ".join(collapse(k) for k in x))
        + " "
        + movies["cast"].apply(lambda x: " ".join(collapse(c) for c in x))
        + " "
        + movies["director"].apply(lambda x: " ".join(collapse(d) for d in x))
        + " "
        + movies["overview"]
    )

    tf = TfidfVectorizer(max_features=5000, stop_words="english")

    v1 = tf.fit_transform(movies["tags2"])

    similarity = cosine_similarity(v1)

    movies_indices = pd.Series(
        movies.index,
        index=movies["title"]
    ).to_dict()

    return movies, similarity, movies_indices
movies, similarity, movies_indices = load_data()

# DERIVED, READ-ONLY STATS (no model/logic changes — display only)

@st.cache_data
def compute_similarity_stats(_similarity):
    """Purely descriptive stats about the similarity matrix, for the
    Model Rating & Accuracy tab. Does not alter recommendation logic."""
    n = _similarity.shape[0]
    # sample to keep this fast on large matrices
    rng = np.random.default_rng(42)
    sample_size = min(400, n)
    idx = rng.choice(n, size=sample_size, replace=False)
    sub = _similarity[np.ix_(idx, idx)]
    off_diag = sub[~np.eye(sample_size, dtype=bool)]
    return {
        "mean": float(np.mean(off_diag)),
        "median": float(np.median(off_diag)),
        "p90": float(np.percentile(off_diag, 90)),
        "std": float(np.std(off_diag)),
        "hist_values": off_diag,
        "n_movies": n,
    }

@st.cache_data
def compute_rating_stats(_movies):
    """Descriptive stats over TMDB's own vote_average / vote_count columns,
    used as the 'user ratings' signal. Read-only, no scoring logic touched."""
    df = _movies.copy()
    if "vote_average" not in df.columns:
        df["vote_average"] = np.nan
    if "vote_count" not in df.columns:
        df["vote_count"] = 0
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce")
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0)
    return df

similarity_stats = compute_similarity_stats(similarity)
ratings_df = compute_rating_stats(movies)


# CUSTOM CSS — cinema marquee identity

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/tabler-icons/2.44.0/iconfont/tabler-icons.min.css');

    :root {
        --void: #0a0a0c;
        --surface: #1c1c20;
        --surface-raised: #232328;
        --hairline: #2e2e34;
        --cream: #f4f1e8;
        --cream-dim: #a8a59c;
        --marquee-red: #e8482c;
        --marquee-red-dim: rgba(232, 72, 44, 0.16);
        --brass: #d4a843;
        --brass-dim: rgba(212, 168, 67, 0.14);
        --teal: #3aa6a0;
        --teal-dim: rgba(58, 166, 160, 0.14);
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(232,72,44,0.05) 0%, transparent 45%),
            radial-gradient(circle at 85% 10%, rgba(212,168,67,0.04) 0%, transparent 40%),
            var(--void);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu, footer, header { visibility: hidden; }

    /* ---------------- MARQUEE HERO ---------------- */
    .marquee-frame {
        position: relative;
        border: 2px solid var(--hairline);
        border-radius: 4px;
        padding: 36px 24px 30px 24px;
        margin: 4px 0 22px 0;
        background: linear-gradient(180deg, #111114 0%, var(--void) 100%);
        overflow: hidden;
    }
    .marquee-frame::before, .marquee-frame::after {
        content: '';
        position: absolute;
        top: 10px; bottom: 10px;
        width: 14px;
        background-image: radial-gradient(circle, var(--void) 3.5px, var(--hairline) 4px, var(--hairline) 4.5px, transparent 4.5px);
        background-size: 14px 22px;
        background-repeat: repeat-y;
    }
    .marquee-frame::before { left: 0; }
    .marquee-frame::after { right: 0; }

    .chase-border {
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        overflow: hidden;
    }
    .chase-border span {
        position: absolute;
        width: 22px; height: 3px;
        background: var(--marquee-red);
        animation: chase 3.2s linear infinite;
        box-shadow: 0 0 6px var(--marquee-red);
    }
    @keyframes chase {
        0% { left: -22px; }
        100% { left: 100%; }
    }

    .hero-wrap {
        text-align: center;
        padding: 4px 28px 0 28px;
        position: relative;
        z-index: 1;
    }
    .hero-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 18px;
        letter-spacing: 4px;
        color: #00cc44;
        text-transform: uppercase;
        margin-bottom: 14px;
        font-weight: 700;
    }
    .hero-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 72px;
        letter-spacing: 6px;
        color:  #ff2222;;
        margin: 0;
        line-height: 1;
        text-shadow: 0 0 24px rgba(255,34,34,0.2);
    }
    .hero-title span { color: #2266ff; }
    .hero-subtitle {
        color: var(--cream-dim);
        font-size: 14.5px;
        margin-top: 10px;
        letter-spacing: 0.4px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ---------------- TOP NAV TABS ---------------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        justify-content: center;
        border-bottom: 1px solid var(--hairline);
        margin-bottom: 28px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        background: transparent;
        border-radius: 4px 4px 0 0;
        padding: 0 20px;
        color: var(--cream-dim);
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: var(--cream) !important;
        background: var(--surface) !important;
        border-bottom: 2px solid var(--marquee-red) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { background: transparent; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ---------------- CENTERED SEARCH CONSOLE ----------------
       Targets the Streamlit-generated wrapper for st.container(key=
       "search_console") rather than a hand-written <div>. Streamlit
       renders each st.markdown/st.radio/st.button call as its own
       sibling node — an opening "<div>" in one markdown call and a
       closing "</div>" in another never actually nest the widgets in
       between, so the box rendered empty and the controls fell outside
       it. st.container(key=...) is a real, single DOM element that the
       controls are genuinely children of, so this same styling now
       wraps real content. */
    .st-key-search_console {
        max-width: 760px;
        margin: 0 auto 30px auto;
        background: var(--surface);
        border: 1px solid var(--hairline);
        border-radius: 10px;
        padding: 26px 30px 22px 30px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.35);
        position: relative;
    }
    .st-key-search_console::before {
        content: '';
        position: absolute;
        top: -1px; left: 50%;
        transform: translateX(-50%);
        width: 70px; height: 2px;
        background: var(--marquee-red);
        box-shadow: 0 0 10px var(--marquee-red);
    }
    .search-console-label {
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: var(--brass);
        margin-bottom: 16px;
    }

    /* center the radio pills */
    .st-key-search_console div[role="radiogroup"] {
        justify-content: center;
        gap: 8px;
    }
    .st-key-search_console div[role="radiogroup"] label {
        background: var(--surface-raised);
        border: 1px solid var(--hairline);
        border-radius: 20px;
        padding: 7px 16px !important;
        transition: all 0.2s ease;
    }
    .st-key-search_console div[role="radiogroup"] label:hover {
        border-color: var(--marquee-red);
    }

    /* ---------------- SIDEBAR (kept minimal, for posters note only) ---------------- */
    section[data-testid="stSidebar"] {
        border-right: 1px solid var(--hairline);
    }

    /* ---------------- SECTION HEADERS ---------------- */
    .section-row {
        display: flex;
        align-items: baseline;
        gap: 12px;
        margin: 4px 0 2px 0;
        justify-content: center;
    }
    .section-ticket {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: var(--void);
        background: var(--brass);
        padding: 2px 8px;
        border-radius: 3px;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: var(--cream);
        margin: 0;
    }
    .section-caption {
        color: var(--cream-dim);
        font-size: 13.5px;
        margin: 6px 0 22px 0;
        font-family: 'JetBrains Mono', monospace;
        text-align: center;
    }
    .section-caption b { color: var(--cream); font-weight: 500; }

    /* ---------------- KPI / STAT CARDS ---------------- */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin: 22px 0 30px 0;
    }
    .kpi-card {
        background: var(--surface);
        border: 1px solid var(--hairline);
        border-radius: 8px;
        padding: 20px 18px;
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 3px; height: 100%;
        background: var(--accent, var(--marquee-red));
    }
    .kpi-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10.5px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: var(--cream-dim);
        margin-bottom: 10px;
    }
    .kpi-value {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 38px;
        letter-spacing: 1px;
        color: var(--cream);
        line-height: 1;
    }
    .kpi-unit {
        font-size: 16px;
        color: var(--cream-dim);
        margin-left: 4px;
    }
    .kpi-sub {
        font-size: 11.5px;
        color: var(--cream-dim);
        margin-top: 8px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ---------------- INFO / ABOUT PANEL ---------------- */
    .info-panel {
        background: var(--surface);
        border: 1px solid var(--hairline);
        border-radius: 8px;
        padding: 24px 26px;
        margin-bottom: 18px;
    }
    .info-panel h4 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        color: var(--brass);
        margin: 0 0 14px 0;
    }
    .info-panel p {
        color: var(--cream-dim);
        font-size: 14px;
        line-height: 1.75;
        margin: 0 0 12px 0;
    }
    .info-panel p:last-child { margin-bottom: 0; }
    .info-panel b { color: var(--cream); font-weight: 600; }
    .pipeline-steps {
        list-style: none;
        margin: 14px 0 0 0;
        padding: 0;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .pipeline-step {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        font-size: 13.5px;
        color: var(--cream-dim);
        line-height: 1.5;
    }
    .pipeline-num {
        flex-shrink: 0;
        width: 24px; height: 24px;
        border-radius: 50%;
        background: var(--marquee-red-dim);
        color: var(--marquee-red);
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 1px;
    }
    .pipeline-step b { color: var(--cream); }

    .badge-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px; }
    .tech-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11.5px;
        color: var(--teal);
        background: var(--teal-dim);
        padding: 4px 10px;
        border-radius: 4px;
        letter-spacing: 0.3px;
    }

    /* ---------------- MOVIE CARD (ticket stub) ---------------- */
    .movie-card {
        position: relative;
        background: var(--surface);
        border: 1px solid var(--hairline);
        border-radius: 6px;
        overflow: hidden;
        transition: transform 0.3s cubic-bezier(.2,.8,.2,1), border-color 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
        text-decoration: none;
        display: block;
        cursor: pointer;
    }
    .movie-card:hover {
        transform: translateY(-8px);
        border-color: var(--marquee-red);
        box-shadow: 0 18px 36px rgba(0,0,0,0.45), 0 0 0 1px rgba(232,72,44,0.3);
    }
    .movie-card:focus-visible {
        outline: 2px solid var(--brass);
        outline-offset: 2px;
    }

    /* ticket notch */
    .movie-card::before {
        content: '';
        position: absolute;
        top: calc(133.33% - 9px);
        left: -9px;
        width: 18px; height: 18px;
        background: var(--void);
        border-radius: 50%;
        z-index: 3;
    }
    .movie-card::after {
        content: '';
        position: absolute;
        top: calc(133.33% - 9px);
        right: -9px;
        width: 18px; height: 18px;
        background: var(--void);
        border-radius: 50%;
        z-index: 3;
    }

    .poster-wrap {
        position: relative;
        width: 100%;
        aspect-ratio: 1 / 1;
        overflow: hidden;
        max-width: 300px;  
        margin: 0 auto; 
    }
    .movie-poster {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: None;
        transition: transform 0.4s ease, filter 0.4s ease;
        filter: saturate(0.92);
    }
    .movie-card:hover .movie-poster {
        transform: scale(1.07);
        filter: saturate(1.05);
    }

    .poster-overlay {
        position: absolute;
        inset: 0;
        background: linear-gradient(180deg, rgba(10,10,12,0.1) 0%, rgba(10,10,12,0.97) 76%);
        opacity: 0;
        transition: opacity 0.28s ease;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        padding: 14px;
    }
    .movie-card:hover .poster-overlay {
        opacity: 1;
    }
    .overlay-overview {
        color: var(--cream-dim);
        font-size: 11.5px;
        line-height: 1.55;
        margin-bottom: 10px;
        display: -webkit-box;
        -webkit-line-clamp: 6;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .overlay-genres {
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
        margin-bottom: 9px;
    }
    .genre-chip {
        background: var(--marquee-red-dim);
        color: #ff8a73;
        font-size: 10px;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 3px;
        letter-spacing: 0.3px;
        font-family: 'JetBrains Mono', monospace;
    }
    .overlay-cta {
        display: flex;
        align-items: center;
        gap: 6px;
        color: var(--brass);
        font-size: 11.5px;
        font-weight: 600;
        letter-spacing: 0.4px;
        font-family: 'JetBrains Mono', monospace;
    }

    .movie-info {
        padding: 16px 14px 14px 14px;
        position: relative;
    }
    .movie-info::before {
        content: '';
        position: absolute;
        top: 0; left: 14px; right: 14px;
        border-top: 1px dashed var(--hairline);
    }
    .movie-title {
        font-size: 14.5px;
        font-weight: 600;
        color: var(--cream);
        margin: 0 0 7px 0;
        line-height: 1.3;
        min-height: 38px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .movie-meta {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 12px;
        color: var(--cream-dim);
        font-family: 'JetBrains Mono', monospace;
    }
    .rating-badge {
        background: var(--brass-dim);
        color: var(--brass);
        padding: 2px 8px;
        border-radius: 3px;
        font-weight: 600;
        font-size: 11.5px;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }

    /* ---------------- LEADERBOARD / RATINGS TABLE ROWS ---------------- */
    .rank-row {
        display: grid;
        grid-template-columns: 44px 1fr 110px 130px;
        align-items: center;
        gap: 14px;
        padding: 12px 16px;
        border-radius: 6px;
        border: 1px solid transparent;
        transition: background 0.2s ease, border-color 0.2s ease;
    }
    .rank-row:hover {
        background: var(--surface);
        border-color: var(--hairline);
    }
    .rank-num {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 20px;
        color: var(--cream-dim);
        text-align: center;
    }
    .rank-poster {
        width: 48px;
        aspect-ratio: 2/3;
        object-fit: cover;
        border-radius: 4px;
        border: 1px solid var(--hairline);
    }
    .rank-title {
        font-size: 14px;
        font-weight: 600;
        color: var(--cream);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .rank-votes {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11.5px;
        color: var(--cream-dim);
    }
    .rank-score-bar-bg {
        width: 100%;
        height: 7px;
        background: var(--surface-raised);
        border-radius: 4px;
        overflow: hidden;
    }
    .rank-score-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--brass), var(--marquee-red));
        border-radius: 4px;
    }
    .rank-score-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12.5px;
        color: var(--brass);
        font-weight: 600;
        margin-left: 8px;
        white-space: nowrap;
    }

    /* ---------------- SKELETON LOADING ---------------- */
    .skeleton-card {
        background: var(--surface);
        border: 1px solid var(--hairline);
        border-radius: 6px;
        overflow: hidden;
        height: 100%;
    }
    .skeleton-poster {
        width: 100%;
        aspect-ratio: 2 / 3;
        background: linear-gradient(100deg, var(--surface) 30%, var(--surface-raised) 50%, var(--surface) 70%);
        background-size: 200% 100%;
        animation: shimmer 1.4s ease-in-out infinite;
    }
    .skeleton-line {
        height: 10px;
        margin: 14px 14px 8px 14px;
        border-radius: 3px;
        background: linear-gradient(100deg, var(--surface) 30%, var(--surface-raised) 50%, var(--surface) 70%);
        background-size: 200% 100%;
        animation: shimmer 1.4s ease-in-out infinite;
    }
    .skeleton-line.short { width: 50%; }
    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }

    /* ---------------- BUTTONS ---------------- */
    .stButton > button {
        background: var(--marquee-red);
        color: var(--cream);
        font-weight: 700;
        border: none;
        border-radius: 4px;
        padding: 11px 0;
        letter-spacing: 0.6px;
        transition: all 0.2s ease;
        font-family: 'Inter', sans-serif;
    }
    .stButton > button:hover {
        background: #ff5736;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(232,72,44,0.35);
    }
    .stButton > button:active {
        transform: translateY(0px) scale(0.98);
    }

    /* ---------------- EMPTY / ERROR STATES ---------------- */
    .state-block {
        border: 1px dashed var(--hairline);
        border-radius: 6px;
        padding: 40px 28px;
        text-align: center;
        margin-top: 8px;
    }
    .state-icon {
        font-size: 32px;
        color: var(--cream-dim);
        margin-bottom: 14px;
        display: block;
    }
    .state-title {
        font-size: 17px;
        font-weight: 600;
        color: var(--cream);
        margin-bottom: 6px;
    }
    .state-text {
        font-size: 13.5px;
        color: var(--cream-dim);
        max-width: 420px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* ---------------- MISC ---------------- */
    .block-container { padding-top: 1.5rem; max-width: 1180px; }
    hr { border-color: var(--hairline) !important; }

    /* ---------------- FOOTER CREDITS ---------------- */
    .credits-roll {
        position: relative;
        margin-top: 48px;
        padding: 28px 24px 22px 24px;
        border-top: 1px solid var(--hairline);
        text-align: center;
        overflow: hidden;
    }
    .credits-roll::before {
        content: '';
        position: absolute;
        top: -1px; left: 50%;
        transform: translateX(-50%);
        width: 60px; height: 1px;
        background: var(--marquee-red);
        box-shadow: 0 0 8px var(--marquee-red);
    }
    .credits-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        letter-spacing: 3px;
        color: var(--cream-dim);
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .credits-name {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 26px;
        letter-spacing: 2.5px;
        color: var(--cream);
        margin-bottom: 14px;
    }
    .credits-links {
        display: flex;
        justify-content: center;
        gap: 10px;
        flex-wrap: wrap;
    }
    .credit-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12.5px;
        color: var(--cream-dim);
        background: var(--surface);
        border: 1px solid var(--hairline);
        padding: 7px 14px;
        border-radius: 20px;
        text-decoration: none;
        transition: border-color 0.2s ease, color 0.2s ease, transform 0.2s ease;
    }
    .credit-pill:hover {
        border-color: var(--marquee-red);
        color: var(--cream);
        transform: translateY(-2px);
    }
    .credit-pill i {
        font-size: 14px;
        color: var(--brass);
    }
    .credits-tagline {
        margin-top: 16px;
        font-size: 11px;
        color: var(--cream-dim);
        opacity: 0.6;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.5px;
    }

    @media (max-width: 900px) {
        .kpi-grid { grid-template-columns: repeat(2, 1fr); }
        .rank-row { grid-template-columns: 32px 48px 1fr 90px; }
        .rank-votes { display: none; }
        .hero-title { font-size: 48px; letter-spacing: 3px; }
    }

    @media (prefers-reduced-motion: reduce) {
        .chase-border span, .skeleton-poster, .skeleton-line { animation: none !important; }
        .movie-card, .movie-poster { transition: none !important; }
    }
</style>
""", unsafe_allow_html=True)

# HERO HEADER — marquee with chasing light border
st.markdown("""
<div class="marquee-frame">
    <div class="chase-border"><span></span></div>
    <div class="hero-wrap">
        <div class="hero-eyebrow">Now showing</div>
        <div class="hero-title">Movie Recomme<span>ndation-system</span></div>
        <div class="hero-subtitle">find your next favorite movie, instantly</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Warn if API key missing, but don't block the app
if not get_api_key():
    st.info("Posters are showing as placeholders — add your TMDB API key to `.streamlit/secrets.toml` as `TMDB_API_KEY` to enable them.")

COLS_PER_ROW = 5


# TOP-LEVEL NAVIGATION TABS

tab_recommend, tab_model, tab_ratings, tab_about = st.tabs([
    "🎬  Recommend",
    "📊  Model Rating & Accuracy",
    "⭐  User Ratings",
    "ℹ️  About"
])


# TAB 1 — RECOMMEND (centered search console)

with tab_recommend:

    all_genres = sorted(set(
        genre.strip().strip("'")
        for genres in movies['genres'].dropna()
        for genre in (genres if isinstance(genres, list) else [])
    ))

    # st.container(key=...) is a real DOM element, unlike the previous
    # st.markdown('<div>') ... st.markdown('</div>') pattern — so every
    # widget created inside this `with` block is a genuine child of the
    # styled box (see ".st-key-search_console" in the CSS above) instead
    # of rendering as an empty box with the controls orphaned outside it.
    with st.container(key="search_console"):
        st.markdown('<div class="search-console-label">▸ Search the reel</div>', unsafe_allow_html=True)

        mode = st.radio(
            "How do you want to search?",
            ["By movie", "By genre", "Movie + genre"],
            label_visibility="collapsed",
            horizontal=True,
            key="search_mode"
        )

        movie_input = None
        genre_input = None

        if mode == "By movie":
            movie_input = st.selectbox("Pick a movie you like", movies['title'].values, key="movie_only")

        elif mode == "By genre":
            genre_input = st.selectbox("Pick a genre", all_genres, key="genre_only")

        elif mode == "Movie + genre":
            c1, c2 = st.columns(2)
            with c1:
                movie_input = st.selectbox("Pick a movie you like", movies['title'].values, key="movie_combo")
            with c2:
                genre_choice = st.selectbox("Narrow down by genre", ["All"] + all_genres, key="genre_combo")
                genre_input = None if genre_choice == "All" else genre_choice

        run_search = st.button("🎟️  Recommend", use_container_width=True)

        st.markdown(
            '<div style="text-align:center; margin-top:10px; font-size:11.5px; color:var(--cream-dim); font-family:\'JetBrains Mono\',monospace;">'
            ' Powered by TMDB · typos in movie titles are auto-corrected'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div style="text-align:center; margin-top:6px; font-size:12px; color:var(--brass); font-family:\'JetBrains Mono\',monospace; animation: pulse 2s infinite;">'
            '🎬 Click any movie card for full details on TMDB'
            '</div>',
            unsafe_allow_html=True
        )


    def render_skeleton_grid(n=10):
        """Shows shimmering placeholder cards while posters are being fetched."""
        for row_start in range(0, n, COLS_PER_ROW):
            cols = st.columns(COLS_PER_ROW)
            for col in cols:
                with col:
                    st.markdown("""
                        <div class="skeleton-card">
                            <div class="skeleton-poster"></div>
                            <div class="skeleton-line"></div>
                            <div class="skeleton-line short"></div>
                        </div>
                    """, unsafe_allow_html=True)


    if run_search:
        result = recommend(
            movies=movies,
            movies_indices=movies_indices,
            similarity=similarity,
            movie_name=movie_input,
            genre=genre_input
        )

        if result["correction_note"]:
            st.warning(f"📝 {result['correction_note']}")

        if result["status"] == "not_found":
            st.markdown(f"""
                <div class="state-block">
                    <i class="ti ti-mood-confuzed state-icon" aria-hidden="true"></i>
                    <div class="state-title">No match found</div>
                    <div class="state-text">{result['message']} Try a different title or pick another genre above.</div>
                </div>
            """, unsafe_allow_html=True)

        elif result["recommendations"]:
            label = result["matched_title"] or genre_input or "your picks"
            st.markdown("""
                <div class="section-row">
                    <span class="section-ticket">REC</span>
                    <p class="section-header">Recommended for you</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f'<div class="section-caption">based on <b>{label}</b></div>', unsafe_allow_html=True)

            recommendations = result["recommendations"]
            placeholder = st.empty()

            # Show skeleton cards immediately so the wait doesn't feel dead
            with placeholder.container():
                render_skeleton_grid(len(recommendations))

            details = [fetch_movie_details(title) for title in recommendations]

            # Replace skeletons with real cards
            with placeholder.container():
                for row_start in range(0, len(recommendations), COLS_PER_ROW):
                    row_titles = recommendations[row_start:row_start + COLS_PER_ROW]
                    row_details = details[row_start:row_start + COLS_PER_ROW]
                    cols = st.columns(COLS_PER_ROW)

                    for col, title, info in zip(cols, row_titles, row_details):
                        with col:
                            rating_html = (
                                f'<span class="rating-badge"><i class="ti ti-star-filled" style="font-size:11px"></i> {info["rating"]}</span>'
                                if info["rating"] else ""
                            )
                            year_html = f'<span>{info["year"]}</span>' if info["year"] else "<span></span>"

                            genre_chips = "".join(
                                f'<span class="genre-chip">{g}</span>' for g in info["genres"][:3]
                            )

                            overview_text = info["overview"]
                            if len(overview_text) > 220:
                                overview_text = overview_text[:220].rsplit(" ", 1)[0] + "…"

                            link_url = info["tmdb_url"] or (
                                f"https://www.themoviedb.org/search?query={title.replace(' ', '+')}"
                            )

                            st.markdown(f"""
                                <a class="movie-card" href="{link_url}" target="_blank" rel="noopener noreferrer">
                                    <div class="poster-wrap">
                                        <img class="movie-poster" src="{info['poster']}" alt="{title}" loading="lazy">
                                        <div class="poster-overlay">
                                            <div class="overlay-genres">{genre_chips}</div>
                                            <div class="overlay-overview">{overview_text}</div>
                                            <div class="overlay-cta"><i class="ti ti-external-link"></i> View on TMDB</div>
                                        </div>
                                    </div>
                                    <div class="movie-info">
                                        <p class="movie-title">{title}</p>
                                        <div class="movie-meta">
                                            {year_html}
                                            {rating_html}
                                        </div>
                                    </div>
                                </a>
                            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="state-block">
                    <i class="ti ti-filter-off state-icon" aria-hidden="true"></i>
                    <div class="state-title">Nothing matches these filters</div>
                    <div class="state-text">Try loosening the genre filter or pick a different movie to compare against.</div>
                </div>
            """, unsafe_allow_html=True)

    else:
        # Empty state before first search
        st.markdown("""
            <div class="state-block">
                <i class="ti ti-movie state-icon" aria-hidden="true"></i>
                <div class="state-title">Ready when you are</div>
                <div class="state-text">Pick a movie, a genre, or both above, then hit <b>Recommend</b> to roll the reel.</div>
            </div>
        """, unsafe_allow_html=True)

# TAB 2 — MODEL RATING & ACCURACY (demo / illustrative metrics)

with tab_model:

    st.markdown("""
        <div class="section-row">
            <span class="section-ticket">SYS</span>
            <p class="section-header">Model rating &amp; accuracy</p>
        </div>
        <div class="section-caption">a look under the hood of the recommendation engine</div>
    """, unsafe_allow_html=True)

    st.info(
        "📌 The figures below are **illustrative demo metrics** meant to show how a model "
        "report card would look. They are not the result of a formal offline evaluation "
        "against held-out user data.",
        icon="📌"
    )

    # ---- Demo / mock headline metrics ----
    demo_precision_at_10 = 0.80
    demo_recall_at_10 = 0.79
    demo_map = 0.71
    demo_coverage = 0.91

    st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card" style="--accent:#e8482c;">
                <div class="kpi-eyebrow">Precision@10</div>
                <div class="kpi-value">{demo_precision_at_10*100:.0f}<span class="kpi-unit">%</span></div>
                <div class="kpi-sub">relevant items in top 10 (demo)</div>
            </div>
            <div class="kpi-card" style="--accent:#d4a843;">
                <div class="kpi-eyebrow">Recall@10</div>
                <div class="kpi-value">{demo_recall_at_10*100:.0f}<span class="kpi-unit">%</span></div>
                <div class="kpi-sub">relevant items surfaced (demo)</div>
            </div>
            <div class="kpi-card" style="--accent:#3aa6a0;">
                <div class="kpi-eyebrow">Mean Avg. Precision</div>
                <div class="kpi-value">{demo_map:.2f}</div>
                <div class="kpi-sub">ranking quality, MAP (demo)</div>
            </div>
            <div class="kpi-card" style="--accent:#e8482c;">
                <div class="kpi-eyebrow">Catalog coverage</div>
                <div class="kpi-value">{demo_coverage*100:.0f}<span class="kpi-unit">%</span></div>
                <div class="kpi-sub">titles reachable as a rec (demo)</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        st.markdown("""
            <div class="info-panel">
                <h4>▸ Similaristy score distribution</h4>
                <p>This is computed live from the actual TF-IDF cosine similarity
                matrix the engine uses — a sample of pairwise scores across the
                catalog, not a demo number.</p>
            </div>
        """, unsafe_allow_html=True)

        hist_df = pd.DataFrame({"similarity": similarity_stats["hist_values"]})
        st.bar_chart(
            np.histogram(hist_df["similarity"], bins=24)[0],
            height=240,
            color="#e8482c"
        )

        s1, s2, s3 = st.columns(3)
        s1.metric("Mean", f"{similarity_stats['mean']:.3f}")
        s2.metric("Median", f"{similarity_stats['median']:.3f}")
        s3.metric("Std. dev.", f"{similarity_stats['std']:.3f}")

    with col_right:
        st.markdown("""
            <div class="info-panel">
                <h4>▸ Demo confusion breakdown</h4>
                <p>A mock breakdown of recommendation outcomes on a hypothetical
                100-recommendation sample, for illustration only.</p>
            </div>
        """, unsafe_allow_html=True)

        demo_breakdown = pd.DataFrame({
            "outcome": ["Relevant pick", "Loosely relevant", "Off-target"],
            "count": [52, 26, 22]
        }).set_index("outcome")
        st.bar_chart(demo_breakdown, height=240, color="#d4a843")

        st.markdown("""
            <div class="info-panel" style="margin-top:14px;">
                <h4>▸ How scoring works</h4>
                <p><b>Content-based filtering:</b> each movie's genres, keywords,
                cast, and director are combined into a single tag string along
                with the overview, vectorized with <b>TF-IDF</b>, then compared
                pairwise using <b>cosine similarity</b>. No user behavior data is
                required — this is why coverage is high even for less-watched
                titles.</p>
            </div>
        """, unsafe_allow_html=True)

    st.caption(f"Similarity matrix computed over {similarity_stats['n_movies']:,} titles in the catalog.")


# TAB 3 — USER RATINGS (TMDB vote_average / vote_count, real data)

with tab_ratings:

    st.markdown("""
        <div class="section-row">
            <span class="section-ticket">★</span>
            <p class="section-header">User ratings</p>
        </div>
        <div class="section-caption">crowd scores sourced from TMDB's <b>vote_average</b> &amp; <b>vote_count</b></div>
    """, unsafe_allow_html=True)

    rc1, rc2, rc3 = st.columns([1, 1, 1])
    with rc1:
        min_votes = st.slider("Minimum number of votes", 0, 5000, 500, step=50)
    with rc2:
        sort_choice = st.selectbox("Sort by", ["Highest rated", "Most voted"])
    with rc3:
        top_n = st.selectbox("Show top", [10, 20, 50], index=0)

    filtered = ratings_df[ratings_df["vote_count"] >= min_votes].copy()
    filtered = filtered.dropna(subset=["vote_average"])

    if sort_choice == "Highest rated":
        filtered = filtered.sort_values(["vote_average", "vote_count"], ascending=[False, False])
    else:
        filtered = filtered.sort_values(["vote_count", "vote_average"], ascending=[False, False])

    filtered = filtered.head(top_n).reset_index(drop=True)

    if filtered.empty:
        st.markdown("""
            <div class="state-block">
                <i class="ti ti-filter-off state-icon" aria-hidden="true"></i>
                <div class="state-title">No titles meet that vote threshold</div>
                <div class="state-text">Lower the minimum vote count to see more results.</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Titles shown", f"{len(filtered)}")
        m2.metric("Avg. rating (shown)", f"{filtered['vote_average'].mean():.2f} / 10")
        m3.metric("Avg. vote count (shown)", f"{filtered['vote_count'].mean():,.0f}")

        st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)

        poster_lookup = {}

        for i, row in filtered.iterrows():
            info = poster_lookup.get(row["title"], {})
            poster_url = info.get("poster", "")
            score_pct = (row["vote_average"] / 10.0) * 100

            st.markdown(f"""
                <div class="rank-row">
                    <div class="rank-num">{i + 1}</div>
                    <div class="rank-title" style="flex:1; padding-left:10px;">{row['title']}</div>
                    <div class="rank-votes">{int(row['vote_count']):,} votes</div>
                    <div style="display:flex; align-items:center;">
                        <div class="rank-score-bar-bg">
                            <div class="rank-score-bar-fill" style="width:{score_pct:.0f}%;"></div>
                        </div>
                        <span class="rank-score-text">{row['vote_average']:.1f}/10</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)


# TAB 4 — ABOUT

with tab_about:

    st.markdown("""
        <div class="section-row">
            <span class="section-ticket">i</span>
            <p class="section-header">About:- Movie Recommendation-system</p>
        </div>
        <div class="section-caption">what's running behind the marquee</div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns([1.2, 1], gap="large")

    with a1:
        st.markdown("""
            <div class="info-panel">
                <h4>▸ What this is</h4>
                <p><b>Movies recommendation system</b> is a content-based movie recommender. Tell it a
                movie you like, a genre, or both, and it finds titles with the
                closest thematic fingerprint — without needing any of your
                watch history or ratings.</p>
            </div>

            <div class="info-panel">
                <h4>▸ How a recommendation is made</h4>
                <ul class="pipeline-steps">
                    <li class="pipeline-step">
                        <span class="pipeline-num">1</span>
                        <span><b>Tag building</b> — each movie's genres, keywords,
                        top-billed cast, and director are merged with the overview
                        into one descriptive tag string.</span>
                    </li>
                    <li class="pipeline-step">
                        <span class="pipeline-num">2</span>
                        <span><b>Vectorizing</b> — tags are converted into numeric vectors
                        with <b>TF-IDF</b>, which weighs distinctive words more heavily.</span>
                    </li>
                    <li class="pipeline-step">
                        <span class="pipeline-num">3</span>
                        <span><b>Similarity scoring</b> — every movie is compared to every
                        other movie with <b>cosine similarity</b> to build a similarity matrix.</span>
                    </li>
                    <li class="pipeline-step">
                        <span class="pipeline-num">4</span>
                        <span><b>Ranking</b> — for your chosen movie/genre, the closest-scoring
                        titles are surfaced as recommendations.</span>
                    </li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
            <div class="info-panel">
                <h4>▸ Data &amp; tech</h4>
                <p>Built on the TMDB 5000 movie &amp; credits dataset, with live
                poster/overview lookups from the TMDB API.</p>
                <div class="badge-row">
                    <span class="tech-badge">Python</span>
                    <span class="tech-badge">Streamlit</span>
                    <span class="tech-badge">pandas</span>
                    <span class="tech-badge">scikit-learn</span>
                    <span class="tech-badge">TF-IDF</span>
                    <span class="tech-badge">Cosine Similarity</span>
                    <span class="tech-badge">TMDB API</span>
                </div>
            </div>

            <div class="info-panel">
                <h4>▸ Good to know</h4>
                <p>This is a <b>content-based</b> system, not collaborative
                filtering — it doesn't learn from what other users liked, so
                recommendations stay consistent and explainable, but it won't
                pick up on trends or crowd taste.</p>
                <p>Typos in the movie search are auto-corrected to the closest
                title in the catalog.</p>
            </div>
        """, unsafe_allow_html=True)


# FOOTER CREDITS (shown on every tab)

st.markdown("""
    <div class="credits-roll">
        <div class="credits-eyebrow" style="color: #ffb6c1; text-shadow: 0 0 10px rgba(255,182,193,0.8), 0 0 20px rgba(255,182,193,0.5), 0 0 40px rgba(255,182,193,0.3);">Directed &amp; built by</div>
        <div class="credits-name" style="font-family: 'Bebas Neue', sans-serif; font-size: 34px; letter-spacing: 4px; background: linear-gradient(90deg, #ff0000, #0066ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 14px; display: inline-block; animation: redBlueGlow 2.5s ease-in-out infinite;">Karan Farne</div>
        <div class="credits-links">
            <a class="credit-pill" href="https://github.com/karanfarane-prog" target="_blank">
                GitHub
            </a>
            <a class="credit-pill" href="https://www.linkedin.com/in/karan-farane/" target="_blank">
                LinkedIn
            </a>
        </div>
        <div class="credits-tagline" style="color: white; font-size: 20px; font-weight: bold; text-shadow: 0 0 10px rgba(255,255,255,0.8), 0 0 20px rgba(255,255,255,0.5), 0 0 40px rgba(255,255,255,0.3); font-family: 'JetBrains Mono', monospace; letter-spacing: 4px; margin-top: 16px; animation: whitePulse 2s ease-in-out infinite;"> END </div>
    </div>
""", unsafe_allow_html=True)


