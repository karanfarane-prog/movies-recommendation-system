import os
import difflib
import functools

import pandas as pd
import requests

try:
    import streamlit as st
except ImportError:  # keeps this module importable/testable outside Streamlit
    st = None

TMDB_API_BASE = "https://api.themoviedb.org/3/movie/"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500/"
PLACEHOLDER_POSTER = "https://via.placeholder.com/500x750/1c1c20/a8a59c?text=Poster"
REQUEST_TIMEOUT = 6  # seconds — fail fast rather than hang the UI on a slow request

# Global movies dataframe for local poster lookup
_movies_df = None


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------

def get_api_key():
    """
    Looks for a TMDB v3 API key, checking (in order):
      1. st.secrets["TMDB_API_KEY"]   (.streamlit/secrets.toml)
      2. the TMDB_API_KEY environment variable
    Returns None if neither is set — callers must treat that as
    "no live data available" and fall back to placeholders, not crash.
    """
    if st is not None:
        try:
            if "TMDB_API_KEY" in st.secrets and st.secrets["TMDB_API_KEY"]:
                return st.secrets["TMDB_API_KEY"]
        except Exception:
            pass  # no secrets.toml at all — fine, just fall through

    return os.environ.get("TMDB_API_KEY") or None


# ---------------------------------------------------------------------------
# Load movies data on demand
# ---------------------------------------------------------------------------

def _load_movies_if_needed():
    """Lazily load movies CSV if not already loaded"""
    global _movies_df
    if _movies_df is None:
        try:
            _movies_df = pd.read_csv("tmdb_5000_movies.csv")
            print(f"📚 Loaded {len(_movies_df)} movies from CSV")
        except Exception as e:
            print(f"⚠️ Could not load CSV: {e}")
            _movies_df = pd.DataFrame()
    return _movies_df


def _placeholder_details():
    return {
        "poster": PLACEHOLDER_POSTER,
        "rating": None,
        "year": None,
        "genres": [],
        "overview": "",
        "tmdb_url": None,
    }


# ---------------------------------------------------------------------------
# Local poster lookup from CSV
# ---------------------------------------------------------------------------

def _get_local_poster(title):
    """
    Try to find movie in local CSV and extract poster info.
    Returns dict with poster, rating, year, genres, overview.
    """
    movies_df = _load_movies_if_needed()

    if movies_df is None or movies_df.empty:
        return None

    # Exact match (case-insensitive)
    mask = movies_df['title'].str.lower() == title.lower()
    if mask.any():
        row = movies_df[mask].iloc[0]

        # Build poster URL from poster_path if available
        poster_url = PLACEHOLDER_POSTER
        if pd.notna(row.get('poster_path')) and row['poster_path']:
            poster_path = row['poster_path']
            if isinstance(poster_path, str) and poster_path.startswith('/'):
                poster_url = f"{TMDB_IMAGE_BASE}{poster_path}"
            elif isinstance(poster_path, str):
                poster_url = f"{TMDB_IMAGE_BASE}/{poster_path}"

        # Extract year from release_date
        year = None
        if pd.notna(row.get('release_date')) and row['release_date']:
            try:
                year = str(row['release_date'])[:4]
            except:
                pass

        # Get rating
        rating = None
        if pd.notna(row.get('vote_average')):
            try:
                rating = round(float(row['vote_average']), 1)
            except:
                pass

        # Get genres
        genres = []
        if pd.notna(row.get('genres')) and row['genres']:
            try:
                if isinstance(row['genres'], list):
                    genres = row['genres']
                elif isinstance(row['genres'], str):
                    import ast
                    genres = ast.literal_eval(row['genres']) if row['genres'] else []
            except:
                pass

        # Get overview
        overview = ""
        if pd.notna(row.get('overview')):
            overview = str(row['overview'])

        return {
            "poster": poster_url,
            "rating": rating,
            "year": year,
            "genres": genres,
            "overview": overview,
            "tmdb_url": None,
        }

    return None


# ---------------------------------------------------------------------------
# Live TMDB lookups (with fallback to local CSV)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _genre_id_map(api_key):
    """TMDB's /search/movie only returns genre_ids, not names. Fetch the
    id->name table once per API key and cache it for the process lifetime."""
    try:
        resp = requests.get(
            f"{TMDB_API_BASE}/genre/movie/list",
            params={"api_key": api_key},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return {g["id"]: g["name"] for g in resp.json().get("genres", [])}
    except requests.RequestException as e:
        print(f"⚠️ Genre map fetch failed: {e}")
        return {}


@functools.lru_cache(maxsize=4000)
def _search_tmdb(title, api_key):
    """
    Cached TMDB search with intelligent fallback:
    1. Search by title
    2. Prefer results WITH posters
    3. Fallback to first result even if no poster (better than nothing)
    """
    try:
        print(f"📡 Searching TMDB for: '{title}'")
        print(f"📡 Using API key: {api_key[:10] if api_key else 'NONE'}...")

        resp = requests.get(
            f"{TMDB_API_BASE}/search/movie",
            params={"api_key": api_key, "query": title},
            timeout=REQUEST_TIMEOUT,
        )

        print(f"📡 Response status: {resp.status_code}")
        resp.raise_for_status()

        data = resp.json()
        results = data.get("results", [])
        print(f"📡 Found {len(results)} results")

        if results:
            print(f"   First result: {results[0].get('title')} (has poster: {bool(results[0].get('poster_path'))})")

        if not results:
            print(f"✗ No results found for '{title}'")
            return None

        # PREFERENCE 1: Movie with poster_path (most reliable)
        with_poster = [r for r in results if r.get("poster_path")]
        if with_poster:
            print(f"✓ Found result WITH poster: {with_poster[0].get('title')}")
            return with_poster[0]

        # PREFERENCE 2: First result even without poster (fallback)
        print(f"⚠️ No poster found, using first result anyway: {results[0].get('title')}")
        return results[0]

    except requests.RequestException as e:
        print(f"✗ API Error: {type(e).__name__}: {e}")
        return None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return None


def fetch_movie_details(title):
    """
    Live lookup for a single movie title: poster URL, rating, release year,
    genre names, overview text, and a TMDB page link.

    Strategy:
    1. Try local CSV first (no network needed)
    2. Fall back to TMDB API if available
    3. Return placeholder if all else fails

    Never raises — on any failure it returns the same placeholder shape
    so the UI always has something sane to render.
    """
    print(f"\n=== fetch_movie_details('{title}') ===")

    if not title:
        print(f"✗ No title provided")
        return _placeholder_details()

    # STEP 1: Try local CSV first (no network needed)
    print(f"📚 Checking local CSV data...")
    local_result = _get_local_poster(title)
    if local_result:
        print(
            f"✓ Found in local CSV: poster={'real' if 'placeholder' not in local_result['poster'] else 'placeholder'}")
        return local_result

    print(f"✗ Not found in local CSV")

    # STEP 2: Try TMDB API if API key available
    api_key = get_api_key()
    print(f"API key available: {bool(api_key)}")

    if api_key:
        result = _search_tmdb(title, api_key)
        if result:
            poster_path = result.get("poster_path")
            release_date = result.get("release_date") or ""
            genre_map = _genre_id_map(api_key)
            genre_names = [genre_map[g] for g in result.get("genre_ids", []) if g in genre_map]

            final_poster = f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else PLACEHOLDER_POSTER
            print(f"✓ Returning from TMDB: poster_path={poster_path is not None}")

            return {
                "poster": final_poster,
                "rating": round(result["vote_average"], 1) if result.get("vote_average") else None,
                "year": release_date[:4] if release_date else None,
                "genres": genre_names,
                "overview": result.get("overview") or "",
                "tmdb_url": f"https://www.themoviedb.org/movie/{result['id']}" if result.get("id") else None,
            }

    print(f"✗ Returning placeholder")
    return _placeholder_details()


# ---------------------------------------------------------------------------
# Title resolution (handles typos / case differences)
# ---------------------------------------------------------------------------

def _normalize(text):
    return str(text).strip().lower()


def _resolve_title(movies_indices, movie_name):
    """
    Exact case-insensitive match first. If that fails, fuzzy-match against
    every known title so a typo like "Avtaar" still resolves to "Avatar".
    Returns (resolved_title_or_None, correction_note_or_None).
    """
    if not movie_name:
        return None, None

    lower_to_original = {}
    for t in movies_indices.keys():
        lower_to_original.setdefault(_normalize(t), t)

    query = _normalize(movie_name)
    if query in lower_to_original:
        return lower_to_original[query], None

    close = difflib.get_close_matches(query, list(lower_to_original.keys()), n=1, cutoff=0.6)
    if close:
        matched = lower_to_original[close[0]]
        note = f'Couldn\'t find "{movie_name}" exactly — showing results for "{matched}" instead.'
        return matched, note

    return None, None


# ---------------------------------------------------------------------------
# Weighted rating (IMDB-style), used to rank genre-only requests where
# there's no anchor movie to compute similarity against.
# ---------------------------------------------------------------------------

def _weighted_score(row, m, c):
    v = row.get("vote_count", 0) or 0
    r = row.get("vote_average", 0) or 0
    if (v + m) == 0:
        return 0.0
    return (v / (v + m)) * r + (m / (v + m)) * c


# ---------------------------------------------------------------------------
# Main recommendation entry point
# ---------------------------------------------------------------------------

def recommend(movies, movies_indices, similarity, movie_name=None, genre=None, top_n=10):
    """
    Three modes, dispatched on which of movie_name/genre are given:

    - genre only      -> highest (weighted) rated movies carrying that genre
    - movie only       -> most similar movies by content (cosine similarity)
    - movie + genre    -> most similar movies, restricted to that genre

    Always returns:
        {
            "status": "ok" | "not_found",
            "message": str,                # human-readable, set on not_found
            "matched_title": str | None,   # resolved title, if movie_name given
            "correction_note": str | None, # set if a typo was auto-corrected
            "recommendations": list[str],  # movie titles, possibly empty
        }
    """
    result = {
        "status": "ok",
        "message": "",
        "matched_title": None,
        "correction_note": None,
        "recommendations": [],
    }

    if not movie_name and not genre:
        result["status"] = "not_found"
        result["message"] = "Pick a movie or a genre first."
        return result

    # ---- Genre-only: no anchor movie, rank by weighted rating ----
    if genre and not movie_name:
        mask = movies["genres"].apply(lambda g: genre in g if isinstance(g, list) else False)
        subset = movies[mask]

        if subset.empty:
            result["status"] = "not_found"
            result["message"] = f'No movies found in the "{genre}" genre.'
            return result

        c = pd.to_numeric(movies.get("vote_average", pd.Series(dtype=float)), errors="coerce").mean()
        m = pd.to_numeric(subset.get("vote_count", pd.Series(dtype=float)), errors="coerce").quantile(0.60)

        scored = subset.copy()
        scored["_score"] = scored.apply(lambda r: _weighted_score(r, m, c), axis=1)
        scored = scored.sort_values("_score", ascending=False)

        result["recommendations"] = scored["title"].head(top_n).tolist()
        return result

    # ---- Movie given (optionally narrowed by genre) ----
    resolved, note = _resolve_title(movies_indices, movie_name)
    result["correction_note"] = note

    if resolved is None:
        result["status"] = "not_found"
        result["message"] = f'"{movie_name}" isn\'t in the catalog.'
        return result

    result["matched_title"] = resolved
    idx = movies_indices[resolved]

    scores = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)

    picked = []
    for i, _score in scores:
        if i == idx:
            continue  # never recommend the movie back to itself

        if genre:
            row_genres = movies.iloc[i]["genres"]
            if not isinstance(row_genres, list) or genre not in row_genres:
                continue

        picked.append(movies.iloc[i]["title"])
        if len(picked) >= top_n:
            break

    result["recommendations"] = picked
    return result
