# -*- coding: utf-8 -*-
"""
data_loader.py
==============
Handles downloading, loading, and preprocessing of the MovieLens 100K dataset.

Beyond basic loading, this module also engineers features from raw data:
  - User demographic encoding (age bins, gender, occupation)
  - Movie temporal features (decade, age of movie at rating time)
  - Per-user and per-movie aggregate statistics
  - Merged feature matrices for model consumption

These side features help the recommender handle cold-start scenarios
and improve prediction accuracy by giving models richer input signals.
"""

import sys
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from colorama import Fore, Style, init

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

init(autoreset=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
ML100K_URL   = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML100K_DIR   = RAW_DIR / "ml-100k"


# ── Genre list (MovieLens 100K bitmask column order) ──────────────────────────
GENRES = [
    "unknown", "Action", "Adventure", "Animation",
    "Children's", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
]

# Occupations present in the dataset - we need a fixed ordering for encoding
OCCUPATIONS = [
    "administrator", "artist", "doctor", "educator", "engineer",
    "entertainment", "executive", "healthcare", "homemaker", "lawyer",
    "librarian", "marketing", "none", "other", "programmer",
    "retired", "salesman", "scientist", "student", "technician",
    "writer",
]


def _download_movielens(verbose: bool = True) -> None:
    """Download & extract MovieLens 100K if not already present."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "ml-100k.zip"

    if ML100K_DIR.exists():
        if verbose:
            print(f"{Fore.GREEN}[DataLoader] MovieLens 100K already downloaded → {ML100K_DIR}")
        return

    if verbose:
        print(f"{Fore.CYAN}[DataLoader] Downloading MovieLens 100K from GroupLens …")

    resp  = requests.get(ML100K_URL, stream=True, timeout=60)
    total = int(resp.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                          desc="ml-100k.zip") as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(RAW_DIR)

    zip_path.unlink()
    print(f"{Fore.GREEN}[DataLoader] Extracted → {ML100K_DIR}")


def load_ratings(verbose: bool = True) -> pd.DataFrame:
    """
    Load user–movie ratings.

    Returns
    -------
    pd.DataFrame  columns: ['user_id', 'movie_id', 'rating', 'timestamp']
    """
    _download_movielens(verbose)
    path = ML100K_DIR / "u.data"
    df   = pd.read_csv(path, sep="\t",
                       names=["user_id", "movie_id", "rating", "timestamp"])
    if verbose:
        print(f"{Fore.CYAN}[DataLoader] Ratings loaded  → {len(df):,} rows | "
              f"{df['user_id'].nunique()} users | {df['movie_id'].nunique()} movies")
    return df


def load_movies(verbose: bool = True) -> pd.DataFrame:
    """
    Load movie metadata.

    Returns
    -------
    pd.DataFrame  columns: ['movie_id', 'title', 'release_date', 'genres', 'genre_list', 'year']
    """
    _download_movielens(verbose=False)
    path = ML100K_DIR / "u.item"

    cols = ["movie_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRES
    df   = pd.read_csv(path, sep="|", names=cols, encoding="latin-1", low_memory=False)

    def _genre_list(row):
        return [g for g in GENRES if row.get(g, 0) == 1]

    df["genre_list"] = df.apply(_genre_list, axis=1)
    df["genres"]     = df["genre_list"].apply(lambda lst: "|".join(lst) if lst else "unknown")
    df["year"]       = df["title"].str.extract(r"\((\d{4})\)").astype("float")

    df = df[["movie_id", "title", "release_date", "genres", "genre_list", "year"]].copy()

    if verbose:
        print(f"{Fore.CYAN}[DataLoader] Movies loaded   → {len(df):,} movies")
    return df


def load_users(verbose: bool = True) -> pd.DataFrame:
    """
    Load user demographic info.

    Returns
    -------
    pd.DataFrame  columns: ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    """
    _download_movielens(verbose=False)
    path = ML100K_DIR / "u.user"
    df   = pd.read_csv(path, sep="|",
                       names=["user_id", "age", "gender", "occupation", "zip_code"])
    if verbose:
        print(f"{Fore.CYAN}[DataLoader] Users loaded    → {len(df):,} users")
    return df


# ── Feature Engineering ────────────────────────────────────────────────────────

def encode_user_features(users_df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode user demographics into numeric features.

    Produces:
      - age_norm       : normalised age (0-1)
      - age_bucket     : categorical (0-4) for <18, 18-25, 25-35, 35-50, 50+
      - gender_enc     : binary (0=F, 1=M)
      - occupation_enc : integer label (0 to n_occupations-1)

    We use simple encodings here rather than one-hot to keep the feature
    vector compact. For 21 occupations, one-hot would add 21 columns which
    is overkill for our use case — the integer encoding works fine with
    tree-based models and embedding layers.
    """
    df = users_df.copy()

    # Age: normalise to [0, 1] range
    df["age_norm"] = (df["age"] - df["age"].min()) / (df["age"].max() - df["age"].min() + 1e-8)

    # Age buckets — matches common demographic groupings
    bins   = [0, 18, 25, 35, 50, 100]
    labels = [0, 1, 2, 3, 4]
    df["age_bucket"] = pd.cut(df["age"], bins=bins, labels=labels, right=False).astype(int)

    # Gender: binary encoding
    df["gender_enc"] = (df["gender"] == "M").astype(int)

    # Occupation: label encoding using our fixed ordering
    occ_map = {occ: i for i, occ in enumerate(OCCUPATIONS)}
    df["occupation_enc"] = df["occupation"].str.lower().map(occ_map).fillna(len(OCCUPATIONS)).astype(int)

    return df


def compute_user_stats(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-user aggregate statistics from their rating history.

    These features capture individual rating behaviour patterns:
      - n_ratings : total number of movies rated (activity level)
      - avg_rating: mean rating given (generous vs. critical rater)
      - std_rating: standard deviation of ratings (consistency)
      - genre_breadth: not computed here but could be added later

    The intuition is that a user who rates 500 movies has very different
    characteristics from someone who rated 20, even if their average is similar.
    """
    stats = (
        ratings_df.groupby("user_id")["rating"]
        .agg(n_ratings="count", avg_rating="mean", std_rating="std")
        .reset_index()
    )
    stats["std_rating"] = stats["std_rating"].fillna(0)

    # Normalise count to [0, 1] — log scale since distribution is very skewed
    stats["activity_score"] = np.log1p(stats["n_ratings"]) / np.log1p(stats["n_ratings"].max())

    return stats


def compute_movie_stats(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-movie aggregate statistics.

    These act as "popularity features" that help the model understand
    how well-known or controversial a movie is. A movie with 500 ratings
    and std=0.3 is very different from one with 5 ratings and std=1.5.
    """
    stats = (
        ratings_df.groupby("movie_id")["rating"]
        .agg(n_ratings="count", avg_rating="mean", std_rating="std")
        .reset_index()
    )
    stats["std_rating"] = stats["std_rating"].fillna(0)

    # Popularity score — log-normalised rating count
    stats["popularity"] = np.log1p(stats["n_ratings"]) / np.log1p(stats["n_ratings"].max())

    return stats


def add_temporal_features(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from rating timestamps.

    Temporal signals can capture patterns like:
      - Users tend to rate more generously at certain times
      - Older movies may get different ratings than recent releases
      - The gap between movie release and rating time matters
    """
    df = ratings_df.copy()

    # Convert unix timestamp to datetime components
    dt = pd.to_datetime(df["timestamp"], unit="s")
    df["hour"]       = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek    # 0=Monday, 6=Sunday
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    # Movie age at time of rating (in years)
    # This captures whether the user is watching a recent release or a classic
    movie_years = movies_df.set_index("movie_id")["year"].to_dict()
    df["movie_year"]  = df["movie_id"].map(movie_years)
    df["rating_year"] = dt.dt.year
    df["movie_age"]   = (df["rating_year"] - df["movie_year"]).clip(lower=0)
    df["movie_age"]   = df["movie_age"].fillna(0)

    return df


def build_feature_matrices(ratings_df: pd.DataFrame, movies_df: pd.DataFrame,
                           users_df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Build enriched feature matrices that can be used by advanced models.

    Returns a dict with:
      - user_features : DataFrame indexed by user_id
      - movie_features: DataFrame indexed by movie_id
      - user_stats    : per-user rating statistics
      - movie_stats   : per-movie rating statistics
    """
    if verbose:
        print(f"{Fore.CYAN}[DataLoader] Engineering features …")

    user_feats = encode_user_features(users_df)
    user_stats = compute_user_stats(ratings_df)
    movie_stats = compute_movie_stats(ratings_df)

    # merge user demographics with their rating stats
    user_features = user_feats.merge(user_stats, on="user_id", how="left")

    # Movie features: combine stats with genre info
    movie_features = movies_df[["movie_id", "title", "genres", "year"]].copy()
    movie_features = movie_features.merge(movie_stats, on="movie_id", how="left")
    movie_features["decade"] = (movie_features["year"] // 10 * 10).fillna(0).astype(int)

    if verbose:
        print(f"{Fore.GREEN}[DataLoader] Features ready  → "
              f"user_features: {user_features.shape}, movie_features: {movie_features.shape}")

    return {
        "user_features":  user_features,
        "movie_features": movie_features,
        "user_stats":     user_stats,
        "movie_stats":    movie_stats,
    }


# ── Main Loader ────────────────────────────────────────────────────────────────

def load_all(verbose: bool = True) -> dict:
    """
    Convenience function — load ratings, movies, and users then merge.

    Returns
    -------
    dict with keys: 'ratings', 'movies', 'users', 'merged',
                    'user_features', 'movie_features', 'user_stats', 'movie_stats'
    """
    ratings = load_ratings(verbose)
    movies  = load_movies(verbose)
    users   = load_users(verbose)

    merged = (ratings
              .merge(movies, on="movie_id", how="left")
              .merge(users,  on="user_id",  how="left"))

    # Build enriched feature matrices
    features = build_feature_matrices(ratings, movies, users, verbose)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(PROC_DIR / "ratings.csv", index=False)
    movies.to_csv(PROC_DIR  / "movies.csv",  index=False)
    users.to_csv(PROC_DIR   / "users.csv",   index=False)
    merged.to_csv(PROC_DIR  / "merged.csv",  index=False)

    if verbose:
        print(f"{Fore.GREEN}[DataLoader] Processed data saved → {PROC_DIR}")

    return {
        "ratings": ratings, "movies": movies, "users": users, "merged": merged,
        **features,
    }


if __name__ == "__main__":
    data = load_all(verbose=True)
    print(data["merged"].head())
    print(f"\nUser features sample:\n{data['user_features'].head()}")
    print(f"\nMovie features sample:\n{data['movie_features'].head()}")
