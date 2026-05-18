"""
data_loader.py
==============
Handles downloading, loading, and preprocessing of the MovieLens 100K dataset.

Key Steps:
  1. Download the dataset directly from GroupLens (no external ML library needed)
  2. Load ratings into a pandas DataFrame
  3. Load movie metadata (titles + genres)
  4. Merge and clean the data
  5. Expose helper functions for the rest of the pipeline
"""

import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from colorama import Fore, Style, init

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
    pd.DataFrame  columns: ['movie_id', 'title', 'release_date', 'genres', 'genre_list']
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


def load_all(verbose: bool = True) -> dict:
    """
    Convenience function — load ratings, movies, and users then merge.

    Returns
    -------
    dict with keys: 'ratings', 'movies', 'users', 'merged'
    """
    ratings = load_ratings(verbose)
    movies  = load_movies(verbose)
    users   = load_users(verbose)

    merged = (ratings
              .merge(movies, on="movie_id", how="left")
              .merge(users,  on="user_id",  how="left"))

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(PROC_DIR / "ratings.csv", index=False)
    movies.to_csv(PROC_DIR  / "movies.csv",  index=False)
    users.to_csv(PROC_DIR   / "users.csv",   index=False)
    merged.to_csv(PROC_DIR  / "merged.csv",  index=False)

    if verbose:
        print(f"{Fore.GREEN}[DataLoader] Processed data saved → {PROC_DIR}")

    return {"ratings": ratings, "movies": movies, "users": users, "merged": merged}


if __name__ == "__main__":
    data = load_all(verbose=True)
    print(data["merged"].head())
