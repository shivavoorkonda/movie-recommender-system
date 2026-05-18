"""
utils.py
========
General-purpose helper functions shared across the project.
"""

import sys
import time
import textwrap
from typing import Optional
import pandas as pd
import numpy as np
from colorama import Fore, Style, init
from tabulate import tabulate

init(autoreset=True)


# ── Pretty Printing ────────────────────────────────────────────────────────────

def print_header(title: str, width: int = 62) -> None:
    """Print a styled section header."""
    bar = "=" * width
    print(f"\n{Fore.CYAN}{bar}")
    print(f"{Fore.CYAN}  {title}")
    print(f"{Fore.CYAN}{bar}")


def print_recommendations(df: pd.DataFrame, user_id: int,
                           mode: str = "Hybrid", n: int = 10) -> None:
    """
    Pretty-print a recommendations DataFrame as a rich table.

    Parameters
    ----------
    df      : recommendations DataFrame
    user_id : int
    mode    : str  — 'Hybrid', 'CF', or 'Content-Based'
    n       : int  — number to display
    """
    print_header(f"Top-{n} {mode} Recommendations  ·  User #{user_id}")

    display_df = df.head(n).copy()

    # Truncate long titles
    display_df["title"] = display_df["title"].apply(
        lambda t: (t[:38] + "…") if len(str(t)) > 40 else t
    )

    # Truncate genres
    display_df["genres"] = display_df["genres"].apply(
        lambda g: (g[:28] + "…") if len(str(g)) > 30 else g
    )

    # Choose columns based on mode
    if "hybrid_score" in display_df.columns:
        cols = ["title", "genres", "cf_score", "cb_score", "hybrid_score"]
    elif "predicted_rating" in display_df.columns:
        cols = ["title", "genres", "predicted_rating"]
    elif "similarity_score" in display_df.columns:
        cols = ["title", "genres", "similarity_score"]
    else:
        cols = list(display_df.columns)

    # Add rank column
    display_df.insert(0, "#", range(1, len(display_df) + 1))
    cols = ["#"] + [c for c in cols if c in display_df.columns]

    print(tabulate(display_df[cols], headers="keys",
                   tablefmt="rounded_outline", showindex=False,
                   floatfmt=".4f"))
    print()


def print_movie_details(movie_id: int, movies_df: pd.DataFrame,
                        ratings_df: pd.DataFrame) -> None:
    """Print details for a specific movie."""
    row = movies_df[movies_df["movie_id"] == movie_id]
    if row.empty:
        print(f"{Fore.RED}Movie {movie_id} not found.")
        return
    row = row.iloc[0]

    movie_ratings = ratings_df[ratings_df["movie_id"] == movie_id]["rating"]
    avg_r  = movie_ratings.mean() if len(movie_ratings) > 0 else "N/A"
    n_r    = len(movie_ratings)

    print_header(f"Movie Details — {row['title']}")
    print(f"  {'Movie ID':<16}: {movie_id}")
    print(f"  {'Genres':<16}: {row['genres']}")
    print(f"  {'Avg Rating':<16}: {avg_r:.2f} ★ ({n_r} ratings)")
    print()


def format_elapsed(start: float) -> str:
    """Return formatted elapsed time string."""
    elapsed = time.time() - start
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    return f"{elapsed // 60:.0f}m {elapsed % 60:.0f}s"


# ── Data Helpers ───────────────────────────────────────────────────────────────

def movie_id_from_title(title_substr: str, movies_df: pd.DataFrame) -> Optional[int]:
    """
    Return the movie_id of the first movie whose title contains title_substr.
    Case-insensitive. Returns None if not found.
    """
    mask = movies_df["title"].str.contains(title_substr, case=False, na=False)
    hits = movies_df[mask]
    if hits.empty:
        print(f"{Fore.YELLOW}No movie found matching '{title_substr}'.")
        return None
    if len(hits) > 1:
        print(f"{Fore.YELLOW}Multiple matches for '{title_substr}':")
        for _, r in hits.iterrows():
            print(f"  [{r['movie_id']}] {r['title']}")
    return int(hits.iloc[0]["movie_id"])


def get_user_history(user_id: int, ratings_df: pd.DataFrame,
                     movies_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return a user's rating history, sorted by rating descending."""
    user_r = ratings_df[ratings_df["user_id"] == user_id].copy()
    user_r = user_r.merge(movies_df[["movie_id", "title", "genres"]], on="movie_id")
    user_r = user_r.sort_values("rating", ascending=False).head(top_n)
    return user_r[["movie_id", "title", "genres", "rating"]].reset_index(drop=True)


def describe_dataset(ratings_df: pd.DataFrame, movies_df: pd.DataFrame,
                     users_df: pd.DataFrame) -> None:
    """Print a summary of the dataset statistics."""
    print_header("Dataset Statistics — MovieLens 100K")
    stats = [
        ["Total Ratings",   f"{len(ratings_df):,}"],
        ["Unique Users",    f"{ratings_df['user_id'].nunique():,}"],
        ["Unique Movies",   f"{ratings_df['movie_id'].nunique():,}"],
        ["Rating Range",    f"{ratings_df['rating'].min()} – {ratings_df['rating'].max()}"],
        ["Avg Rating",      f"{ratings_df['rating'].mean():.3f}"],
        ["Sparsity",        f"{1 - len(ratings_df) / (ratings_df['user_id'].nunique() * ratings_df['movie_id'].nunique()):.4%}"],
        ["Total Movies",    f"{len(movies_df):,}"],
        ["Total Users",     f"{len(users_df):,}"],
    ]
    print(tabulate(stats, headers=["Metric", "Value"],
                   tablefmt="rounded_outline", showindex=False))
    print()
