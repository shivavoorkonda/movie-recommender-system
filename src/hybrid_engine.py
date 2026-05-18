"""
hybrid_engine.py
================
Combines Collaborative Filtering (SVD) + Content-Based Filtering
into a single Hybrid Recommender Engine.

Hybrid Strategy — Weighted Score Fusion:
  hybrid_score(movie) = α * CF_score + (1 - α) * CB_score

  α (alpha): blending weight
    - α = 1.0  → pure Collaborative Filtering
    - α = 0.0  → pure Content-Based
    - α = 0.7  → default: CF-dominant (CF tends to be more accurate with sufficient ratings)

Cold-Start Handling:
  If user has < cold_start_threshold ratings, we lower α automatically
  so the system relies more on content signals (genres, etc.).

This mirrors how Netflix/Spotify blend matrix-factorisation + item features.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from colorama import Fore, Style, init

from collaborative_filtering import CollaborativeFilteringModel
from content_based import ContentBasedModel

init(autoreset=True)


class HybridRecommender:
    """
    Hybrid Recommender that blends CF (SVD) and Content-Based scores.

    Parameters
    ----------
    alpha                : float  — weight for CF score (0–1). Default 0.7
    cold_start_threshold : int    — if user has fewer ratings, reduce alpha
    """

    def __init__(self, alpha: float = 0.7, cold_start_threshold: int = 10):
        self.alpha                = alpha
        self.cold_start_threshold = cold_start_threshold
        self.cf_model             = None
        self.cb_model             = None
        self._fitted              = False

    # ── Setup ─────────────────────────────────────────────────────────────────

    def set_models(self, cf_model: CollaborativeFilteringModel,
                   cb_model: ContentBasedModel) -> "HybridRecommender":
        """Attach already-trained CF and CB models."""
        if not cf_model._fitted:
            raise ValueError("CF model is not fitted. Call .fit() on it first.")
        if not cb_model._fitted:
            raise ValueError("CB model is not fitted. Call .fit() on it first.")
        self.cf_model = cf_model
        self.cb_model = cb_model
        self._fitted  = True
        print(f"{Fore.GREEN}[Hybrid] Models attached. α = {self.alpha}")
        return self

    # ── Effective Alpha (cold-start adaptation) ────────────────────────────────

    def _effective_alpha(self, user_id: int, ratings_df: pd.DataFrame) -> float:
        """
        Dynamically adjust α based on how many ratings the user has.

        Users with sparse history → lower α → trust content-based more.
        """
        n_rated = len(ratings_df[ratings_df["user_id"] == user_id])
        if n_rated < self.cold_start_threshold:
            # Scale α down proportionally
            effective = self.alpha * (n_rated / self.cold_start_threshold)
            print(f"{Fore.YELLOW}[Hybrid] Cold-start detected for user {user_id} "
                  f"({n_rated} ratings). Effective α = {effective:.2f}")
            return effective
        return self.alpha

    # ── Score Normalisation ───────────────────────────────────────────────────

    @staticmethod
    def _min_max_normalise(series: pd.Series) -> pd.Series:
        """Normalise a score column to [0, 1]."""
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - mn) / (mx - mn)

    # ── Main Recommendation ───────────────────────────────────────────────────

    def recommend(self, user_id: int, movies_df: pd.DataFrame,
                  ratings_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """
        Generate top-N hybrid recommendations for a user.

        Steps
        -----
        1. Get CF predicted ratings for all unseen movies.
        2. Get CB similarity scores for the user's liked-movie profile.
        3. Normalise both score columns to [0, 1].
        4. Compute hybrid_score = α * cf_norm + (1-α) * cb_norm.
        5. Sort and return top-N.

        Returns
        -------
        pd.DataFrame  columns: ['movie_id', 'title', 'genres',
                                 'cf_score', 'cb_score',
                                 'cf_norm', 'cb_norm', 'hybrid_score']
        """
        if not self._fitted:
            raise RuntimeError("Engine not ready. Call set_models() first.")

        alpha = self._effective_alpha(user_id, ratings_df)
        rated_ids = set(ratings_df[ratings_df["user_id"] == user_id]["movie_id"].tolist())

        # ── 1. CF scores ──────────────────────────────────────────────────────
        print(f"{Fore.CYAN}[Hybrid] Computing CF scores ...")
        cf_preds = []
        for _, row in movies_df.iterrows():
            mid = int(row["movie_id"])
            if mid not in rated_ids:
                u_idx = self.cf_model.user_map.get(user_id)
                i_idx = self.cf_model.item_map.get(mid)
                if u_idx is not None and i_idx is not None:
                    est = self.cf_model._predict_single(u_idx, i_idx)
                else:
                    est = float(self.cf_model.mu) if self.cf_model.mu else 3.5
                cf_preds.append({"movie_id": mid, "cf_score": est})
        cf_df = pd.DataFrame(cf_preds)

        # ── 2. CB scores ──────────────────────────────────────────────────────
        print(f"{Fore.CYAN}[Hybrid] Computing CB scores …")
        try:
            cb_df = self.cb_model.get_user_profile_recommendations(
                user_id=user_id, ratings_df=ratings_df,
                n=len(movies_df), min_rating=3.5
            ).rename(columns={"similarity_score": "cb_score"})[["movie_id", "cb_score"]]
        except ValueError:
            print(f"{Fore.YELLOW}[Hybrid] CB profile failed for user {user_id}. Using zeros.")
            cb_df = pd.DataFrame({"movie_id": cf_df["movie_id"], "cb_score": 0.0})

        # ── 3. Merge & Normalise ──────────────────────────────────────────────
        merged = cf_df.merge(cb_df, on="movie_id", how="left").fillna(0)
        merged["cf_norm"] = self._min_max_normalise(merged["cf_score"])
        merged["cb_norm"] = self._min_max_normalise(merged["cb_score"])

        # ── 4. Hybrid Score ───────────────────────────────────────────────────
        merged["hybrid_score"] = (alpha * merged["cf_norm"] +
                                  (1 - alpha) * merged["cb_norm"])

        # ── 5. Enrich with metadata ───────────────────────────────────────────
        result = (merged
                  .merge(movies_df[["movie_id", "title", "genres"]], on="movie_id")
                  .sort_values("hybrid_score", ascending=False)
                  .head(n)
                  .reset_index(drop=True))

        # Round for readability
        for col in ["cf_score", "cb_score", "cf_norm", "cb_norm", "hybrid_score"]:
            result[col] = result[col].round(4)

        print(f"{Fore.GREEN}[Hybrid] Top-{n} recommendations generated for user {user_id} ✓")
        return result

    # ── Popularity Fallback ───────────────────────────────────────────────────

    @staticmethod
    def popularity_fallback(movies_df: pd.DataFrame, ratings_df: pd.DataFrame,
                             n: int = 10) -> pd.DataFrame:
        """
        Popularity-based recommendations for completely new users (no history at all).

        Returns the top-N movies by average rating (weighted by number of ratings).
        """
        stats = (ratings_df
                 .groupby("movie_id")
                 .agg(avg_rating=("rating", "mean"), n_ratings=("rating", "count"))
                 .reset_index())
        # Bayesian average: prior = global mean, weight by rating count
        global_mean = ratings_df["rating"].mean()
        C           = stats["n_ratings"].quantile(0.60)  # min votes threshold
        stats["weighted_score"] = (
            (stats["n_ratings"] / (stats["n_ratings"] + C)) * stats["avg_rating"] +
            (C / (stats["n_ratings"] + C)) * global_mean
        )
        stats = stats.merge(movies_df[["movie_id", "title", "genres"]], on="movie_id")
        top   = stats.sort_values("weighted_score", ascending=False).head(n)
        return top[["movie_id", "title", "genres", "avg_rating", "n_ratings", "weighted_score"]].reset_index(drop=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

    from data_loader import load_all, build_surprise_dataset

    data    = load_all(verbose=True)
    ratings = data["ratings"]
    movies  = data["movies"]
    dataset = build_surprise_dataset(ratings)

    cf = CollaborativeFilteringModel(n_factors=100, n_epochs=20)
    cf.fit(dataset)

    cb = ContentBasedModel()
    cb.fit(movies)

    engine = HybridRecommender(alpha=0.7)
    engine.set_models(cf, cb)

    print("\n── Hybrid Recommendations for User 1 ──")
    recs = engine.recommend(user_id=1, movies_df=movies, ratings_df=ratings, n=10)
    print(recs[["title", "genres", "cf_score", "cb_score", "hybrid_score"]].to_string())

    print("\n── Popularity Fallback (new users) ──")
    pop = HybridRecommender.popularity_fallback(movies, ratings, n=10)
    print(pop[["title", "avg_rating", "n_ratings", "weighted_score"]].to_string())
