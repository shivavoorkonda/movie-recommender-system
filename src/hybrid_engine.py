# -*- coding: utf-8 -*-
"""
hybrid_engine.py
================
Combines Collaborative Filtering (SGD + ALS) and Content-Based Filtering
into a Hybrid Recommender Engine using two strategies:

1. Weighted Score Fusion (simple, fast):
     hybrid_score = α * CF_score + (1 - α) * CB_score

2. Stacked Ensemble (learned, more accurate):
     A Ridge regression meta-learner trained on a held-out validation set
     learns optimal weights from individual model scores + side features.

Additional features:
  - Cold-start handling (reduce CF weight for sparse users)
  - Maximal Marginal Relevance (MMR) re-ranking for diversity
  - Popularity debiasing to improve novelty

This mirrors how Netflix/Spotify blend matrix-factorisation + item features,
but with the added benefit of a learned combination model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from colorama import Fore, Style, init
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

from collaborative_filtering import CollaborativeFilteringModel, ALSModel
from content_based import ContentBasedModel

init(autoreset=True)


class HybridRecommender:
    """
    Hybrid Recommender that blends CF (SGD + ALS) and Content-Based scores.

    Parameters
    ----------
    alpha                : float  — weight for CF score in simple fusion (0–1)
    cold_start_threshold : int    — if user has fewer ratings, reduce alpha
    use_stacking         : bool   — whether to use the learned meta-learner
    mmr_lambda           : float  — diversity parameter for MMR re-ranking (0=pure diversity, 1=pure relevance)
    """

    def __init__(self, alpha: float = 0.7, cold_start_threshold: int = 10,
                 use_stacking: bool = False, mmr_lambda: float = 0.7):
        self.alpha                = alpha
        self.cold_start_threshold = cold_start_threshold
        self.use_stacking         = use_stacking
        self.mmr_lambda           = mmr_lambda

        self.cf_model   = None   # SGD model
        self.als_model  = None   # ALS model (optional)
        self.cb_model   = None
        self._fitted    = False

        # Stacking meta-learner
        self._meta_model  = None
        self._meta_scaler = None
        self._stacking_trained = False

    # ── Setup ─────────────────────────────────────────────────────────────────

    def set_models(self, cf_model: CollaborativeFilteringModel,
                   cb_model: ContentBasedModel,
                   als_model: ALSModel = None) -> "HybridRecommender":
        """Attach already-trained CF, CB, and optionally ALS models."""
        if not cf_model._fitted:
            raise ValueError("CF model is not fitted. Call .fit() on it first.")
        if not cb_model._fitted:
            raise ValueError("CB model is not fitted. Call .fit() on it first.")
        self.cf_model  = cf_model
        self.cb_model  = cb_model
        self.als_model = als_model
        self._fitted   = True
        models = "SGD + CB"
        if als_model is not None:
            models += " + ALS"
        print(f"{Fore.GREEN}[Hybrid] Models attached ({models}). α = {self.alpha}")
        return self

    # ── Stacking Meta-Learner ─────────────────────────────────────────────────

    def train_stacking(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame,
                        val_size: float = 0.15) -> None:
        """
        Train the stacking meta-learner on a held-out validation set.

        The idea: instead of using a fixed alpha, we learn the optimal
        combination weights from data. The meta-learner gets features like:
          - CF-SGD predicted rating
          - CF-ALS predicted rating (if available)
          - CB similarity score
          - Number of user ratings (activity level)
          - Movie popularity (number of ratings)

        And it learns to predict the actual rating from these inputs.
        This is essentially what the Netflix Prize winners did.
        """
        if not self._fitted:
            raise RuntimeError("Models not attached. Call set_models() first.")

        from sklearn.model_selection import train_test_split
        print(f"{Fore.CYAN}[Hybrid] Training stacking meta-learner …")

        # Split off a validation set
        _, val_df = train_test_split(ratings_df, test_size=val_size, random_state=42)
        val_df = val_df.reset_index(drop=True)

        # Compute per-user and per-movie stats for features
        user_counts = ratings_df.groupby("user_id")["rating"].count().to_dict()
        movie_counts = ratings_df.groupby("movie_id")["rating"].count().to_dict()

        features, targets = [], []
        for _, row in val_df.iterrows():
            uid = int(row["user_id"])
            mid = int(row["movie_id"])
            r   = float(row["rating"])

            # CF-SGD score
            cf_score = self.cf_model.predict_rating(uid, mid)

            # CF-ALS score (if available)
            als_score = self.als_model.predict_rating(uid, mid) if self.als_model else cf_score

            # CB score — use the raw TF-IDF similarity to the user's profile
            # This is a simplified version; in production you'd cache user profiles
            cb_score = 3.5  # default
            if uid in [int(x) for x in ratings_df["user_id"].unique()[:200]]:
                # Only compute CB for a subset to keep training fast
                try:
                    cb_recs = self.cb_model.get_user_profile_recommendations(
                        uid, ratings_df, n=1000, min_rating=3.5, popularity_penalty=0.0)
                    match = cb_recs[cb_recs["movie_id"] == mid]
                    if not match.empty:
                        cb_score = float(match.iloc[0]["similarity_score"]) * 5
                except (ValueError, IndexError):
                    pass

            feat = [
                cf_score,
                als_score,
                cb_score,
                np.log1p(user_counts.get(uid, 0)),    # user activity
                np.log1p(movie_counts.get(mid, 0)),    # movie popularity
            ]
            features.append(feat)
            targets.append(r)

        X = np.array(features)
        y = np.array(targets)

        # Standardise features
        self._meta_scaler = StandardScaler()
        X_scaled = self._meta_scaler.fit_transform(X)

        # Train a Ridge regression — simple, fast, and regularised
        # We don't want anything too complex here since we have limited val data
        self._meta_model = Ridge(alpha=1.0)
        self._meta_model.fit(X_scaled, y)
        self._stacking_trained = True

        # Report learned weights
        coefs = self._meta_model.coef_
        names = ["CF-SGD", "CF-ALS", "CB", "UserActivity", "MoviePopularity"]
        print(f"{Fore.GREEN}[Hybrid] Stacking trained ✓  "
              f"Weights: {', '.join(f'{n}={c:.3f}' for n, c in zip(names, coefs))}")

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

    # ── MMR Re-ranking ────────────────────────────────────────────────────────

    def _mmr_rerank(self, candidates: pd.DataFrame, n: int,
                     score_col: str = "hybrid_score") -> pd.DataFrame:
        """
        Maximal Marginal Relevance re-ranking for diversity.

        MMR balances relevance with diversity by iteratively selecting items
        that are both highly relevant AND dissimilar to already-selected items.

        MMR(i) = λ * relevance(i) - (1-λ) * max_similarity(i, selected)

        A λ of 1.0 = pure relevance (no diversity), 0.0 = max diversity.
        We use the CB similarity matrix to measure item-item similarity.
        """
        if self.mmr_lambda >= 0.99 or len(candidates) <= n:
            return candidates.head(n)

        # Get indices into the CB similarity matrix
        valid_ids = [mid for mid in candidates["movie_id"] if mid in self.cb_model.movie_index]
        if len(valid_ids) < n:
            return candidates.head(n)

        scores = candidates.set_index("movie_id")[score_col].to_dict()
        selected = []
        remaining = list(candidates["movie_id"])

        # Greedy MMR selection
        for _ in range(min(n, len(remaining))):
            best_id, best_mmr = None, -np.inf

            for mid in remaining:
                relevance = scores.get(mid, 0)

                # Max similarity to any already-selected item
                if selected and mid in self.cb_model.movie_index:
                    max_sim = 0.0
                    idx_mid = self.cb_model.movie_index[mid]
                    for sel_mid in selected:
                        if sel_mid in self.cb_model.movie_index:
                            idx_sel = self.cb_model.movie_index[sel_mid]
                            sim = self.cb_model.sim_matrix[idx_mid, idx_sel]
                            max_sim = max(max_sim, sim)
                else:
                    max_sim = 0.0

                mmr = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_sim

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_id  = mid

            if best_id is not None:
                selected.append(best_id)
                remaining.remove(best_id)

        # Rebuild DataFrame in MMR order
        result = candidates[candidates["movie_id"].isin(selected)]
        result = result.set_index("movie_id").loc[selected].reset_index()
        return result

    # ── Main Recommendation ───────────────────────────────────────────────────

    def recommend(self, user_id: int, movies_df: pd.DataFrame,
                  ratings_df: pd.DataFrame, n: int = 10,
                  diversity: bool = True) -> pd.DataFrame:
        """
        Generate top-N hybrid recommendations for a user.

        Steps
        -----
        1. Get CF predicted ratings for all unseen movies (SGD + optionally ALS).
        2. Get CB similarity scores for the user's liked-movie profile.
        3. Normalise both score columns to [0, 1].
        4. Compute hybrid_score = α * cf_norm + (1-α) * cb_norm.
        5. Optionally apply MMR re-ranking for diversity.
        6. Sort and return top-N.
        """
        if not self._fitted:
            raise RuntimeError("Engine not ready. Call set_models() first.")

        alpha = self._effective_alpha(user_id, ratings_df)
        rated_ids = set(ratings_df[ratings_df["user_id"] == user_id]["movie_id"].tolist())

        # ── 1. CF-SGD scores ──────────────────────────────────────────────────
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

                # Also get ALS score if available
                als_est = est  # default to SGD
                if self.als_model is not None:
                    au = self.als_model.user_map.get(user_id)
                    ai = self.als_model.item_map.get(mid)
                    if au is not None and ai is not None:
                        als_est = self.als_model._predict_single(au, ai)

                cf_preds.append({"movie_id": mid, "cf_score": est, "als_score": als_est})
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
        merged["cf_norm"]  = self._min_max_normalise(merged["cf_score"])
        merged["als_norm"] = self._min_max_normalise(merged["als_score"])
        merged["cb_norm"]  = self._min_max_normalise(merged["cb_score"])

        # ── 4. Hybrid Score ───────────────────────────────────────────────────
        # Blend SGD and ALS scores (equal weight), then combine with CB
        merged["cf_combined"] = 0.6 * merged["cf_norm"] + 0.4 * merged["als_norm"]
        merged["hybrid_score"] = (alpha * merged["cf_combined"] +
                                  (1 - alpha) * merged["cb_norm"])

        # ── 5. Enrich with metadata ───────────────────────────────────────────
        result = (merged
                  .merge(movies_df[["movie_id", "title", "genres"]], on="movie_id")
                  .sort_values("hybrid_score", ascending=False))

        # ── 6. MMR re-ranking for diversity ───────────────────────────────────
        # Get more candidates than needed, then re-rank
        candidates = result.head(n * 3).reset_index(drop=True)
        if diversity and self.mmr_lambda < 0.99:
            result = self._mmr_rerank(candidates, n)
        else:
            result = candidates.head(n)

        result = result.reset_index(drop=True)

        # Round for readability
        for col in ["cf_score", "als_score", "cb_score", "cf_norm", "als_norm",
                     "cb_norm", "cf_combined", "hybrid_score"]:
            if col in result.columns:
                result[col] = result[col].round(4)

        print(f"{Fore.GREEN}[Hybrid] Top-{n} recommendations generated for user {user_id} ✓")
        return result

    # ── Popularity Fallback ───────────────────────────────────────────────────

    @staticmethod
    def popularity_fallback(movies_df: pd.DataFrame, ratings_df: pd.DataFrame,
                             n: int = 10) -> pd.DataFrame:
        """
        Popularity-based recommendations for completely new users.
        Returns top-N by Bayesian-weighted average rating.
        """
        stats = (ratings_df
                 .groupby("movie_id")
                 .agg(avg_rating=("rating", "mean"), n_ratings=("rating", "count"))
                 .reset_index())
        # Bayesian average: prior = global mean, weight by rating count
        global_mean = ratings_df["rating"].mean()
        C           = stats["n_ratings"].quantile(0.60)
        stats["weighted_score"] = (
            (stats["n_ratings"] / (stats["n_ratings"] + C)) * stats["avg_rating"] +
            (C / (stats["n_ratings"] + C)) * global_mean
        )
        stats = stats.merge(movies_df[["movie_id", "title", "genres"]], on="movie_id")
        top   = stats.sort_values("weighted_score", ascending=False).head(n)
        return top[["movie_id", "title", "genres", "avg_rating", "n_ratings", "weighted_score"]].reset_index(drop=True)


if __name__ == "__main__":
    from data_loader import load_all

    data    = load_all(verbose=True)
    ratings = data["ratings"]
    movies  = data["movies"]

    from collaborative_filtering import CollaborativeFilteringModel, ALSModel

    cf = CollaborativeFilteringModel(n_factors=50, n_epochs=10)
    cf.fit(ratings)

    als = ALSModel(n_factors=50, n_iterations=10)
    als.fit(ratings)

    cb = ContentBasedModel()
    cb.fit(movies, ratings)

    engine = HybridRecommender(alpha=0.7, mmr_lambda=0.8)
    engine.set_models(cf, cb, als)

    print("\n── Hybrid Recommendations for User 1 ──")
    recs = engine.recommend(user_id=1, movies_df=movies, ratings_df=ratings, n=10)
    print(recs[["title", "genres", "cf_score", "als_score", "cb_score", "hybrid_score"]].to_string())

    print("\n── Popularity Fallback (new users) ──")
    pop = HybridRecommender.popularity_fallback(movies, ratings, n=10)
    print(pop[["title", "avg_rating", "n_ratings", "weighted_score"]].to_string())
