"""
tests/test_recommender.py
=========================
Unit tests for all three recommender components.

Run with:
    cd movie_recommender_system
    python -m pytest tests/ -v
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_loader             import load_ratings, load_movies
from collaborative_filtering import CollaborativeFilteringModel
from content_based           import ContentBasedModel
from hybrid_engine           import HybridRecommender
from evaluator               import precision_recall_at_k, f1_at_k, intra_list_diversity


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ratings():
    return load_ratings(verbose=False)

@pytest.fixture(scope="module")
def movies():
    return load_movies(verbose=False)

@pytest.fixture(scope="module")
def cf_model(ratings):
    """Train a lightweight CF model (fewer epochs for speed)."""
    cf = CollaborativeFilteringModel(n_factors=20, n_epochs=3)
    cf.evaluate_on_testset(ratings, test_size=0.2, random_state=42)
    return cf

@pytest.fixture(scope="module")
def cb_model(movies):
    cb = ContentBasedModel()
    cb.fit(movies)
    return cb

@pytest.fixture(scope="module")
def hybrid(cf_model, cb_model):
    engine = HybridRecommender(alpha=0.7)
    engine.set_models(cf_model, cb_model)
    return engine


# ── Data Loading Tests ────────────────────────────────────────────────────────

class TestDataLoader:
    def test_ratings_shape(self, ratings):
        assert len(ratings) == 100_000, "MovieLens 100K must have 100,000 ratings"

    def test_ratings_columns(self, ratings):
        for col in ["user_id", "movie_id", "rating"]:
            assert col in ratings.columns

    def test_rating_range(self, ratings):
        assert ratings["rating"].min() >= 1
        assert ratings["rating"].max() <= 5

    def test_movies_columns(self, movies):
        for col in ["movie_id", "title", "genres", "genre_list"]:
            assert col in movies.columns

    def test_movies_count(self, movies):
        assert len(movies) > 1600, "Should have ~1682 movies"


# ── Collaborative Filtering Tests ─────────────────────────────────────────────

class TestCollaborativeFiltering:
    def test_model_is_fitted(self, cf_model):
        assert cf_model._fitted is True

    def test_predict_returns_float(self, cf_model):
        pred = cf_model.predict_rating(user_id=1, movie_id=1)
        assert isinstance(pred, float)
        assert 1.0 <= pred <= 5.0

    def test_factor_matrices_shape(self, cf_model):
        n_users = len(cf_model.user_map)
        n_items = len(cf_model.item_map)
        assert cf_model.P.shape == (n_users, cf_model.n_factors)
        assert cf_model.Q.shape == (n_items, cf_model.n_factors)

    def test_recommendations_shape(self, cf_model, movies, ratings):
        recs = cf_model.get_top_n_recommendations(1, movies, ratings, n=10)
        assert len(recs) == 10
        assert "title"            in recs.columns
        assert "predicted_rating" in recs.columns

    def test_recommendations_no_already_rated(self, cf_model, movies, ratings):
        uid   = 1
        rated = set(ratings[ratings["user_id"] == uid]["movie_id"].tolist())
        recs  = cf_model.get_top_n_recommendations(uid, movies, ratings, n=20)
        assert len(set(recs["movie_id"]) & rated) == 0, \
            "Recommendations must not include already-rated movies"

    def test_predictions_stored(self, cf_model):
        assert hasattr(cf_model, "predictions")
        assert len(cf_model.predictions) > 0

    def test_predictions_range(self, cf_model):
        for p in cf_model.predictions[:100]:
            assert 1.0 <= p.est <= 5.0

    def test_global_mean_in_range(self, cf_model):
        assert 1.0 <= cf_model.mu <= 5.0


# ── Content-Based Tests ───────────────────────────────────────────────────────

class TestContentBased:
    def test_model_is_fitted(self, cb_model):
        assert cb_model._fitted is True

    def test_tfidf_matrix_shape(self, cb_model, movies):
        rows, _ = cb_model.tfidf_matrix.shape
        assert rows == len(movies)

    def test_sim_matrix_shape(self, cb_model, movies):
        n = len(movies)
        assert cb_model.sim_matrix.shape == (n, n)

    def test_sim_matrix_symmetric(self, cb_model):
        n   = 50
        sub = cb_model.sim_matrix[:n, :n]
        assert np.allclose(sub, sub.T, atol=1e-5), "Cosine similarity must be symmetric"

    def test_sim_matrix_diagonal(self, cb_model):
        diag = np.diag(cb_model.sim_matrix[:100, :100])
        assert np.allclose(diag, 1.0, atol=1e-5), "Self-similarity must be 1.0"

    def test_similar_movies(self, cb_model):
        sims = cb_model.get_similar_movies(movie_id=1, n=10)
        assert len(sims) == 10
        assert 1 not in sims["movie_id"].values, "Must not return query movie itself"
        assert all(sims["similarity_score"] >= 0)
        assert all(sims["similarity_score"] <= 1)

    def test_similarity_sorted_descending(self, cb_model):
        sims   = cb_model.get_similar_movies(movie_id=1, n=10)
        scores = sims["similarity_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_user_profile_recommendations(self, cb_model, ratings):
        recs = cb_model.get_user_profile_recommendations(user_id=1, ratings_df=ratings, n=10)
        assert len(recs) <= 10
        assert "similarity_score" in recs.columns


# ── Hybrid Engine Tests ───────────────────────────────────────────────────────

class TestHybridEngine:
    def test_recommend_shape(self, hybrid, movies, ratings):
        recs = hybrid.recommend(user_id=1, movies_df=movies, ratings_df=ratings, n=10)
        assert len(recs) == 10

    def test_hybrid_columns(self, hybrid, movies, ratings):
        recs = hybrid.recommend(user_id=1, movies_df=movies, ratings_df=ratings, n=10)
        for col in ["movie_id", "title", "genres", "cf_score", "cb_score", "hybrid_score"]:
            assert col in recs.columns

    def test_hybrid_scores_in_range(self, hybrid, movies, ratings):
        recs = hybrid.recommend(user_id=1, movies_df=movies, ratings_df=ratings, n=10)
        assert recs["hybrid_score"].between(0, 1).all()

    def test_hybrid_sorted_descending(self, hybrid, movies, ratings):
        recs   = hybrid.recommend(user_id=1, movies_df=movies, ratings_df=ratings, n=10)
        scores = recs["hybrid_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_popularity_fallback(self, movies, ratings):
        pop = HybridRecommender.popularity_fallback(movies, ratings, n=10)
        assert len(pop) == 10
        assert "weighted_score" in pop.columns
        scores = pop["weighted_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_cold_start_alpha(self, hybrid, ratings):
        # Verify alpha reduction for sparse users
        cold_uid  = 943
        n_rated   = len(ratings[ratings["user_id"] == cold_uid])
        eff_alpha = hybrid._effective_alpha(cold_uid, ratings)
        if n_rated < hybrid.cold_start_threshold:
            assert eff_alpha < hybrid.alpha
        else:
            assert eff_alpha == hybrid.alpha


# ── Evaluator Tests ───────────────────────────────────────────────────────────

class TestEvaluator:
    def test_precision_recall_at_k(self, cf_model):
        p, r = precision_recall_at_k(cf_model.predictions, k=10, threshold=3.5)
        assert 0.0 <= p <= 1.0
        assert 0.0 <= r <= 1.0

    def test_f1_at_k(self):
        assert f1_at_k(0.5, 0.5) == pytest.approx(0.5)
        assert f1_at_k(0.0, 0.0) == 0.0

    def test_f1_increases_with_both(self):
        f1_low  = f1_at_k(0.2, 0.2)
        f1_high = f1_at_k(0.8, 0.8)
        assert f1_high > f1_low

    def test_intra_list_diversity(self, cb_model, hybrid, movies, ratings):
        recs = hybrid.recommend(user_id=1, movies_df=movies, ratings_df=ratings, n=10)
        ild  = intra_list_diversity(recs, cb_model)
        assert 0.0 <= ild <= 1.0
