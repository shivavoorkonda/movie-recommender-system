# -*- coding: utf-8 -*-
"""
content_based.py
================
Implements Content-Based Filtering using TF-IDF + Cosine Similarity on movie genres,
enhanced with demographic-aware user profiling and temporal similarity.

How Content-Based Filtering Works:
  1. Build a "profile" for each movie from its features (genres + decade).
  2. Vectorise these profiles using TF-IDF:
       - TF  = how often a term appears in a movie's feature string
       - IDF = penalises terms that appear in EVERY movie (not discriminating)
  3. Compute cosine similarity between all pairs of movie vectors:
       - cos(A, B) = (A · B) / (|A| * |B|)   → 1 = identical, 0 = completely different
  4. For a given query movie, return the movies with highest cosine similarity.

Enhancements over basic TF-IDF:
  - Release decade is included as a feature (captures era preferences)
  - User profiles weight movies by their rating (5-star movies contribute more)
  - Popularity penalty reduces bias towards universally-watched blockbusters
  - Genre pair co-occurrences are preserved via bigram features
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from colorama import Fore, Style, init

init(autoreset=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR    = PROJECT_ROOT / "models" / "saved"


class ContentBasedModel:
    """
    Content-Based recommender using TF-IDF over movie genre strings.

    Attributes
    ----------
    tfidf       : TfidfVectorizer  — the fitted vectoriser
    tfidf_matrix: np.ndarray       — shape (n_movies, n_features)
    sim_matrix  : np.ndarray       — shape (n_movies, n_movies) cosine similarity
    movie_index : dict             — movie_id → row index in sim_matrix
    """

    def __init__(self):
        self.tfidf        = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),   # unigrams and bigrams
            min_df=1,             # minimum 1 document (include all terms)
            stop_words="english",
        )
        self.tfidf_matrix = None
        self.sim_matrix   = None
        self.movies_df    = None
        self.movie_index  = {}   # movie_id  → integer index
        self.index_movie  = {}   # integer index → movie_id
        self._fitted      = False
        self._movie_popularity = {}   # movie_id → log(n_ratings)

    # ── Feature Engineering ───────────────────────────────────────────────────

    def _build_feature_string(self, row: pd.Series) -> str:
        """
        Combine movie features into one text blob for TF-IDF.

        We include:
          - Genre tokens (repeated 3× for importance weighting)
          - Decade token (e.g., "decade_1990s") to capture era preferences
          - Genre pairs appear naturally through bigram extraction

        This is deliberately simple — in a production system you'd use
        movie descriptions, cast, director, and plot keywords too. But for
        MovieLens 100K which only has genre metadata, this works well.
        """
        genres  = " ".join(row["genre_list"]) if isinstance(row.get("genre_list"), list) else str(row.get("genres", ""))
        genres  = genres.replace("|", " ").replace("-", " ")
        # Repeat genres 3× to give them more weight vs. other features
        features = f"{genres} {genres} {genres}"

        # Add decade as a feature — movies from the same era tend to be similar
        year = row.get("year")
        if pd.notna(year) and year > 0:
            decade = int(year // 10 * 10)
            features += f" decade_{decade}s"

        return features.lower().strip()

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame = None) -> "ContentBasedModel":
        """
        Build TF-IDF matrix and cosine-similarity matrix from movies_df.

        Parameters
        ----------
        movies_df  : pd.DataFrame  must have ['movie_id', 'title', 'genres', 'genre_list']
        ratings_df : pd.DataFrame  optional, used to compute popularity for debiasing
        """
        print(f"{Fore.CYAN}[CB] Building TF-IDF feature matrix …")
        self.movies_df = movies_df.reset_index(drop=True).copy()

        # Build index maps
        for idx, movie_id in enumerate(self.movies_df["movie_id"]):
            self.movie_index[int(movie_id)] = idx
            self.index_movie[idx]           = int(movie_id)

        # Build feature strings
        self.movies_df["feature_str"] = self.movies_df.apply(self._build_feature_string, axis=1)

        # TF-IDF fit + transform
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies_df["feature_str"])
        print(f"{Fore.CYAN}[CB] TF-IDF matrix shape: {self.tfidf_matrix.shape}")

        # Cosine similarity (n_movies × n_movies)
        print(f"{Fore.CYAN}[CB] Computing cosine similarity matrix …")
        self.sim_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        # Pre-compute popularity scores if ratings data is available
        if ratings_df is not None:
            pop = ratings_df.groupby("movie_id")["rating"].count()
            max_pop = pop.max()
            self._movie_popularity = {
                mid: float(np.log1p(cnt) / np.log1p(max_pop))
                for mid, cnt in pop.items()
            }

        self._fitted = True
        print(f"{Fore.GREEN}[CB] Content-Based model ready ✓  "
              f"(similarity matrix: {self.sim_matrix.shape})")
        return self

    # ── Recommendation ────────────────────────────────────────────────────────

    def get_similar_movies(self, movie_id: int, n: int = 10) -> pd.DataFrame:
        """
        Return the top-N movies most similar to a given movie.

        Parameters
        ----------
        movie_id : int  — the reference movie
        n        : int  — how many similar movies to return

        Returns
        -------
        pd.DataFrame  columns: ['movie_id', 'title', 'genres', 'similarity_score']
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted yet. Call .fit() first.")

        if movie_id not in self.movie_index:
            raise ValueError(f"movie_id {movie_id} not found in training data.")

        idx   = self.movie_index[movie_id]
        scores = list(enumerate(self.sim_matrix[idx]))

        # Sort descending, skip the movie itself (score=1.0)
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [(i, s) for i, s in scores if i != idx][:n]

        recs = []
        for i, score in scores:
            mid  = self.index_movie[i]
            row  = self.movies_df[self.movies_df["movie_id"] == mid].iloc[0]
            recs.append({
                "movie_id"        : mid,
                "title"           : row["title"],
                "genres"          : row["genres"],
                "similarity_score": round(float(score), 4),
            })

        return pd.DataFrame(recs).reset_index(drop=True)

    def get_user_profile_recommendations(self, user_id: int, ratings_df: pd.DataFrame,
                                          n: int = 10, min_rating: float = 4.0,
                                          popularity_penalty: float = 0.15) -> pd.DataFrame:
        """
        Recommend movies based on a user's highly-rated movies.

        Approach:
          1. Filter user's ratings ≥ min_rating (movies they loved).
          2. Average the TF-IDF vectors of those movies → user profile vector.
          3. Compute cosine similarity of profile with all movies.
          4. Apply popularity penalty to avoid always recommending blockbusters.
          5. Exclude already-rated movies, return top-N.

        Parameters
        ----------
        user_id            : int
        ratings_df         : pd.DataFrame  ['user_id', 'movie_id', 'rating']
        n                  : int
        min_rating         : float  — threshold to consider a movie "liked"
        popularity_penalty : float  — how much to penalise popular movies (0 = no penalty)

        Returns
        -------
        pd.DataFrame  columns: ['movie_id', 'title', 'genres', 'similarity_score']
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted yet. Call .fit() first.")

        user_ratings = ratings_df[
            (ratings_df["user_id"] == user_id) &
            (ratings_df["rating"]  >= min_rating)
        ]

        if len(user_ratings) == 0:
            print(f"{Fore.YELLOW}[CB] User {user_id} has no ratings ≥ {min_rating}. "
                  "Falling back to all their ratings.")
            user_ratings = ratings_df[ratings_df["user_id"] == user_id]

        rated_ids = set(ratings_df[ratings_df["user_id"] == user_id]["movie_id"].tolist())

        # Build user profile as weighted average of liked-movie TF-IDF vectors
        profile_vectors = []
        for _, r in user_ratings.iterrows():
            mid = int(r["movie_id"])
            if mid in self.movie_index:
                idx = self.movie_index[mid]
                vec = self.tfidf_matrix[idx].toarray().flatten()
                # Weight by (rating / 5) so 5-star movies contribute more
                profile_vectors.append(vec * (r["rating"] / 5.0))

        if not profile_vectors:
            raise ValueError(f"No valid movies found for user {user_id} in TF-IDF index.")

        user_profile = np.mean(profile_vectors, axis=0).reshape(1, -1)
        sims         = cosine_similarity(user_profile, self.tfidf_matrix).flatten()

        recs = []
        for idx, score in enumerate(sims):
            mid = self.index_movie[idx]
            if mid not in rated_ids:
                # Popularity penalty: reduce score for very popular movies
                # so we don't just recommend the same blockbusters to everyone
                pop = self._movie_popularity.get(mid, 0.5)
                adjusted_score = score * (1.0 - popularity_penalty * pop)

                row = self.movies_df[self.movies_df["movie_id"] == mid].iloc[0]
                recs.append({
                    "movie_id"        : mid,
                    "title"           : row["title"],
                    "genres"          : row["genres"],
                    "similarity_score": round(float(adjusted_score), 4),
                })

        recs_df = pd.DataFrame(recs).sort_values("similarity_score", ascending=False).head(n)
        return recs_df.reset_index(drop=True)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, filename: str = "cb_model.pkl") -> str:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / filename
        joblib.dump(self, path)
        print(f"{Fore.GREEN}[CB] Model saved → {path}")
        return str(path)

    @classmethod
    def load_from_disk(cls, filename: str = "cb_model.pkl") -> "ContentBasedModel":
        path = MODEL_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"No saved model at {path}. Fit first.")
        obj = joblib.load(path)
        if not hasattr(obj, '_movie_popularity'):
            obj._movie_popularity = {}
        print(f"{Fore.GREEN}[CB] Model loaded ← {path}")
        return obj



if __name__ == "__main__":
    from data_loader import load_movies, load_ratings

    movies  = load_movies()
    ratings = load_ratings()

    cb = ContentBasedModel()
    cb.fit(movies, ratings)
    cb.save()

    print("\n── Movies similar to Toy Story (id=1) ──")
    print(cb.get_similar_movies(movie_id=1, n=10))

    print("\n── Content-Based recs for user 1 ──")
    print(cb.get_user_profile_recommendations(user_id=1, ratings_df=ratings, n=10))
