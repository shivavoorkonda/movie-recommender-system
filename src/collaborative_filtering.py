"""
collaborative_filtering.py
==========================
Implements Collaborative Filtering using Matrix Factorisation (SVD-style)
built directly on top of NumPy + SciPy — no compiled C extensions required.

Algorithm: Regularised Matrix Factorisation via SGD (same maths as SVD in surprise)

How it works:
  The rating matrix R (users × movies) is factored into:
        R ≈ P · Qᵀ   +  bias terms

  where:
    P[u]   = user  latent factor vector   (shape: n_factors)
    Q[i]   = movie latent factor vector   (shape: n_factors)
    bu[u]  = user  bias (how generous this user rates overall)
    bi[i]  = item  bias (how popular this movie is globally)
    mu     = global mean rating

  Predicted rating:
    r̂(u,i) = mu + bu[u] + bi[i] + P[u] · Q[i]

  We minimise MSE + L2 regularisation via SGD (stochastic gradient descent):
    Loss = Σ (r(u,i) - r̂(u,i))²  +  λ(|P|² + |Q|² + bu² + bi²)

This is exactly what Netflix-Prize winning SVD models do.
"""

import numpy as np
import pandas as pd
import joblib
import time
from dataclasses import dataclass
from pathlib import Path
from colorama import Fore, Style, init

init(autoreset=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR    = PROJECT_ROOT / "models" / "saved"


@dataclass
class Prediction:
    """
    Simple container for a single rating prediction.
    Mirrors the interface used by evaluator.py.
    """
    uid:  str    # user_id (as string, matching Surprise convention)
    iid:  str    # item/movie_id (as string)
    r_ui: float  # actual rating
    est:  float  # estimated/predicted rating


class CollaborativeFilteringModel:
    """
    Regularised Matrix Factorisation Recommender (SGD).

    Parameters
    ----------
    n_factors  : int   — latent dimension size (default 100)
    n_epochs   : int   — SGD training passes over the data (default 20)
    lr         : float — learning rate (default 0.005)
    reg        : float — L2 regularisation strength (default 0.02)
    """

    def __init__(self, n_factors: int = 100, n_epochs: int = 20,
                 lr: float = 0.005, reg: float = 0.02, random_state: int = 42):
        self.n_factors    = n_factors
        self.n_epochs     = n_epochs
        self.lr           = lr
        self.reg          = reg
        self.random_state = random_state

        # Will be set during fit()
        self.P            = None   # user factor matrix  (n_users  × n_factors)
        self.Q            = None   # item factor matrix  (n_movies × n_factors)
        self.bu           = None   # user biases         (n_users,)
        self.bi           = None   # item biases         (n_movies,)
        self.mu           = None   # global mean
        self.user_map     = {}     # user_id   → row index
        self.item_map     = {}     # movie_id  → col index
        self.user_ids     = []     # index → user_id
        self.item_ids     = []     # index → movie_id
        self._fitted      = False
        self.predictions  = []     # stored after evaluate_on_testset

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_index(self, ratings_df: pd.DataFrame) -> None:
        """Create integer index maps for users and items."""
        users = sorted(ratings_df["user_id"].unique())
        items = sorted(ratings_df["movie_id"].unique())
        self.user_map = {uid: i for i, uid in enumerate(users)}
        self.item_map = {mid: i for i, mid in enumerate(items)}
        self.user_ids = users
        self.item_ids = items

    def _predict_single(self, u_idx: int, i_idx: int) -> float:
        """Internal: predict rating for integer indices."""
        pred = (self.mu
                + self.bu[u_idx]
                + self.bi[i_idx]
                + self.P[u_idx] @ self.Q[i_idx])
        # Clamp to valid rating range
        return float(np.clip(pred, 1.0, 5.0))

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, ratings_df: pd.DataFrame) -> "CollaborativeFilteringModel":
        """
        Train Matrix Factorisation via SGD on the full ratings DataFrame.

        Parameters
        ----------
        ratings_df : pd.DataFrame  with columns ['user_id', 'movie_id', 'rating']
        """
        print(f"{Fore.CYAN}[CF] Building index maps …")
        self._build_index(ratings_df)

        n_users = len(self.user_map)
        n_items = len(self.item_map)
        rng     = np.random.RandomState(self.random_state)

        # Initialise factor matrices (small random values)
        self.P  = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q  = rng.normal(0, 0.1, (n_items, self.n_factors))
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.mu = ratings_df["rating"].mean()

        # Convert to numpy arrays for fast iteration
        u_arr = ratings_df["user_id"].map(self.user_map).values
        i_arr = ratings_df["movie_id"].map(self.item_map).values
        r_arr = ratings_df["rating"].values.astype(np.float64)
        n     = len(r_arr)

        print(f"{Fore.CYAN}[CF] Training SGD Matrix Factorisation  "
              f"(factors={self.n_factors}, epochs={self.n_epochs}, "
              f"users={n_users}, items={n_items}) …")

        t0 = time.time()
        for epoch in range(self.n_epochs):
            # Shuffle data each epoch
            perm = rng.permutation(n)
            u_arr_s, i_arr_s, r_arr_s = u_arr[perm], i_arr[perm], r_arr[perm]

            total_loss = 0.0
            for k in range(n):
                u = u_arr_s[k]
                i = i_arr_s[k]
                r = r_arr_s[k]

                # Compute error
                pred  = self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]
                err   = r - pred
                total_loss += err ** 2

                # Update biases
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])

                # Update latent factors
                P_u_old     = self.P[u].copy()
                self.P[u]  += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i]  += self.lr * (err * P_u_old  - self.reg * self.Q[i])

            rmse = np.sqrt(total_loss / n)
            print(f"  Epoch {epoch+1:>2}/{self.n_epochs}  RMSE={rmse:.4f}  "
                  f"[{time.time()-t0:.1f}s]")

        self._fitted = True
        print(f"{Fore.GREEN}[CF] Training complete ✓  "
              f"({time.time()-t0:.1f}s total)")
        return self

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate_on_testset(self, ratings_df: pd.DataFrame,
                             test_size: float = 0.2,
                             random_state: int = 42) -> dict:
        """
        Split data into train/test, train on train, evaluate on test.
        Stores self.predictions as a list of (true, predicted) namedtuple-like objects.

        Returns dict with 'rmse' and 'mae'.
        """
        from sklearn.model_selection import train_test_split as sk_split

        train_df, test_df = sk_split(ratings_df, test_size=test_size,
                                     random_state=random_state)
        train_df = train_df.reset_index(drop=True)
        test_df  = test_df.reset_index(drop=True)

        # Refit on training data only
        self.fit(train_df)

        # Predict on test
        errors_sq, errors_abs = [], []
        preds = []
        for _, row in test_df.iterrows():
            uid = int(row["user_id"])
            mid = int(row["movie_id"])
            r   = float(row["rating"])

            if uid in self.user_map and mid in self.item_map:
                est = self._predict_single(self.user_map[uid], self.item_map[mid])
            else:
                est = self.mu  # fallback to global mean

            errors_sq.append((r - est) ** 2)
            errors_abs.append(abs(r - est))
            preds.append(Prediction(uid=str(uid), iid=str(mid), r_ui=r, est=est))

        self.predictions = preds
        rmse = float(np.sqrt(np.mean(errors_sq)))
        mae  = float(np.mean(errors_abs))
        print(f"{Fore.GREEN}[CF] Hold-out Test → RMSE: {rmse:.4f} | MAE: {mae:.4f}")
        return {"rmse": rmse, "mae": mae}

    def cross_validate(self, ratings_df: pd.DataFrame, cv: int = 5) -> dict:
        """
        K-fold cross-validation. Returns average RMSE and MAE.
        """
        from sklearn.model_selection import KFold

        print(f"{Fore.CYAN}[CF] Running {cv}-fold cross-validation …")
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        rmses, maes = [], []

        for fold, (train_idx, test_idx) in enumerate(kf.split(ratings_df)):
            train_df = ratings_df.iloc[train_idx].reset_index(drop=True)
            test_df  = ratings_df.iloc[test_idx].reset_index(drop=True)

            self.fit(train_df)

            fold_sq, fold_abs = [], []
            for _, row in test_df.iterrows():
                uid = int(row["user_id"])
                mid = int(row["movie_id"])
                r   = float(row["rating"])
                if uid in self.user_map and mid in self.item_map:
                    est = self._predict_single(self.user_map[uid], self.item_map[mid])
                else:
                    est = self.mu
                fold_sq.append((r - est) ** 2)
                fold_abs.append(abs(r - est))

            rmses.append(np.sqrt(np.mean(fold_sq)))
            maes.append(np.mean(fold_abs))
            print(f"  Fold {fold+1}/{cv}  RMSE={rmses[-1]:.4f}  MAE={maes[-1]:.4f}")

        summary = {
            "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
            "mae_mean" : float(np.mean(maes)),  "mae_std" : float(np.std(maes)),
        }
        print(f"{Fore.GREEN}[CF] CV Results → "
              f"RMSE: {summary['rmse_mean']:.4f} ± {summary['rmse_std']:.4f} | "
              f"MAE : {summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}")

        # Retrain on full data after CV
        self.fit(ratings_df)
        return summary

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict the rating user_id would give movie_id."""
        if not self._fitted:
            raise RuntimeError("Model not trained. Call .fit() first.")
        u_idx = self.user_map.get(user_id)
        i_idx = self.item_map.get(movie_id)
        if u_idx is None or i_idx is None:
            return round(self.mu, 3)
        return round(self._predict_single(u_idx, i_idx), 3)

    def get_top_n_recommendations(self, user_id: int, movies_df: pd.DataFrame,
                                  ratings_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """
        Get top-N movie recommendations for a user.
        Returns only movies the user has NOT yet rated.
        """
        if not self._fitted:
            raise RuntimeError("Model not trained. Call .fit() first.")

        rated_ids = set(ratings_df[ratings_df["user_id"] == user_id]["movie_id"].tolist())
        u_idx     = self.user_map.get(user_id)

        preds = []
        for _, row in movies_df.iterrows():
            mid = int(row["movie_id"])
            if mid not in rated_ids:
                i_idx = self.item_map.get(mid)
                if u_idx is not None and i_idx is not None:
                    est = self._predict_single(u_idx, i_idx)
                else:
                    est = float(self.mu)
                preds.append({"movie_id": mid, "predicted_rating": round(est, 3)})

        pred_df = pd.DataFrame(preds)
        result  = pred_df.merge(movies_df[["movie_id", "title", "genres"]], on="movie_id")
        result  = result.sort_values("predicted_rating", ascending=False).head(n)
        return result.reset_index(drop=True)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, filename: str = "cf_svd_model.pkl") -> str:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / filename
        joblib.dump(self, path)
        print(f"{Fore.GREEN}[CF] Model saved → {path}")
        return str(path)

    def load(self, filename: str = "cf_svd_model.pkl") -> "CollaborativeFilteringModel":
        path = MODEL_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"No saved model at {path}. Train first.")
        obj = joblib.load(path)
        # Copy all attributes
        self.__dict__.update(obj.__dict__)
        print(f"{Fore.GREEN}[CF] Model loaded ← {path}")
        return self


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_ratings, load_movies

    ratings = load_ratings()
    movies  = load_movies()

    cf = CollaborativeFilteringModel(n_factors=50, n_epochs=10)
    cf.evaluate_on_testset(ratings, test_size=0.2)
    cf.save()

    recs = cf.get_top_n_recommendations(user_id=1, movies_df=movies, ratings_df=ratings, n=10)
    print(recs)
