# -*- coding: utf-8 -*-
"""
collaborative_filtering.py
==========================
Implements Collaborative Filtering using two Matrix Factorisation approaches:

1. SGD (Stochastic Gradient Descent) — the classic Netflix Prize approach
2. ALS (Alternating Least Squares) — used by Spotify and implicit-feedback systems

Both factorise the rating matrix:  R ≈ P · Qᵀ + bias terms

SGD updates all parameters simultaneously via gradient descent.
ALS alternates: fix P, solve for Q (closed-form), then fix Q, solve for P.
ALS tends to converge more reliably on very sparse matrices, while SGD
is more flexible with its learning rate schedule.

Having both lets us ensemble them for better overall accuracy.
"""

import numpy as np
import pandas as pd
import joblib
import time
from dataclasses import dataclass
from pathlib import Path
from scipy import sparse
from scipy.sparse.linalg import spsolve
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
    lr_decay   : float — multiply lr by this factor each epoch (default 0.96)
    """

    def __init__(self, n_factors: int = 100, n_epochs: int = 20,
                 lr: float = 0.005, reg: float = 0.02,
                 lr_decay: float = 0.96, random_state: int = 42):
        self.n_factors    = n_factors
        self.n_epochs     = n_epochs
        self.lr           = lr
        self.reg          = reg
        self.lr_decay     = lr_decay
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
        self.train_rmse_history = []   # track convergence

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
        print(f"{Fore.CYAN}[CF-SGD] Building index maps …")
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

        print(f"{Fore.CYAN}[CF-SGD] Training SGD Matrix Factorisation  "
              f"(factors={self.n_factors}, epochs={self.n_epochs}, "
              f"users={n_users}, items={n_items}) …")

        t0 = time.time()
        current_lr = self.lr
        self.train_rmse_history = []

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
                self.bu[u] += current_lr * (err - self.reg * self.bu[u])
                self.bi[i] += current_lr * (err - self.reg * self.bi[i])

                # Update latent factors
                P_u_old     = self.P[u].copy()
                self.P[u]  += current_lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i]  += current_lr * (err * P_u_old  - self.reg * self.Q[i])

            rmse = np.sqrt(total_loss / n)
            self.train_rmse_history.append(rmse)

            # Learning rate decay — helps fine-tune in later epochs
            current_lr *= self.lr_decay

            print(f"  Epoch {epoch+1:>2}/{self.n_epochs}  RMSE={rmse:.4f}  "
                  f"lr={current_lr:.5f}  [{time.time()-t0:.1f}s]")

        self._fitted = True
        print(f"{Fore.GREEN}[CF-SGD] Training complete ✓  "
              f"({time.time()-t0:.1f}s total)")
        return self

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate_on_testset(self, ratings_df: pd.DataFrame,
                             test_size: float = 0.2,
                             random_state: int = 42) -> dict:
        """
        Split data into train/test, train on train, evaluate on test.
        Stores self.predictions as a list of Prediction objects.

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
        print(f"{Fore.GREEN}[CF-SGD] Hold-out Test → RMSE: {rmse:.4f} | MAE: {mae:.4f}")
        return {"rmse": rmse, "mae": mae}

    def cross_validate(self, ratings_df: pd.DataFrame, cv: int = 5) -> dict:
        """
        K-fold cross-validation. Returns average RMSE and MAE.
        """
        from sklearn.model_selection import KFold

        print(f"{Fore.CYAN}[CF-SGD] Running {cv}-fold cross-validation …")
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
        print(f"{Fore.GREEN}[CF-SGD] CV Results → "
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

    def predict_with_confidence(self, user_id: int, movie_id: int, n_bootstrap: int = 25) -> dict:
        """
        Estimate prediction rating along with standard deviation (uncertainty)
        and confidence intervals (ci_lower, ci_upper) using bootstrapped noise
        addition to latent representations based on residual variance.
        """
        if not self._fitted:
            raise RuntimeError("Model not trained. Call .fit() first.")
        
        pred = self.predict_rating(user_id, movie_id)
        
        u_idx = self.user_map.get(user_id)
        i_idx = self.item_map.get(movie_id)
        
        if u_idx is None or i_idx is None:
            return {
                "predicted_rating": pred,
                "mean_prediction": pred,
                "uncertainty": 0.5,
                "ci_lower": float(np.clip(pred - 1.0, 1.0, 5.0)),
                "ci_upper": float(np.clip(pred + 1.0, 1.0, 5.0))
            }
        
        # We can add small random perturbations proportional to user/item biases
        # and factor variance to generate n_bootstrap prediction samples.
        rng = np.random.RandomState(self.random_state)
        
        # User/item vector norms indicate variance (smaller norms -> higher uncertainty)
        u_norm = np.linalg.norm(self.P[u_idx])
        i_norm = np.linalg.norm(self.Q[i_idx])
        
        # Normalise scale
        avg_u_norm = np.mean(np.linalg.norm(self.P, axis=1)) + 1e-8
        avg_i_norm = np.mean(np.linalg.norm(self.Q, axis=1)) + 1e-8
        
        # Base standard deviation of error
        base_std = 0.5 / (0.1 + (u_norm / avg_u_norm) * (i_norm / avg_i_norm))
        base_std = np.clip(base_std, 0.1, 1.2)
        
        # Generate bootstrapped predictions
        samples = []
        for _ in range(n_bootstrap):
            noise = rng.normal(0, base_std)
            samples.append(np.clip(pred + noise, 1.0, 5.0))
            
        samples = np.array(samples)
        mean_pred = float(np.mean(samples))
        std_pred = float(np.std(samples))
        ci_lower = float(np.percentile(samples, 5))
        ci_upper = float(np.percentile(samples, 95))
        
        return {
            "predicted_rating": pred,
            "mean_prediction": round(mean_pred, 3),
            "uncertainty": round(std_pred, 3),
            "ci_lower": round(ci_lower, 3),
            "ci_upper": round(ci_upper, 3)
        }


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
        print(f"{Fore.GREEN}[CF-SGD] Model saved → {path}")
        return str(path)

    def load(self, filename: str = "cf_svd_model.pkl") -> "CollaborativeFilteringModel":
        path = MODEL_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"No saved model at {path}. Train first.")
        obj = joblib.load(path)
        # Copy all attributes
        self.__dict__.update(obj.__dict__)
        print(f"{Fore.GREEN}[CF-SGD] Model loaded ← {path}")
        return self


# ══════════════════════════════════════════════════════════════════════════════
#  ALS — Alternating Least Squares
# ══════════════════════════════════════════════════════════════════════════════

class ALSModel:
    """
    Matrix Factorisation via Alternating Least Squares (ALS).

    ALS is different from SGD in how it optimises:
      - Fix user factors P → solve for item factors Q (closed-form least squares)
      - Fix item factors Q → solve for user factors P (closed-form least squares)
      - Repeat until convergence

    This approach is:
      - More numerically stable than SGD (no learning rate to tune)
      - Better suited for implicit feedback data
      - Embarrassingly parallelisable (each user/item can be solved independently)
      - Used by Spotify for music recommendations at scale

    The closed-form update for a single user u is:
      P[u] = (Q_u^T Q_u + λI)^{-1} Q_u^T r_u

    where Q_u is the submatrix of Q for items rated by user u,
    and r_u is the vector of ratings from user u.
    """

    def __init__(self, n_factors: int = 80, n_iterations: int = 15,
                 reg: float = 0.1, random_state: int = 42):
        self.n_factors    = n_factors
        self.n_iterations = n_iterations
        self.reg          = reg
        self.random_state = random_state

        self.P          = None
        self.Q          = None
        self.mu         = None
        self.user_map   = {}
        self.item_map   = {}
        self.user_ids   = []
        self.item_ids   = []
        self._fitted    = False
        self.predictions = []
        self.train_rmse_history = []

    def _build_index(self, ratings_df: pd.DataFrame) -> None:
        users = sorted(ratings_df["user_id"].unique())
        items = sorted(ratings_df["movie_id"].unique())
        self.user_map = {uid: i for i, uid in enumerate(users)}
        self.item_map = {mid: i for i, mid in enumerate(items)}
        self.user_ids = users
        self.item_ids = items

    def _predict_single(self, u_idx: int, i_idx: int) -> float:
        pred = self.mu + self.P[u_idx] @ self.Q[i_idx]
        return float(np.clip(pred, 1.0, 5.0))

    def fit(self, ratings_df: pd.DataFrame) -> "ALSModel":
        """
        Train ALS on the ratings DataFrame.

        We build a sparse user-item matrix and alternate between
        updating user factors and item factors using ridge regression.
        """
        print(f"{Fore.CYAN}[CF-ALS] Building sparse rating matrix …")
        self._build_index(ratings_df)

        n_users = len(self.user_map)
        n_items = len(self.item_map)
        rng     = np.random.RandomState(self.random_state)

        self.mu = ratings_df["rating"].mean()

        # Build sparse matrix (centered ratings)
        rows = ratings_df["user_id"].map(self.user_map).values
        cols = ratings_df["movie_id"].map(self.item_map).values
        vals = ratings_df["rating"].values.astype(np.float64) - self.mu
        R    = sparse.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
        RT   = R.T.tocsr()

        # Initialise factors
        self.P = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = rng.normal(0, 0.1, (n_items, self.n_factors))

        reg_I = self.reg * sparse.eye(self.n_factors)

        print(f"{Fore.CYAN}[CF-ALS] Training ALS  "
              f"(factors={self.n_factors}, iterations={self.n_iterations}, "
              f"users={n_users}, items={n_items}) …")

        t0 = time.time()
        self.train_rmse_history = []

        for it in range(self.n_iterations):
            # ── Fix Q, solve for P ────────────────────────────────────────
            QTQ = self.Q.T @ self.Q
            for u in range(n_users):
                # Get items rated by this user
                rated = R[u].nonzero()[1]
                if len(rated) == 0:
                    continue
                Q_u = self.Q[rated]
                r_u = R[u, rated].toarray().flatten()
                # Solve: (Q_u^T Q_u + λI) p_u = Q_u^T r_u
                A = Q_u.T @ Q_u + self.reg * np.eye(self.n_factors)
                b = Q_u.T @ r_u
                self.P[u] = np.linalg.solve(A, b)

            # ── Fix P, solve for Q ────────────────────────────────────────
            PTP = self.P.T @ self.P
            for i in range(n_items):
                rated = RT[i].nonzero()[1]
                if len(rated) == 0:
                    continue
                P_i = self.P[rated]
                r_i = RT[i, rated].toarray().flatten()
                A = P_i.T @ P_i + self.reg * np.eye(self.n_factors)
                b = P_i.T @ r_i
                self.Q[i] = np.linalg.solve(A, b)

            # ── Compute training RMSE ─────────────────────────────────────
            # only on observed entries, not the full matrix
            preds = np.array([self.P[rows[k]] @ self.Q[cols[k]] for k in range(len(rows))])
            residuals = vals - preds
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            self.train_rmse_history.append(rmse)

            print(f"  Iteration {it+1:>2}/{self.n_iterations}  RMSE={rmse:.4f}  "
                  f"[{time.time()-t0:.1f}s]")

        self._fitted = True
        print(f"{Fore.GREEN}[CF-ALS] Training complete ✓  "
              f"({time.time()-t0:.1f}s total)")
        return self

    def evaluate_on_testset(self, ratings_df: pd.DataFrame,
                             test_size: float = 0.2,
                             random_state: int = 42) -> dict:
        """Split data, train on train, evaluate on test."""
        from sklearn.model_selection import train_test_split

        train_df, test_df = train_test_split(ratings_df, test_size=test_size,
                                              random_state=random_state)
        train_df = train_df.reset_index(drop=True)
        test_df  = test_df.reset_index(drop=True)

        self.fit(train_df)

        errors_sq, errors_abs = [], []
        preds = []
        for _, row in test_df.iterrows():
            uid = int(row["user_id"])
            mid = int(row["movie_id"])
            r   = float(row["rating"])

            if uid in self.user_map and mid in self.item_map:
                est = self._predict_single(self.user_map[uid], self.item_map[mid])
            else:
                est = self.mu

            errors_sq.append((r - est) ** 2)
            errors_abs.append(abs(r - est))
            preds.append(Prediction(uid=str(uid), iid=str(mid), r_ui=r, est=est))

        self.predictions = preds
        rmse = float(np.sqrt(np.mean(errors_sq)))
        mae  = float(np.mean(errors_abs))
        print(f"{Fore.GREEN}[CF-ALS] Hold-out Test → RMSE: {rmse:.4f} | MAE: {mae:.4f}")
        return {"rmse": rmse, "mae": mae}

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict the rating user_id would give movie_id."""
        if not self._fitted:
            raise RuntimeError("ALS model not trained. Call .fit() first.")
        u_idx = self.user_map.get(user_id)
        i_idx = self.item_map.get(movie_id)
        if u_idx is None or i_idx is None:
            return round(self.mu, 3)
        return round(self._predict_single(u_idx, i_idx), 3)

    def get_top_n_recommendations(self, user_id: int, movies_df: pd.DataFrame,
                                  ratings_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Get top-N recommendations (exclude already-rated movies)."""
        if not self._fitted:
            raise RuntimeError("ALS model not trained.")

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

    def save(self, filename: str = "cf_als_model.pkl") -> str:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / filename
        joblib.dump(self, path)
        print(f"{Fore.GREEN}[CF-ALS] Model saved → {path}")
        return str(path)

    def load(self, filename: str = "cf_als_model.pkl") -> "ALSModel":
        path = MODEL_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"No saved ALS model at {path}. Train first.")
        obj = joblib.load(path)
        self.__dict__.update(obj.__dict__)
        print(f"{Fore.GREEN}[CF-ALS] Model loaded ← {path}")
        return self


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_ratings, load_movies

    ratings = load_ratings()
    movies  = load_movies()

    # Quick demo of both models
    print("\n── SGD Model ──")
    cf = CollaborativeFilteringModel(n_factors=50, n_epochs=10)
    cf.evaluate_on_testset(ratings, test_size=0.2)
    cf.save()

    print("\n── ALS Model ──")
    als = ALSModel(n_factors=50, n_iterations=10)
    als.evaluate_on_testset(ratings, test_size=0.2)
    als.save()

    recs = cf.get_top_n_recommendations(user_id=1, movies_df=movies, ratings_df=ratings, n=5)
    print(f"\nSGD recs for user 1:\n{recs}")

    recs_als = als.get_top_n_recommendations(user_id=1, movies_df=movies, ratings_df=ratings, n=5)
    print(f"\nALS recs for user 1:\n{recs_als}")
