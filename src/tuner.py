# -*- coding: utf-8 -*-
"""
tuner.py
========
Hyperparameter tuning for the Movie Recommender System.

Uses random search with intelligent sampling — no extra dependencies needed.
Logs every trial so you can inspect what worked and what didn't.

Why random search over grid search?
  - Grid search wastes time on bad combinations
  - Random search finds good configs ~10x faster (Bergstra & Bengio, 2012)
  - We can stop early if we find something good enough
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from colorama import Fore, Style, init
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent))
from collaborative_filtering import CollaborativeFilteringModel
from hybrid_engine import HybridRecommender
from evaluator import precision_recall_at_k

init(autoreset=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
try:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass


class HyperparamTuner:
    """
    Random-search hyperparameter tuner for CF and Hybrid models.

    All trials are logged with their configs and metrics so you
    can look back later and understand the parameter landscape.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng          = np.random.RandomState(random_state)
        self.cf_trials    = []    # list of dicts: {params, rmse, mae, duration}
        self.hybrid_trials = []
        self.best_cf_params    = None
        self.best_hybrid_params = None

    # ── CF Tuning ─────────────────────────────────────────────────────────────

    def tune_cf(self, ratings_df, n_trials: int = 20, cv: int = 3) -> dict:
        """
        Random search over CF-SGD hyperparameters using k-fold CV.

        Search space:
          n_factors : [20, 50, 80, 100, 150]
          lr        : log-uniform in [0.001, 0.01]
          reg       : log-uniform in [0.005, 0.05]
          n_epochs  : [10, 15, 20, 30]
        """
        from sklearn.model_selection import KFold

        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}  Hyperparameter Tuning — CF-SGD ({n_trials} trials, {cv}-fold CV)")
        print(f"{Fore.CYAN}{'='*60}")

        factor_choices = [20, 50, 80, 100, 150]
        epoch_choices  = [10, 15, 20, 30]

        best_rmse = np.inf

        for trial in range(1, n_trials + 1):
            params = {
                "n_factors": int(self.rng.choice(factor_choices)),
                "lr"       : float(np.exp(self.rng.uniform(np.log(0.001), np.log(0.01)))),
                "reg"      : float(np.exp(self.rng.uniform(np.log(0.005), np.log(0.05)))),
                "n_epochs" : int(self.rng.choice(epoch_choices)),
                "lr_decay" : float(self.rng.choice([0.93, 0.95, 0.97, 1.0])),
            }

            print(f"\n{Fore.YELLOW}Trial {trial}/{n_trials}: "
                  f"factors={params['n_factors']}, lr={params['lr']:.4f}, "
                  f"reg={params['reg']:.4f}, epochs={params['n_epochs']}")

            kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            fold_rmses, fold_maes = [], []
            t0 = time.time()

            for fold, (train_idx, val_idx) in enumerate(kf.split(ratings_df)):
                train_df = ratings_df.iloc[train_idx].reset_index(drop=True)
                val_df   = ratings_df.iloc[val_idx].reset_index(drop=True)

                model = CollaborativeFilteringModel(
                    n_factors=params["n_factors"],
                    n_epochs =params["n_epochs"],
                    lr       =params["lr"],
                    reg      =params["reg"],
                    lr_decay =params["lr_decay"],
                )
                model.fit(train_df)

                # Evaluate on validation fold
                errors_sq, errors_abs = [], []
                for _, row in val_df.iterrows():
                    uid, mid, r = int(row["user_id"]), int(row["movie_id"]), float(row["rating"])
                    if uid in model.user_map and mid in model.item_map:
                        est = model._predict_single(model.user_map[uid], model.item_map[mid])
                    else:
                        est = model.mu
                    errors_sq.append((r - est) ** 2)
                    errors_abs.append(abs(r - est))

                fold_rmses.append(float(np.sqrt(np.mean(errors_sq))))
                fold_maes.append(float(np.mean(errors_abs)))
                print(f"  Fold {fold+1}/{cv}: RMSE={fold_rmses[-1]:.4f}")

            mean_rmse = float(np.mean(fold_rmses))
            mean_mae  = float(np.mean(fold_maes))
            duration  = time.time() - t0

            trial_record = {
                "trial"   : trial,
                "params"  : params,
                "rmse"    : round(mean_rmse, 4),
                "mae"     : round(mean_mae,  4),
                "duration": round(duration,  1),
            }
            self.cf_trials.append(trial_record)

            status = f"{Fore.GREEN}✓ NEW BEST" if mean_rmse < best_rmse else ""
            print(f"  → Mean RMSE={mean_rmse:.4f}, MAE={mean_mae:.4f}  {status}")

            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                self.best_cf_params = params.copy()

        print(f"\n{Fore.GREEN}Best CF config found:")
        for k, v in self.best_cf_params.items():
            print(f"  {k}: {v}")
        return self.best_cf_params

    # ── Hybrid Tuning ─────────────────────────────────────────────────────────

    def tune_hybrid(self, cf_model, cb_model, ratings_df, movies_df,
                    n_trials: int = 15) -> dict:
        """
        Tune hybrid engine alpha and cold_start_threshold.
        Uses Precision@10 as the objective (maximise).
        """
        from sklearn.model_selection import train_test_split

        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}  Hyperparameter Tuning — Hybrid Engine ({n_trials} trials)")
        print(f"{Fore.CYAN}{'='*60}")

        _, val_df = train_test_split(ratings_df, test_size=0.15, random_state=42)
        best_p10 = -np.inf

        for trial in range(1, n_trials + 1):
            alpha = float(self.rng.uniform(0.2, 0.9))
            cst   = int(self.rng.choice([3, 5, 8, 10, 15]))
            mmr   = float(self.rng.choice([0.6, 0.7, 0.8, 0.9, 1.0]))

            print(f"\n{Fore.YELLOW}Trial {trial}/{n_trials}: "
                  f"alpha={alpha:.2f}, cold_start={cst}, mmr_lambda={mmr:.1f}")

            engine = HybridRecommender(alpha=alpha, cold_start_threshold=cst, mmr_lambda=mmr)
            engine.set_models(cf_model, cb_model)

            # Quick evaluation on a few users
            sample_users = val_df["user_id"].unique()[:30]
            preds = []
            for uid in sample_users:
                try:
                    recs = engine.recommend(int(uid), movies_df, ratings_df, n=10, diversity=False)
                    for _, rec_row in recs.iterrows():
                        mid = int(rec_row["movie_id"])
                        # check if actual rating exists
                        actual = val_df[
                            (val_df["user_id"] == uid) & (val_df["movie_id"] == mid)
                        ]
                        if not actual.empty:
                            preds.append(type("P", (), {
                                "uid": str(uid), "iid": str(mid),
                                "r_ui": float(actual.iloc[0]["rating"]),
                                "est":  float(rec_row["hybrid_score"]) * 5
                            })())
                except Exception:
                    pass

            if preds:
                p10, r10 = precision_recall_at_k(preds, k=10)
            else:
                p10 = 0.0

            t0 = time.time()
            trial_record = {
                "trial"    : trial,
                "params"   : {"alpha": round(alpha, 2), "cold_start": cst, "mmr_lambda": mmr},
                "p10"      : round(p10, 4),
                "duration" : round(time.time() - t0, 1),
            }
            self.hybrid_trials.append(trial_record)

            status = f"{Fore.GREEN}✓ NEW BEST" if p10 > best_p10 else ""
            print(f"  → Precision@10={p10:.4f}  {status}")

            if p10 > best_p10:
                best_p10 = p10
                self.best_hybrid_params = {"alpha": alpha, "cold_start": cst, "mmr_lambda": mmr}

        print(f"\n{Fore.GREEN}Best Hybrid config found:")
        for k, v in self.best_hybrid_params.items():
            print(f"  {k}: {v}")
        return self.best_hybrid_params

    # ── Reporting ─────────────────────────────────────────────────────────────

    def print_summary(self, top_n: int = 5) -> None:
        """Print a table of the top-N trial configurations."""
        if self.cf_trials:
            sorted_cf = sorted(self.cf_trials, key=lambda x: x["rmse"])[:top_n]
            rows = [[t["trial"],
                     t["params"]["n_factors"], f"{t['params']['lr']:.4f}",
                     f"{t['params']['reg']:.4f}", t["params"]["n_epochs"],
                     t["rmse"], t["mae"], f"{t['duration']}s"]
                    for t in sorted_cf]
            headers = ["Trial", "Factors", "LR", "Reg", "Epochs", "RMSE", "MAE", "Time"]
            print(f"\n{Fore.CYAN}Top-{top_n} CF Configurations:")
            print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    def save_results(self, filename: str = "tuning_results.json") -> str:
        """Save all trial results to a JSON file for reproducibility."""
        path = OUTPUTS_DIR / filename
        results = {
            "cf_trials"          : self.cf_trials,
            "hybrid_trials"      : self.hybrid_trials,
            "best_cf_params"     : self.best_cf_params,
            "best_hybrid_params" : self.best_hybrid_params,
        }
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"{Fore.GREEN}[Tuner] Results saved → {path}")
        return str(path)

    def get_best_params(self) -> dict:
        """Return the best found parameters."""
        return {
            "cf"    : self.best_cf_params,
            "hybrid": self.best_hybrid_params,
        }


if __name__ == "__main__":
    from data_loader import load_ratings, load_movies
    from content_based import ContentBasedModel

    ratings = load_ratings()
    movies  = load_movies()

    tuner = HyperparamTuner()
    best  = tuner.tune_cf(ratings, n_trials=5, cv=2)
    tuner.print_summary()
    tuner.save_results()
    print(f"\nBest params: {best}")
