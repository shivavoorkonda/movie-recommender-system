"""
evaluator.py
============
Evaluation and visualisation utilities for the recommender system.

Metrics covered:
  ─ Rating Prediction Quality ─
  • RMSE (Root Mean Squared Error)  — penalises large errors more
  • MAE  (Mean Absolute Error)      — average absolute prediction error

  ─ Ranking Quality ─
  • Precision@K  — of the K recommended movies, how many did the user actually like?
  • Recall@K     — of all movies the user liked, how many made it into top-K?
  • F1@K         — harmonic mean of Precision@K and Recall@K

  ─ Diversity ─
  • Intra-List Diversity (ILD) — how different are the recommended movies from each other?

Visualisations:
  • Rating distribution histogram
  • RMSE / MAE comparison bar chart
  • Precision & Recall @ K curve
  • Genre distribution of recommendations vs. ground truth
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — saves files without showing windows
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from colorama import Fore, init
from sklearn.metrics.pairwise import cosine_similarity

init(autoreset=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOT_DIR     = PROJECT_ROOT / "outputs" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Plot style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor" : "#0d1117",
    "axes.facecolor"   : "#161b22",
    "axes.edgecolor"   : "#30363d",
    "axes.labelcolor"  : "#c9d1d9",
    "xtick.color"      : "#c9d1d9",
    "ytick.color"      : "#c9d1d9",
    "text.color"       : "#c9d1d9",
    "grid.color"       : "#21262d",
    "grid.linestyle"   : "--",
    "font.family"      : "DejaVu Sans",
    "axes.titlecolor"  : "#58a6ff",
    "axes.titlesize"   : 14,
    "axes.labelsize"   : 12,
})
PALETTE = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657"]


# ── 1. Rating Prediction Metrics ──────────────────────────────────────────────

def compute_rmse(predictions) -> float:
    """Compute RMSE from Surprise prediction objects."""
    errors = [(p.r_ui - p.est) ** 2 for p in predictions]
    return float(np.sqrt(np.mean(errors)))


def compute_mae(predictions) -> float:
    """Compute MAE from Surprise prediction objects."""
    errors = [abs(p.r_ui - p.est) for p in predictions]
    return float(np.mean(errors))


# ── 2. Ranking Metrics ────────────────────────────────────────────────────────

def precision_recall_at_k(predictions, k: int = 10,
                           threshold: float = 3.5) -> tuple[float, float]:
    """
    Compute Precision@K and Recall@K averaged over all users.

    Parameters
    ----------
    predictions : list of Surprise Prediction objects
    k           : int   — recommendation list length
    threshold   : float — minimum rating to consider a movie "relevant"

    Returns
    -------
    (precision_at_k, recall_at_k) — both floats
    """
    # Group predictions by user
    user_preds = {}
    for p in predictions:
        user_preds.setdefault(p.uid, []).append(p)

    precisions, recalls = [], []
    for uid, preds in user_preds.items():
        # Sort by estimated rating (what we'd recommend)
        preds_sorted = sorted(preds, key=lambda x: x.est, reverse=True)
        top_k        = preds_sorted[:k]

        # Number of relevant items in top-K
        n_rel_in_k   = sum(1 for p in top_k    if p.r_ui >= threshold)
        # Total relevant items for this user
        n_rel_total  = sum(1 for p in preds_sorted if p.r_ui >= threshold)

        precisions.append(n_rel_in_k / k)
        recalls.append(n_rel_in_k / n_rel_total if n_rel_total > 0 else 0)

    return float(np.mean(precisions)), float(np.mean(recalls))


def f1_at_k(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── 3. Diversity ──────────────────────────────────────────────────────────────

def intra_list_diversity(recs_df: pd.DataFrame, cb_model) -> float:
    """
    Compute Intra-List Diversity (ILD) for a set of recommendations.

    ILD = 1 - average cosine similarity between all pairs in the list.
    High ILD → diverse recommendations (different genres).

    Parameters
    ----------
    recs_df  : pd.DataFrame with 'movie_id' column
    cb_model : fitted ContentBasedModel (for TF-IDF vectors)
    """
    indices = [cb_model.movie_index[mid]
               for mid in recs_df["movie_id"]
               if mid in cb_model.movie_index]
    if len(indices) < 2:
        return 0.0

    vecs  = cb_model.tfidf_matrix[indices].toarray()
    sims  = cosine_similarity(vecs, vecs)
    # Upper triangle (excluding diagonal)
    n     = len(indices)
    total = sum(sims[i][j] for i in range(n) for j in range(i + 1, n))
    pairs = n * (n - 1) / 2
    return round(1.0 - (total / pairs), 4)


# ── 4. Full Evaluation Report ─────────────────────────────────────────────────

def full_evaluation_report(cf_model, cb_model, hybrid_engine,
                            ratings_df: pd.DataFrame, movies_df: pd.DataFrame,
                            n_users: int = 10) -> dict:
    """
    Run a comprehensive evaluation and print a formatted report.

    Parameters
    ----------
    cf_model      : fitted CollaborativeFilteringModel
    cb_model      : fitted ContentBasedModel
    hybrid_engine : fitted HybridRecommender
    ratings_df    : pd.DataFrame
    movies_df     : pd.DataFrame
    n_users       : int — number of sample users to evaluate ranking metrics

    Returns
    -------
    dict with all computed metrics
    """
    report = {}

    # ── Rating prediction metrics (from stored predictions) ────────────────
    if hasattr(cf_model, "predictions") and cf_model.predictions:
        rmse = compute_rmse(cf_model.predictions)
        mae  = compute_mae(cf_model.predictions)
        p10, r10 = precision_recall_at_k(cf_model.predictions, k=10)
        f1_10    = f1_at_k(p10, r10)
        p20, r20 = precision_recall_at_k(cf_model.predictions, k=20)
        f1_20    = f1_at_k(p20, r20)
        report.update({
            "cf_rmse": round(rmse, 4),
            "cf_mae" : round(mae,  4),
            "precision@10": round(p10,  4),
            "recall@10"   : round(r10,  4),
            "f1@10"       : round(f1_10, 4),
            "precision@20": round(p20,  4),
            "recall@20"   : round(r20,  4),
            "f1@20"       : round(f1_20, 4),
        })
    else:
        report.update({"note": "No held-out predictions stored. Run evaluate_on_testset() first."})

    # ── Diversity for sample users ─────────────────────────────────────────
    sample_users = ratings_df["user_id"].unique()[:n_users]
    ild_scores   = []
    for uid in sample_users:
        try:
            recs = hybrid_engine.recommend(uid, movies_df, ratings_df, n=10)
            ild  = intra_list_diversity(recs, cb_model)
            ild_scores.append(ild)
        except Exception:
            pass

    report["avg_ild"] = round(float(np.mean(ild_scores)), 4) if ild_scores else None

    # ── Print ──────────────────────────────────────────────────────────────
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}  EVALUATION REPORT")
    print(f"{Fore.CYAN}{'='*60}")
    for k, v in report.items():
        print(f"  {Fore.GREEN}{k:<20}{Fore.WHITE}{v}")
    print(f"{Fore.CYAN}{'='*60}\n")
    return report


# ── 5. Visualisations ─────────────────────────────────────────────────────────

def plot_rating_distribution(ratings_df: pd.DataFrame, save: bool = True) -> None:
    """Bar chart of rating frequencies (1-5 stars)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts  = ratings_df["rating"].value_counts().sort_index()
    bars    = ax.bar(counts.index.astype(str), counts.values,
                     color=PALETTE[:len(counts)], edgecolor="#0d1117", linewidth=0.8)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 800,
                f"{bar.get_height():,}", ha="center", va="bottom", fontsize=10, color="#c9d1d9")
    ax.set_title("Rating Distribution — MovieLens 100K")
    ax.set_xlabel("Rating (Stars)")
    ax.set_ylabel("Number of Ratings")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    if save:
        plt.savefig(PLOT_DIR / "rating_distribution.png", dpi=150)
        print(f"{Fore.GREEN}[Plot] Saved → rating_distribution.png")


def plot_model_comparison(metrics: dict, save: bool = True) -> None:
    """
    Grouped bar chart comparing RMSE and MAE.

    metrics : dict with keys like 'cf_rmse', 'cf_mae'
    """
    models = ["SVD (CF)"]
    rmses  = [metrics.get("cf_rmse", 0)]
    maes   = [metrics.get("cf_mae",  0)]

    x      = np.arange(len(models))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width / 2, rmses, width, label="RMSE", color=PALETTE[0], alpha=0.9)
    b2 = ax.bar(x + width / 2, maes,  width, label="MAE",  color=PALETTE[2], alpha=0.9)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_title("Model Error Metrics (Lower is Better)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Error")
    ax.legend(facecolor="#161b22", edgecolor="#30363d")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    if save:
        plt.savefig(PLOT_DIR / "model_comparison.png", dpi=150)
        print(f"{Fore.GREEN}[Plot] Saved → model_comparison.png")


def plot_precision_recall_curve(predictions, ks=None, save: bool = True) -> None:
    """Plot Precision@K and Recall@K for a range of K values."""
    if ks is None:
        ks = [1, 5, 10, 15, 20, 25, 30]
    prec_vals, rec_vals = [], []
    for k in ks:
        p, r = precision_recall_at_k(predictions, k=k)
        prec_vals.append(p)
        rec_vals.append(r)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, prec_vals, "o-", color=PALETTE[0], label="Precision@K", lw=2, ms=6)
    ax.plot(ks, rec_vals,  "s-", color=PALETTE[2], label="Recall@K",    lw=2, ms=6)
    ax.set_title("Precision & Recall @ K")
    ax.set_xlabel("K (Top-K Recommendations)")
    ax.set_ylabel("Score")
    ax.legend(facecolor="#161b22", edgecolor="#30363d")
    ax.grid(alpha=0.4)
    plt.tight_layout()
    if save:
        plt.savefig(PLOT_DIR / "precision_recall_at_k.png", dpi=150)
        print(f"{Fore.GREEN}[Plot] Saved → precision_recall_at_k.png")


def plot_genre_distribution(recs_df: pd.DataFrame, title: str = "Recommendations",
                             save: bool = True, filename: str = "genre_dist.png") -> None:
    """Horizontal bar chart of genre frequencies in a recommendations list."""
    genre_counts = {}
    for genres in recs_df["genres"].dropna():
        for g in genres.split("|"):
            genre_counts[g] = genre_counts.get(g, 0) + 1

    if not genre_counts:
        print(f"{Fore.YELLOW}[Plot] No genre data to plot.")
        return

    genre_df = pd.Series(genre_counts).sort_values(ascending=True)
    fig, ax  = plt.subplots(figsize=(9, max(4, len(genre_df) * 0.4)))
    colours  = [PALETTE[i % len(PALETTE)] for i in range(len(genre_df))]
    genre_df.plot(kind="barh", ax=ax, color=colours, edgecolor="#0d1117")
    ax.set_title(f"Genre Distribution — {title}")
    ax.set_xlabel("Count")
    ax.grid(axis="x", alpha=0.4)
    plt.tight_layout()
    if save:
        plt.savefig(PLOT_DIR / filename, dpi=150)
        print(f"{Fore.GREEN}[Plot] Saved → {filename}")


def plot_user_rating_activity(ratings_df: pd.DataFrame, save: bool = True) -> None:
    """Histogram of how many movies each user has rated."""
    user_counts = ratings_df.groupby("user_id")["rating"].count()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(user_counts, bins=40, color=PALETTE[3], edgecolor="#0d1117", alpha=0.9)
    ax.set_title("User Rating Activity Distribution")
    ax.set_xlabel("Number of Movies Rated per User")
    ax.set_ylabel("Number of Users")
    ax.axvline(user_counts.mean(), color=PALETTE[2], linestyle="--",
               label=f"Mean: {user_counts.mean():.0f}")
    ax.legend(facecolor="#161b22", edgecolor="#30363d")
    ax.grid(alpha=0.4)
    plt.tight_layout()
    if save:
        plt.savefig(PLOT_DIR / "user_rating_activity.png", dpi=150)
        print(f"{Fore.GREEN}[Plot] Saved → user_rating_activity.png")
