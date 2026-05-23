# -*- coding: utf-8 -*-
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
  • NDCG@K       — Normalised Discounted Cumulative Gain (position-aware)
  • MAP@K        — Mean Average Precision (considers rank of relevant items)
  • Hit Rate     — fraction of users with at least one relevant item in top-K

  ─ Beyond-Accuracy ─
  • Intra-List Diversity (ILD) — how different are the recommended movies?
  • Catalogue Coverage — fraction of movies ever recommended
  • Novelty — average inverse popularity of recommended items

Visualisations:
  • Rating distribution histogram
  • RMSE / MAE comparison bar chart (SGD vs ALS)
  • Precision & Recall @ K curve
  • Genre distribution of recommendations vs. ground truth
  • Model comparison table
  • Convergence plot (training RMSE over epochs)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from colorama import Fore, init
from sklearn.metrics.pairwise import cosine_similarity

init(autoreset=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOT_DIR     = PROJECT_ROOT / "outputs" / "plots"
try:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

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
    user_preds = {}
    for p in predictions:
        user_preds.setdefault(p.uid, []).append(p)

    precisions, recalls = [], []
    for uid, preds in user_preds.items():
        preds_sorted = sorted(preds, key=lambda x: x.est, reverse=True)
        top_k        = preds_sorted[:k]

        n_rel_in_k   = sum(1 for p in top_k    if p.r_ui >= threshold)
        n_rel_total  = sum(1 for p in preds_sorted if p.r_ui >= threshold)

        precisions.append(n_rel_in_k / k)
        recalls.append(n_rel_in_k / n_rel_total if n_rel_total > 0 else 0)

    return float(np.mean(precisions)), float(np.mean(recalls))


def f1_at_k(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ndcg_at_k(predictions, k: int = 10, threshold: float = 3.5) -> float:
    """
    Normalised Discounted Cumulative Gain @ K.

    NDCG captures not just WHAT is recommended but WHERE it's ranked.
    A relevant item at position 1 is worth more than one at position 10.

    DCG  = Σ (2^rel_i - 1) / log2(i + 1)
    NDCG = DCG / ideal_DCG
    """
    user_preds = {}
    for p in predictions:
        user_preds.setdefault(p.uid, []).append(p)

    ndcgs = []
    for uid, preds in user_preds.items():
        preds_sorted = sorted(preds, key=lambda x: x.est, reverse=True)[:k]

        # Binary relevance: 1 if actual rating >= threshold, 0 otherwise
        rels = [1.0 if p.r_ui >= threshold else 0.0 for p in preds_sorted]

        # DCG
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(rels))

        # Ideal DCG (sort by actual relevance)
        ideal_rels = sorted(rels, reverse=True)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcgs))


def map_at_k(predictions, k: int = 10, threshold: float = 3.5) -> float:
    """
    Mean Average Precision @ K.

    AP for a single user is the average of precision values computed
    at each position where a relevant item is found. MAP averages
    AP over all users.
    """
    user_preds = {}
    for p in predictions:
        user_preds.setdefault(p.uid, []).append(p)

    aps = []
    for uid, preds in user_preds.items():
        preds_sorted = sorted(preds, key=lambda x: x.est, reverse=True)[:k]
        hits = 0
        ap   = 0.0
        for i, p in enumerate(preds_sorted):
            if p.r_ui >= threshold:
                hits += 1
                ap   += hits / (i + 1)

        n_relevant = sum(1 for p in preds_sorted if p.r_ui >= threshold)
        aps.append(ap / n_relevant if n_relevant > 0 else 0.0)

    return float(np.mean(aps))


def hit_rate_at_k(predictions, k: int = 10, threshold: float = 3.5) -> float:
    """
    Hit Rate @ K — fraction of users who get at least one relevant item in top-K.
    """
    user_preds = {}
    for p in predictions:
        user_preds.setdefault(p.uid, []).append(p)

    hits = 0
    for uid, preds in user_preds.items():
        top_k = sorted(preds, key=lambda x: x.est, reverse=True)[:k]
        if any(p.r_ui >= threshold for p in top_k):
            hits += 1

    return float(hits / len(user_preds)) if user_preds else 0.0


# ── 3. Beyond-Accuracy Metrics ────────────────────────────────────────────────

def intra_list_diversity(recs_df: pd.DataFrame, cb_model) -> float:
    """
    Compute Intra-List Diversity (ILD) for a set of recommendations.
    ILD = 1 - average cosine similarity between all pairs in the list.
    High ILD → diverse recommendations (different genres).
    """
    indices = [cb_model.movie_index[mid]
               for mid in recs_df["movie_id"]
               if mid in cb_model.movie_index]
    if len(indices) < 2:
        return 0.0

    vecs  = cb_model.tfidf_matrix[indices].toarray()
    sims  = cosine_similarity(vecs, vecs)
    n     = len(indices)
    total = sum(sims[i][j] for i in range(n) for j in range(i + 1, n))
    pairs = n * (n - 1) / 2
    return round(1.0 - (total / pairs), 4)


def catalogue_coverage(cf_model, movies_df: pd.DataFrame,
                        ratings_df: pd.DataFrame,
                        n_users: int = 100, k: int = 10) -> float:
    """
    Fraction of the catalogue that appears in at least one user's top-K.
    Higher coverage → the system recommends a wider variety of movies.
    """
    sample_users = ratings_df["user_id"].unique()[:n_users]
    recommended  = set()

    for uid in sample_users:
        try:
            recs = cf_model.get_top_n_recommendations(int(uid), movies_df, ratings_df, n=k)
            recommended.update(recs["movie_id"].tolist())
        except Exception:
            pass

    total = len(movies_df)
    return round(len(recommended) / total, 4) if total > 0 else 0.0


def novelty_score(cf_model, movies_df: pd.DataFrame,
                   ratings_df: pd.DataFrame,
                   n_users: int = 100, k: int = 10) -> float:
    """
    Average novelty of recommendations.

    Novelty = average -log2(popularity) of recommended items.
    Higher novelty → less popular (and therefore more surprising) recommendations.
    """
    movie_pop = ratings_df.groupby("movie_id")["rating"].count()
    total_users = ratings_df["user_id"].nunique()
    sample_users = ratings_df["user_id"].unique()[:n_users]

    novelties = []
    for uid in sample_users:
        try:
            recs = cf_model.get_top_n_recommendations(int(uid), movies_df, ratings_df, n=k)
            for mid in recs["movie_id"]:
                pop = movie_pop.get(mid, 1) / total_users
                novelties.append(-np.log2(max(pop, 1e-10)))
        except Exception:
            pass

    return round(float(np.mean(novelties)), 4) if novelties else 0.0


# ── 4. Full Evaluation Report ─────────────────────────────────────────────────

def full_evaluation_report(cf_model, cb_model, hybrid_engine,
                            ratings_df: pd.DataFrame, movies_df: pd.DataFrame,
                            n_users: int = 10, als_model=None) -> dict:
    """
    Run a comprehensive evaluation and print a formatted report.

    Returns dict with all computed metrics.
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
        ndcg10   = ndcg_at_k(cf_model.predictions, k=10)
        map10    = map_at_k(cf_model.predictions, k=10)
        hr10     = hit_rate_at_k(cf_model.predictions, k=10)

        report.update({
            "cf_rmse": round(rmse, 4),
            "cf_mae" : round(mae,  4),
            "precision@10": round(p10,  4),
            "recall@10"   : round(r10,  4),
            "f1@10"       : round(f1_10, 4),
            "ndcg@10"     : round(ndcg10, 4),
            "map@10"      : round(map10, 4),
            "hit_rate@10" : round(hr10, 4),
            "precision@20": round(p20,  4),
            "recall@20"   : round(r20,  4),
            "f1@20"       : round(f1_20, 4),
        })
    else:
        report.update({"note": "No held-out predictions stored. Run evaluate_on_testset() first."})

    # ALS metrics (if available)
    if als_model and hasattr(als_model, "predictions") and als_model.predictions:
        als_rmse = compute_rmse(als_model.predictions)
        als_mae  = compute_mae(als_model.predictions)
        als_ndcg = ndcg_at_k(als_model.predictions, k=10)
        report.update({
            "als_rmse": round(als_rmse, 4),
            "als_mae" : round(als_mae,  4),
            "als_ndcg@10": round(als_ndcg, 4),
        })

    # ── Diversity for sample users ─────────────────────────────────────────
    sample_users = ratings_df["user_id"].unique()[:n_users]
    ild_scores   = []
    for uid in sample_users:
        try:
            recs = hybrid_engine.recommend(uid, movies_df, ratings_df, n=10, diversity=False)
            ild  = intra_list_diversity(recs, cb_model)
            ild_scores.append(ild)
        except Exception:
            pass

    report["avg_ild"] = round(float(np.mean(ild_scores)), 4) if ild_scores else None

    # ── Coverage & Novelty ─────────────────────────────────────────────────
    report["coverage"] = catalogue_coverage(cf_model, movies_df, ratings_df, n_users=50)
    report["novelty"]  = novelty_score(cf_model, movies_df, ratings_df, n_users=50)

    # ── Print ──────────────────────────────────────────────────────────────
    print(f"\n{Fore.CYAN}{'='*62}")
    print(f"{Fore.CYAN}  EVALUATION REPORT")
    print(f"{Fore.CYAN}{'='*62}")
    for k, v in report.items():
        print(f"  {Fore.GREEN}{k:<20}{Fore.WHITE}{v}")
    print(f"{Fore.CYAN}{'='*62}\n")
    return report


# ── 5. Model Comparison ──────────────────────────────────────────────────────

def model_comparison_table(cf_model, als_model=None, hybrid_engine=None,
                            ratings_df=None, movies_df=None) -> str:
    """
    Print a side-by-side comparison table of all models.
    Returns the table as a string.
    """
    from tabulate import tabulate

    rows = []
    headers = ["Metric", "SGD", "ALS", "Hybrid"]

    # SGD metrics
    sgd_vals = {}
    if hasattr(cf_model, "predictions") and cf_model.predictions:
        sgd_vals["RMSE"] = round(compute_rmse(cf_model.predictions), 4)
        sgd_vals["MAE"]  = round(compute_mae(cf_model.predictions),  4)
        p, r = precision_recall_at_k(cf_model.predictions, k=10)
        sgd_vals["P@10"]    = round(p, 4)
        sgd_vals["R@10"]    = round(r, 4)
        sgd_vals["NDCG@10"] = round(ndcg_at_k(cf_model.predictions, k=10), 4)
        sgd_vals["MAP@10"]  = round(map_at_k(cf_model.predictions, k=10), 4)

    # ALS metrics
    als_vals = {}
    if als_model and hasattr(als_model, "predictions") and als_model.predictions:
        als_vals["RMSE"] = round(compute_rmse(als_model.predictions), 4)
        als_vals["MAE"]  = round(compute_mae(als_model.predictions),  4)
        p, r = precision_recall_at_k(als_model.predictions, k=10)
        als_vals["P@10"]    = round(p, 4)
        als_vals["R@10"]    = round(r, 4)
        als_vals["NDCG@10"] = round(ndcg_at_k(als_model.predictions, k=10), 4)
        als_vals["MAP@10"]  = round(map_at_k(als_model.predictions, k=10), 4)

    for metric in ["RMSE", "MAE", "P@10", "R@10", "NDCG@10", "MAP@10"]:
        rows.append([
            metric,
            sgd_vals.get(metric, "-"),
            als_vals.get(metric, "-"),
            "-",  # hybrid doesn't have simple predictions in same format
        ])

    table = tabulate(rows, headers=headers, tablefmt="rounded_outline", floatfmt=".4f")
    print(f"\n{Fore.CYAN}Model Comparison:")
    print(table)
    return table


# ── 6. Visualisations ─────────────────────────────────────────────────────────

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
    plt.close()


def plot_model_comparison(metrics: dict, save: bool = True) -> None:
    """Grouped bar chart comparing RMSE and MAE across models."""
    models = ["SGD"]
    rmses  = [metrics.get("cf_rmse", 0)]
    maes   = [metrics.get("cf_mae",  0)]

    if "als_rmse" in metrics:
        models.append("ALS")
        rmses.append(metrics["als_rmse"])
        maes.append(metrics["als_mae"])

    x      = np.arange(len(models))
    width  = 0.30
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
    plt.close()


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
    plt.close()


def plot_convergence(model, title: str = "SGD", save: bool = True) -> None:
    """Plot training RMSE over epochs/iterations."""
    if not hasattr(model, "train_rmse_history") or not model.train_rmse_history:
        return

    history = model.train_rmse_history
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(history)+1), history, "o-", color=PALETTE[0], lw=2, ms=5)
    ax.set_title(f"Training Convergence — {title}")
    ax.set_xlabel("Epoch" if "SGD" in title else "Iteration")
    ax.set_ylabel("Training RMSE")
    ax.grid(alpha=0.4)
    plt.tight_layout()
    if save:
        fname = f"convergence_{title.lower().replace(' ', '_')}.png"
        plt.savefig(PLOT_DIR / fname, dpi=150)
        print(f"{Fore.GREEN}[Plot] Saved → {fname}")
    plt.close()


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
    plt.close()


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
    plt.close()
