# -*- coding: utf-8 -*-
"""
analysis.py
===========
Fairness, bias, and uncertainty analysis for the Movie Recommender System.

Why this matters:
  Recommender systems can silently amplify inequalities — popular movies get
  recommended more, making them even more popular (popularity bias). Some user
  demographics may get systematically worse predictions (demographic bias).
  
  This module gives you the tools to detect and understand these issues.

Functions
---------
  compute_gini_coefficient     -- measure inequality in recommendation frequency
  popularity_bias_analysis     -- how concentrated are recommendations?
  demographic_fairness         -- are predictions equally accurate across genders/ages?
  genre_coverage_analysis      -- what fraction of genres appear in recommendations?
  estimate_prediction_uncertainty -- bootstrap confidence intervals for predictions
  calibration_analysis         -- are predicted ratings actually accurate on average?
  full_bias_report             -- run everything at once, print + return results
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from colorama import Fore, Style, init
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

init(autoreset=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOT_DIR     = PROJECT_ROOT / "outputs" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Same dark theme as evaluator.py — keep everything visually consistent
plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor":  "#30363d", "axes.labelcolor": "#c9d1d9",
    "xtick.color":     "#c9d1d9", "ytick.color":     "#c9d1d9",
    "text.color":      "#c9d1d9", "grid.color":      "#21262d",
    "grid.linestyle":  "--",      "font.family":     "DejaVu Sans",
    "axes.titlecolor": "#58a6ff", "axes.titlesize":  13,
    "axes.labelsize":  11,
})
PALETTE = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657"]


# ── 1. Gini Coefficient ───────────────────────────────────────────────────────

def compute_gini_coefficient(frequencies: np.ndarray) -> float:
    """
    Compute the Gini coefficient of a frequency distribution.

    Gini = 0 → perfect equality (all movies recommended equally often)
    Gini = 1 → perfect inequality (one movie gets all recommendations)

    This is the same metric economists use to measure income inequality —
    we're just applying it to recommendation frequency instead of wealth.
    """
    if len(frequencies) == 0:
        return 0.0
    freq = np.sort(frequencies.astype(float))
    n    = len(freq)
    # Standard Gini formula
    gini = (2 * np.sum(np.arange(1, n + 1) * freq) - (n + 1) * np.sum(freq)) / (n * np.sum(freq))
    return round(float(gini), 4)


# ── 2. Popularity Bias ────────────────────────────────────────────────────────

def popularity_bias_analysis(cf_model, movies_df: pd.DataFrame,
                              ratings_df: pd.DataFrame,
                              n_users: int = 100, n_recs: int = 10) -> dict:
    """
    Measure how concentrated recommendations are on popular movies.

    A system with high popularity bias always recommends the same few films
    that everyone has already seen — this hurts discovery and diversity.

    Returns dict with:
      - gini_coefficient : inequality in recommendation frequency
      - catalogue_coverage: % of movies that appear in at least one recommendation
      - long_tail_ratio  : % of recs from movies with < median popularity
      - top_10_domination: % of recs that come from the top-10 most popular movies
    """
    print(f"{Fore.CYAN}[Analysis] Running popularity bias analysis "
          f"({n_users} users, {n_recs} recs each) …")

    sample_users = ratings_df["user_id"].unique()[:n_users]
    rec_counts   = {}   # movie_id → how many times it appeared in recommendations

    for uid in sample_users:
        try:
            recs = cf_model.get_top_n_recommendations(
                int(uid), movies_df, ratings_df, n=n_recs)
            for mid in recs["movie_id"].tolist():
                rec_counts[mid] = rec_counts.get(mid, 0) + 1
        except Exception:
            pass

    if not rec_counts:
        return {"error": "No recommendations generated"}

    freq_array      = np.array(list(rec_counts.values()))
    gini            = compute_gini_coefficient(freq_array)
    catalogue_cover = round(len(rec_counts) / len(movies_df) * 100, 2)

    # Long-tail: movies with fewer ratings than median are "long-tail"
    movie_pop   = ratings_df.groupby("movie_id")["rating"].count()
    median_pop  = movie_pop.median()
    long_tail_ids = set(movie_pop[movie_pop < median_pop].index.tolist())
    long_tail_recs = sum(1 for mid in rec_counts if mid in long_tail_ids)
    total_recs     = sum(rec_counts.values())
    long_tail_ratio = round(long_tail_recs / max(total_recs, 1) * 100, 2)

    # Top-10 domination: what % of all recs are just the top-10 movies?
    top10_ids    = set(sorted(rec_counts, key=rec_counts.get, reverse=True)[:10])
    top10_recs   = sum(rec_counts[mid] for mid in top10_ids)
    top10_dom    = round(top10_recs / max(total_recs, 1) * 100, 2)

    results = {
        "gini_coefficient":   gini,
        "catalogue_coverage": catalogue_cover,
        "long_tail_ratio":    long_tail_ratio,
        "top10_domination":   top10_dom,
        "total_unique_recs":  len(rec_counts),
        "total_recs":         total_recs,
        "rec_counts":         rec_counts,
    }

    print(f"  Gini coefficient  : {gini:.4f}  (0=equal, 1=concentrated)")
    print(f"  Catalogue coverage: {catalogue_cover:.1f}%")
    print(f"  Long-tail ratio   : {long_tail_ratio:.1f}%  (higher = more diverse)")
    print(f"  Top-10 domination : {top10_dom:.1f}%  (lower = less bias)")

    return results


# ── 3. Demographic Fairness ───────────────────────────────────────────────────

def demographic_fairness(cf_model, ratings_df: pd.DataFrame,
                          users_df: pd.DataFrame, test_size: float = 0.2) -> dict:
    """
    Compute prediction RMSE separately for different demographic groups.

    A fair model should have similar RMSE across groups. Large differences
    indicate that the model is more accurate for some users than others.

    Groups analysed:
      - Gender: Male vs Female
      - Age: <25, 25-35, 35-50, 50+
    """
    print(f"{Fore.CYAN}[Analysis] Running demographic fairness analysis …")

    # We use the stored test predictions if available, otherwise we do a quick eval
    if not hasattr(cf_model, "predictions") or not cf_model.predictions:
        print(f"{Fore.YELLOW}[Analysis] No stored predictions. "
              "Run evaluate_on_testset() on the model first for best results.")
        return {"error": "No predictions available"}

    # Build a lookup: (user_id, movie_id) → (actual, predicted)
    pred_lookup = {(int(p.uid), int(p.iid)): (p.r_ui, p.est) for p in cf_model.predictions}

    # Merge with user demographics
    user_info = users_df.set_index("user_id")[["gender", "age"]].to_dict("index")

    group_errors = {
        "male": [], "female": [],
        "age_lt25": [], "age_25_35": [], "age_35_50": [], "age_50plus": [],
    }

    for (uid, mid), (actual, pred) in pred_lookup.items():
        info = user_info.get(uid)
        if info is None:
            continue
        err_sq = (actual - pred) ** 2

        # Gender groups
        if info["gender"] == "M":
            group_errors["male"].append(err_sq)
        elif info["gender"] == "F":
            group_errors["female"].append(err_sq)

        # Age groups
        age = info["age"]
        if age < 25:
            group_errors["age_lt25"].append(err_sq)
        elif age < 35:
            group_errors["age_25_35"].append(err_sq)
        elif age < 50:
            group_errors["age_35_50"].append(err_sq)
        else:
            group_errors["age_50plus"].append(err_sq)

    results = {}
    for group, errors in group_errors.items():
        if errors:
            results[group] = {
                "rmse": round(float(np.sqrt(np.mean(errors))), 4),
                "n"   : len(errors),
            }
        else:
            results[group] = {"rmse": None, "n": 0}

    print(f"\n  {'Group':<15} {'RMSE':<10} {'N samples'}")
    print(f"  {'-'*35}")
    for group, vals in results.items():
        rmse_str = f"{vals['rmse']:.4f}" if vals["rmse"] else "N/A"
        print(f"  {group:<15} {rmse_str:<10} {vals['n']}")

    return results


# ── 4. Genre Coverage ─────────────────────────────────────────────────────────

def genre_coverage_analysis(hybrid_engine, movies_df: pd.DataFrame,
                             ratings_df: pd.DataFrame,
                             n_users: int = 50, n_recs: int = 10) -> dict:
    """
    Measure how many genres appear in recommendations vs. the full catalogue.

    Low coverage means the system is stuck recommending a few dominant genres
    (usually Drama + Action) and ignoring Film-Noir, Documentary, etc.
    """
    print(f"{Fore.CYAN}[Analysis] Running genre coverage analysis …")

    # All genres present in dataset
    all_genres = set()
    for genres in movies_df["genres"].dropna():
        all_genres.update(genres.split("|"))
    all_genres.discard("unknown")

    # Genres in recommendations
    rec_genre_counts = {}
    sample_users = ratings_df["user_id"].unique()[:n_users]

    for uid in sample_users:
        try:
            recs = hybrid_engine.recommend(int(uid), movies_df, ratings_df,
                                           n=n_recs, diversity=True)
            for genres_str in recs["genres"].dropna():
                for g in genres_str.split("|"):
                    if g != "unknown":
                        rec_genre_counts[g] = rec_genre_counts.get(g, 0) + 1
        except Exception:
            pass

    covered_genres   = set(rec_genre_counts.keys())
    coverage_ratio   = round(len(covered_genres) / max(len(all_genres), 1), 4)
    uncovered_genres = all_genres - covered_genres

    results = {
        "total_genres"   : len(all_genres),
        "covered_genres" : len(covered_genres),
        "coverage_ratio" : coverage_ratio,
        "uncovered"      : sorted(uncovered_genres),
        "genre_counts"   : rec_genre_counts,
    }

    print(f"  Genres in catalogue  : {len(all_genres)}")
    print(f"  Genres in recs       : {len(covered_genres)}")
    print(f"  Coverage ratio       : {coverage_ratio:.1%}")
    if uncovered_genres:
        print(f"  Missing genres       : {', '.join(sorted(uncovered_genres))}")

    return results


# ── 5. Prediction Uncertainty ─────────────────────────────────────────────────

def estimate_prediction_uncertainty(cf_model, ratings_df: pd.DataFrame,
                                     user_id: int, movie_id: int,
                                     n_bootstrap: int = 15) -> dict:
    """
    Bootstrap confidence interval for a single prediction.

    We retrain a lightweight model n_bootstrap times on resampled data
    and collect the spread of predictions. High spread = low confidence.

    This is computationally expensive — only use for specific user-movie
    pairs where you need to know confidence, not for bulk recommendations.
    """
    from collaborative_filtering import CollaborativeFilteringModel

    preds = []
    for i in range(n_bootstrap):
        # Resample with replacement (bootstrap)
        boot = ratings_df.sample(frac=1.0, replace=True, random_state=i)
        boot = boot.reset_index(drop=True)

        # Lightweight model for speed — don't need full quality here
        mini = CollaborativeFilteringModel(n_factors=20, n_epochs=5,
                                            lr=0.005, reg=0.02)
        mini.fit(boot)
        pred = mini.predict_rating(user_id, movie_id)
        preds.append(pred)

    preds_arr = np.array(preds)
    return {
        "user_id"  : user_id,
        "movie_id" : movie_id,
        "mean"     : round(float(np.mean(preds_arr)), 3),
        "std"      : round(float(np.std(preds_arr)),  3),
        "ci_lower" : round(float(np.percentile(preds_arr, 5)),  3),
        "ci_upper" : round(float(np.percentile(preds_arr, 95)), 3),
        "n_bootstrap": n_bootstrap,
    }


# ── 6. Calibration Analysis ───────────────────────────────────────────────────

def calibration_analysis(cf_model, ratings_df: pd.DataFrame) -> dict:
    """
    Check if the model's predicted ratings match reality on average.

    A well-calibrated model predicts 4.0 for movies that users actually rate
    4.0 on average. If it always predicts 3.8 when actual is 4.5, it's biased.
    """
    print(f"{Fore.CYAN}[Analysis] Running calibration analysis …")

    if not hasattr(cf_model, "predictions") or not cf_model.predictions:
        return {"error": "No predictions stored. Run evaluate_on_testset() first."}

    # Group by actual rating bucket and compare to predicted
    buckets = {1: [], 2: [], 3: [], 4: [], 5: []}
    for p in cf_model.predictions:
        bucket = int(round(p.r_ui))
        bucket = max(1, min(5, bucket))
        buckets[bucket].append(p.est)

    calibration = {}
    total_err   = 0
    n_buckets   = 0
    for rating, preds in buckets.items():
        if preds:
            mean_pred = float(np.mean(preds))
            calibration[rating] = {
                "actual_rating": rating,
                "mean_predicted": round(mean_pred, 3),
                "calibration_error": round(abs(mean_pred - rating), 3),
                "n": len(preds),
            }
            total_err += abs(mean_pred - rating)
            n_buckets += 1

    avg_calib_error = round(total_err / max(n_buckets, 1), 4)

    print(f"\n  {'Actual':>8} {'Pred Mean':>10} {'Error':>8} {'Count':>8}")
    print(f"  {'-'*38}")
    for r, info in calibration.items():
        print(f"  {r:>8} {info['mean_predicted']:>10.3f} {info['calibration_error']:>8.3f} {info['n']:>8}")
    print(f"\n  Average calibration error: {avg_calib_error:.4f}")

    return {"buckets": calibration, "avg_calibration_error": avg_calib_error}


# ── 7. Full Report ────────────────────────────────────────────────────────────

def full_bias_report(cf_model, hybrid_engine, movies_df: pd.DataFrame,
                      ratings_df: pd.DataFrame, users_df: pd.DataFrame) -> dict:
    """Run all analyses, print a formatted report, return combined dict."""
    print(f"\n{Fore.CYAN}{'='*62}")
    print(f"{Fore.CYAN}  FAIRNESS & BIAS ANALYSIS REPORT")
    print(f"{Fore.CYAN}{'='*62}")

    results = {}

    results["popularity_bias"]    = popularity_bias_analysis(cf_model, movies_df, ratings_df)
    results["demographic_fairness"] = demographic_fairness(cf_model, ratings_df, users_df)
    results["genre_coverage"]     = genre_coverage_analysis(hybrid_engine, movies_df, ratings_df, n_users=30)
    results["calibration"]        = calibration_analysis(cf_model, ratings_df)

    print(f"\n{Fore.GREEN}{'='*62}")
    print(f"{Fore.GREEN}  Summary:")
    print(f"  Gini (popularity bias): "
          f"{results['popularity_bias'].get('gini_coefficient', 'N/A')}")
    print(f"  Catalogue coverage    : "
          f"{results['popularity_bias'].get('catalogue_coverage', 'N/A')}%")
    print(f"  Genre coverage        : "
          f"{results['genre_coverage'].get('coverage_ratio', 'N/A'):.1%}")
    print(f"  Calibration error     : "
          f"{results['calibration'].get('avg_calibration_error', 'N/A')}")
    print(f"{Fore.GREEN}{'='*62}\n")

    return results


# ── 8. Plotting ───────────────────────────────────────────────────────────────

def plot_popularity_distribution(bias_results: dict, save: bool = True) -> None:
    """Bar chart of top-30 most recommended movies."""
    rec_counts = bias_results.get("rec_counts", {})
    if not rec_counts:
        return

    top30 = sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)[:30]
    mids  = [str(x[0]) for x in top30]
    cnts  = [x[1]      for x in top30]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(mids)), cnts, color=PALETTE[0], alpha=0.85, edgecolor="#0d1117")
    ax.set_title(f"Recommendation Frequency (Top 30 Movies)  |  "
                 f"Gini={bias_results.get('gini_coefficient', '?')}")
    ax.set_xlabel("Movie Rank")
    ax.set_ylabel("Times Recommended")
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    if save:
        plt.savefig(PLOT_DIR / "popularity_distribution.png", dpi=150)
        print(f"{Fore.GREEN}[Analysis] Plot saved → popularity_distribution.png")
    plt.close()


def plot_demographic_fairness(fairness_results: dict, save: bool = True) -> None:
    """Grouped bar chart of RMSE by demographic group."""
    groups = {k: v["rmse"] for k, v in fairness_results.items()
              if isinstance(v, dict) and v.get("rmse") is not None}
    if not groups:
        return

    labels = list(groups.keys())
    values = list(groups.values())
    colours = [PALETTE[0] if "male" in l else PALETTE[2] if "female" in l
               else PALETTE[1] for l in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colours, edgecolor="#0d1117", alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Prediction RMSE by Demographic Group  (lower = better + fairer)")
    ax.set_ylabel("RMSE")
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", alpha=0.4)
    plt.xticks(rotation=15)
    plt.tight_layout()
    if save:
        plt.savefig(PLOT_DIR / "demographic_fairness.png", dpi=150)
        print(f"{Fore.GREEN}[Analysis] Plot saved → demographic_fairness.png")
    plt.close()


def plot_calibration_curve(calibration_results: dict, save: bool = True) -> None:
    """Line chart of predicted vs actual ratings per bucket."""
    buckets = calibration_results.get("buckets", {})
    if not buckets:
        return

    actual_r = sorted(buckets.keys())
    mean_pred = [buckets[r]["mean_predicted"] for r in actual_r]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(actual_r, actual_r,  "--", color="#30363d", label="Perfect calibration", lw=2)
    ax.plot(actual_r, mean_pred, "o-", color=PALETTE[0], label="Model predictions",  lw=2, ms=8)
    ax.set_title("Calibration Curve — Predicted vs Actual Rating")
    ax.set_xlabel("Actual Rating (bucket)")
    ax.set_ylabel("Mean Predicted Rating")
    ax.set_xticks(actual_r)
    ax.legend(facecolor="#161b22", edgecolor="#30363d")
    ax.grid(alpha=0.4)
    plt.tight_layout()
    if save:
        plt.savefig(PLOT_DIR / "calibration_curve.png", dpi=150)
        print(f"{Fore.GREEN}[Analysis] Plot saved → calibration_curve.png")
    plt.close()


if __name__ == "__main__":
    from data_loader import load_all
    from collaborative_filtering import CollaborativeFilteringModel
    from content_based import ContentBasedModel
    from hybrid_engine import HybridRecommender

    data    = load_all(verbose=True)
    ratings = data["ratings"]
    movies  = data["movies"]
    users   = data["users"]

    cf = CollaborativeFilteringModel(n_factors=50, n_epochs=10)
    cf.evaluate_on_testset(ratings)

    cb = ContentBasedModel()
    cb.fit(movies, ratings)

    engine = HybridRecommender(alpha=0.7)
    engine.set_models(cf, cb)

    results = full_bias_report(cf, engine, movies, ratings, users)
    plot_popularity_distribution(results["popularity_bias"])
    plot_demographic_fairness(results["demographic_fairness"])
    plot_calibration_curve(results["calibration"])
