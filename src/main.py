# -*- coding: utf-8 -*-
"""
main.py
=======
Entry point for the Movie Recommender System.

Run this script to:
  1. Download & load the MovieLens 100K dataset
  2. Train both the Collaborative Filtering models (MF-SGD + ALS)
  3. Build the Content-Based model (TF-IDF + Cosine Similarity)
  4. Assemble the Hybrid Recommender (weighted fusion + MMR diversity)
  5. Evaluate all models (RMSE, MAE, NDCG, MAP, P@K, R@K, Coverage, Novelty)
  6. Generate recommendations for sample users
  7. Save all trained models to disk
  8. Produce evaluation plots and convergence curves

Usage
-----
    python main.py                        # Full pipeline
    python main.py --user 42             # Recs for a specific user
    python main.py --movie "Toy Story"   # Similar movies
    python main.py --new-user            # Popularity fallback for new user
    python main.py --skip-train          # Load saved models (fast re-run)
    python main.py --compare             # Side-by-side model comparison
    python main.py --analyse             # Run fairness & bias analysis
    python main.py --tune                # Hyperparameter tuning
    python main.py --full                # Everything: train + tune + eval + analyse

Architecture
------------
    MovieLens 100K
         |
         +-> [DataLoader]  loads ratings + movies + user demographics
         |        + Feature engineering (age bins, temporal, aggregates)
         |
         +-> [CF-SGD]  Matrix Factorisation via SGD (with lr decay)
         |        - Learns user & movie latent factor vectors
         |
         +-> [CF-ALS]  Matrix Factorisation via ALS
         |        - Alternating closed-form least squares
         |
         +-> [ContentBasedModel]  TF-IDF + Cosine Similarity
         |        - Genre + decade features, popularity debiasing
         |
         +-> [HybridRecommender]  Ensemble Fusion + MMR Diversity
                  - Blends SGD + ALS + CB with adaptive cold-start alpha
"""

import sys
import time
import json
import argparse
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from colorama import Fore, Style, init
init(autoreset=True)

from data_loader             import load_all
from collaborative_filtering import CollaborativeFilteringModel, ALSModel
from content_based           import ContentBasedModel
from hybrid_engine           import HybridRecommender
from evaluator               import (
    full_evaluation_report,
    model_comparison_table,
    plot_rating_distribution,
    plot_model_comparison,
    plot_precision_recall_curve,
    plot_genre_distribution,
    plot_user_rating_activity,
    plot_convergence,
)
from utils import (
    print_header, print_recommendations,
    print_movie_details, get_user_history,
    describe_dataset, movie_id_from_title, format_elapsed,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

BANNER = (
    f"\n{Fore.CYAN}{'='*64}\n"
    f"{Fore.CYAN}  HYBRID MOVIE RECOMMENDER SYSTEM\n"
    f"{Fore.YELLOW}  MovieLens 100K  |  SGD + ALS + TF-IDF + Hybrid Ensemble\n"
    f"{Fore.CYAN}{'='*64}{Style.RESET_ALL}\n"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid Movie Recommender - MovieLens 100K"
    )
    parser.add_argument("--user",       type=int,   default=None,
                        help="Get recommendations for a specific user ID")
    parser.add_argument("--movie",      type=str,   default=None,
                        help="Find movies similar to a movie title substring")
    parser.add_argument("--new-user",   action="store_true",
                        help="Show popularity-based recommendations (no history)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Load saved models instead of retraining")
    parser.add_argument("--alpha",      type=float, default=0.7,
                        help="Hybrid blending weight (0=pure CB, 1=pure CF). Default 0.7")
    parser.add_argument("--n",          type=int,   default=10,
                        help="Number of recommendations to show. Default 10")
    parser.add_argument("--no-plots",   action="store_true",
                        help="Skip generating visualisation plots")
    parser.add_argument("--compare",    action="store_true",
                        help="Print side-by-side model comparison table")
    parser.add_argument("--analyse",    action="store_true",
                        help="Run fairness and bias analysis")
    parser.add_argument("--tune",       action="store_true",
                        help="Run hyperparameter tuning")
    parser.add_argument("--full",       action="store_true",
                        help="Run everything: train + compare + analyse + plot")
    return parser.parse_args()


def run_pipeline(args) -> None:
    print(BANNER)
    t0 = time.time()

    # ── Step 1: Data Loading ───────────────────────────────────────────────────
    print_header("Step 1 | Loading Data")
    data    = load_all(verbose=True)
    ratings = data["ratings"]
    movies  = data["movies"]
    users   = data["users"]
    describe_dataset(ratings, movies, users)

    # ── Step 2: Model Training / Loading ──────────────────────────────────────
    print_header("Step 2 | Building Models")
    cf  = CollaborativeFilteringModel(n_factors=100, n_epochs=20, lr=0.005, reg=0.02)
    als = ALSModel(n_factors=80, n_iterations=15, reg=0.1)
    cb  = ContentBasedModel()

    cf_loaded  = False
    als_loaded = False
    cb_loaded  = False

    if args.skip_train:
        print(f"{Fore.YELLOW}[Main] Loading saved models ...")
        try:
            cf.load()
            cf_loaded = True
        except FileNotFoundError as e:
            print(f"{Fore.YELLOW}[Main] CF-SGD model not found, will train: {e}")

        try:
            als.load()
            als_loaded = True
        except FileNotFoundError as e:
            print(f"{Fore.YELLOW}[Main] CF-ALS model not found, will train: {e}")

        try:
            cb = ContentBasedModel.load_from_disk()
            cb_loaded = True
        except FileNotFoundError as e:
            print(f"{Fore.YELLOW}[Main] CB model not found, will train: {e}")

    if not cf_loaded:
        print(f"\n{Fore.CYAN}-- Collaborative Filtering (SGD) --")
        cf.evaluate_on_testset(ratings, test_size=0.2)
        cf.save()

    if not als_loaded:
        print(f"\n{Fore.CYAN}-- Collaborative Filtering (ALS) --")
        als.evaluate_on_testset(ratings, test_size=0.2)
        als.save()

    if not cb_loaded:
        print(f"\n{Fore.CYAN}-- Content-Based (TF-IDF + Cosine Similarity) --")
        cb.fit(movies, ratings)
        cb.save()

    # ── Step 3: Hybrid Engine ──────────────────────────────────────────────────
    print_header("Step 3 | Hybrid Engine")
    engine = HybridRecommender(alpha=args.alpha, cold_start_threshold=10, mmr_lambda=0.8)
    engine.set_models(cf, cb, als)

    # ── Step 4: Evaluation ─────────────────────────────────────────────────────
    print_header("Step 4 | Evaluation")
    metrics = full_evaluation_report(cf, cb, engine, ratings, movies, n_users=20, als_model=als)

    # ── Step 4b: Model Comparison ──────────────────────────────────────────────
    if args.compare or args.full:
        print_header("Step 4b | Model Comparison")
        model_comparison_table(cf, als)

    # ── Step 5: Visualisations ─────────────────────────────────────────────────
    if not args.no_plots:
        print_header("Step 5 | Generating Plots")
        plot_rating_distribution(ratings)
        plot_user_rating_activity(ratings)
        plot_convergence(cf, title="SGD")
        plot_convergence(als, title="ALS")
        if hasattr(cf, "predictions") and cf.predictions:
            plot_model_comparison(metrics)
            plot_precision_recall_curve(cf.predictions)

    # ── Step 5b: Fairness Analysis ─────────────────────────────────────────────
    if args.analyse or args.full:
        print_header("Step 5b | Fairness & Bias Analysis")
        from analysis import (
            full_bias_report,
            plot_popularity_distribution,
            plot_demographic_fairness,
            plot_calibration_curve,
        )
        bias_results = full_bias_report(cf, engine, movies, ratings, users)
        if not args.no_plots:
            plot_popularity_distribution(bias_results.get("popularity_bias", {}))
            plot_demographic_fairness(bias_results.get("demographic_fairness", {}))
            plot_calibration_curve(bias_results.get("calibration", {}))

    # ── Step 5c: Hyperparameter Tuning ─────────────────────────────────────────
    if args.tune or args.full:
        print_header("Step 5c | Hyperparameter Tuning")
        from tuner import HyperparamTuner
        tuner = HyperparamTuner()
        best_cf = tuner.tune_cf(ratings, n_trials=8, cv=3)
        tuner.print_summary()
        tuner.save_results()

    # ── Step 6: Recommendations ────────────────────────────────────────────────
    print_header("Step 6 | Generating Recommendations")

    # 6a. New-user popularity fallback
    if args.new_user:
        pop = HybridRecommender.popularity_fallback(movies, ratings, n=args.n)
        print_header(f"Popularity Fallback - Top {args.n} (New User)")
        from tabulate import tabulate
        print(tabulate(pop[["title", "genres", "avg_rating", "weighted_score"]],
                       headers=["Title", "Genres", "Avg Rating", "Weighted Score"],
                       tablefmt="rounded_outline", showindex=False, floatfmt=".4f"))
        print()
        return

    # 6b. Similar movies (content-based)
    if args.movie:
        mid = movie_id_from_title(args.movie, movies)
        if mid:
            print_movie_details(mid, movies, ratings)
            similar = cb.get_similar_movies(movie_id=mid, n=args.n)
            print_recommendations(similar, user_id=0, mode=f"Similar to '{args.movie}'", n=args.n)
            if not args.no_plots:
                plot_genre_distribution(similar, title=f"Similar to {args.movie}",
                                        filename="similar_movies_genres.png")
        return

    # 6c. Recommendations for specific / sample users
    target_users = [args.user] if args.user else [1, 42, 100, 200]

    for uid in target_users:
        if uid not in ratings["user_id"].values:
            print(f"{Fore.RED}[Main] User {uid} not found in dataset. Skipping.")
            continue

        # Show user history
        print_header(f"User #{uid} - Rating History (Top 5)")
        history = get_user_history(uid, ratings, movies, top_n=5)
        from tabulate import tabulate
        print(tabulate(history[["title", "genres", "rating"]],
                       headers=["Title", "Genres", "Rating"],
                       tablefmt="rounded_outline", showindex=False))
        print()

        # Hybrid recommendations
        recs = engine.recommend(uid, movies, ratings, n=args.n)
        print_recommendations(recs, user_id=uid, mode="Hybrid", n=args.n)

        # CF-SGD
        cf_recs = cf.get_top_n_recommendations(uid, movies, ratings, n=args.n)
        print_recommendations(cf_recs, user_id=uid, mode="CF-SGD", n=args.n)

        # CF-ALS
        als_recs = als.get_top_n_recommendations(uid, movies, ratings, n=args.n)
        print_recommendations(als_recs, user_id=uid, mode="CF-ALS", n=args.n)

        # CB-only
        cb_recs = cb.get_user_profile_recommendations(uid, ratings, n=args.n)
        print_recommendations(cb_recs, user_id=uid, mode="Content-Based", n=args.n)

        if not args.no_plots:
            plot_genre_distribution(recs, title=f"Hybrid Recs - User {uid}",
                                    filename=f"hybrid_genres_user{uid}.png")

    # ── Save experiment results ────────────────────────────────────────────────
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUTS_DIR / "experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"{Fore.GREEN}[Main] Experiment results saved → {results_path}")

    print(f"\n{Fore.GREEN}Pipeline complete in {format_elapsed(t0)}")
    print(f"{Fore.CYAN}  Plots saved  -> outputs/plots/")
    print(f"{Fore.CYAN}  Models saved -> models/saved/")
    print(f"{Fore.CYAN}  Results      -> outputs/experiment_results.json\n")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
