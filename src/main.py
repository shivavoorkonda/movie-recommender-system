# -*- coding: utf-8 -*-
"""
main.py
=======
Entry point for the Movie Recommender System.

Run this script to:
  1. Download & load the MovieLens 100K dataset
  2. Train both the Collaborative Filtering (MF-SGD) and Content-Based models
  3. Assemble the Hybrid Recommender
  4. Evaluate all models (RMSE, MAE, Precision@K, Recall@K, Diversity)
  5. Generate and display recommendations for sample users
  6. Save all trained models to disk
  7. Produce and save evaluation plots

Usage
-----
    python main.py                        # Full pipeline
    python main.py --user 42             # Recs for a specific user
    python main.py --movie "Toy Story"   # Similar movies
    python main.py --new-user            # Popularity fallback for new user
    python main.py --skip-train          # Load saved models (fast re-run)

Architecture
------------
    MovieLens 100K
         |
         +-> [DataLoader]  loads ratings + movie metadata + user info
         |
         +-> [CollaborativeFilteringModel]  Matrix Factorisation (SGD)
         |        - Learns user & movie latent factor vectors
         |        - Predicts missing ratings
         |
         +-> [ContentBasedModel]  TF-IDF + Cosine Similarity
         |        - Builds movie feature vectors from genres
         |        - Finds similar movies / user genre profiles
         |
         +-> [HybridRecommender]  Weighted Fusion
                  - hybrid_score = alpha*CF + (1-alpha)*CB
                  - Auto-adjusts alpha for cold-start users
"""

import sys
import time
import argparse
from pathlib import Path

# Add src/ to path so we can import our modules
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from colorama import Fore, Style, init
init(autoreset=True)

from data_loader             import load_all
from collaborative_filtering import CollaborativeFilteringModel
from content_based           import ContentBasedModel
from hybrid_engine           import HybridRecommender
from evaluator               import (
    full_evaluation_report,
    plot_rating_distribution,
    plot_model_comparison,
    plot_precision_recall_curve,
    plot_genre_distribution,
    plot_user_rating_activity,
)
from utils import (
    print_header, print_recommendations,
    print_movie_details, get_user_history,
    describe_dataset, movie_id_from_title, format_elapsed,
)


# ── Banner (ASCII-safe) ────────────────────────────────────────────────────────

BANNER = (
    f"\n{Fore.CYAN}{'='*64}\n"
    f"{Fore.CYAN}  HYBRID MOVIE RECOMMENDER SYSTEM\n"
    f"{Fore.YELLOW}  MovieLens 100K  |  MF-SGD + TF-IDF + Cosine Similarity\n"
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
    cf = CollaborativeFilteringModel(n_factors=100, n_epochs=20, lr=0.005, reg=0.02)
    cb = ContentBasedModel()

    cf_loaded = False
    cb_loaded = False

    if args.skip_train:
        print(f"{Fore.YELLOW}[Main] Loading saved models ...")
        try:
            cf.load()
            cf_loaded = True
        except FileNotFoundError as e:
            print(f"{Fore.YELLOW}[Main] CF model not found, will train: {e}")

        try:
            cb = ContentBasedModel.load_from_disk()
            cb_loaded = True
        except FileNotFoundError as e:
            print(f"{Fore.YELLOW}[Main] CB model not found, will train: {e}")

    if not cf_loaded:
        print(f"\n{Fore.CYAN}-- Collaborative Filtering (Matrix Factorisation SGD) --")
        cf.evaluate_on_testset(ratings, test_size=0.2)
        cf.save()

    if not cb_loaded:
        print(f"\n{Fore.CYAN}-- Content-Based (TF-IDF + Cosine Similarity) --")
        cb.fit(movies)
        cb.save()

    # ── Step 3: Hybrid Engine ──────────────────────────────────────────────────
    print_header("Step 3 | Hybrid Engine")
    engine = HybridRecommender(alpha=args.alpha, cold_start_threshold=10)
    engine.set_models(cf, cb)

    # ── Step 4: Evaluation ─────────────────────────────────────────────────────
    print_header("Step 4 | Evaluation")
    metrics = full_evaluation_report(cf, cb, engine, ratings, movies, n_users=20)

    # ── Step 5: Visualisations ─────────────────────────────────────────────────
    if not args.no_plots:
        print_header("Step 5 | Generating Plots")
        plot_rating_distribution(ratings)
        plot_user_rating_activity(ratings)
        if hasattr(cf, "predictions") and cf.predictions:
            plot_model_comparison(metrics)
            plot_precision_recall_curve(cf.predictions)

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

        # CF-only
        cf_recs = cf.get_top_n_recommendations(uid, movies, ratings, n=args.n)
        print_recommendations(cf_recs, user_id=uid, mode="Collaborative Filtering", n=args.n)

        # CB-only
        cb_recs = cb.get_user_profile_recommendations(uid, ratings, n=args.n)
        print_recommendations(cb_recs, user_id=uid, mode="Content-Based", n=args.n)

        if not args.no_plots:
            plot_genre_distribution(recs, title=f"Hybrid Recs - User {uid}",
                                    filename=f"hybrid_genres_user{uid}.png")

    print(f"\n{Fore.GREEN}Pipeline complete in {format_elapsed(t0)}")
    print(f"{Fore.CYAN}  Plots saved  -> outputs/plots/")
    print(f"{Fore.CYAN}  Models saved -> models/saved/\n")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
