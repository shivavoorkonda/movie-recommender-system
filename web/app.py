# -*- coding: utf-8 -*-
"""
app.py - Flask REST API for the Hybrid Movie Recommender System
===============================================================

Endpoints
---------
  GET  /                                -> Web UI (SPA)
  GET  /health                          -> Server health + model status

  GET  /api/stats                       -> Dataset-level statistics
  GET  /api/genres                      -> All unique genres with counts
  GET  /api/popular?n=10                -> Bayesian popularity rankings
  GET  /api/movies?page=1&per_page=20   -> Paginated movie catalogue
  GET  /api/movies/<movie_id>           -> Movie detail + rating distribution
  GET  /api/search?q=<query>&n=20       -> Full-text title search

  GET  /api/users?limit=50             -> Top active users
  GET  /api/user/<user_id>/history?n=20 -> User rating history + demographics
  GET  /api/user/<user_id>/stats        -> Per-user statistics summary

  GET  /api/recommend/<user_id>?n=10&alpha=0.7 -> Hybrid recommendations
  GET  /api/cf/<user_id>?n=10           -> Collaborative Filtering only
  GET  /api/cb/<user_id>?n=10           -> Content-Based only
  GET  /api/similar/<movie_id>?n=10     -> Similar movies (CB cosine sim)
  GET  /api/predict?user=1&movie=50     -> Predict single rating (CF)
"""

import sys, os, time, json, logging
from pathlib import Path
from datetime import datetime

from flask import Flask, jsonify, request, render_template, abort, g

# ── Path setup ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
SRC  = BASE / "src"
sys.path.insert(0, str(SRC))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8",       "1")

from data_loader             import load_all
from collaborative_filtering import CollaborativeFilteringModel
from content_based           import ContentBasedModel
from hybrid_engine           import HybridRecommender

# ── App config ────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["JSON_SORT_KEYS"] = False   # preserve insertion order in responses

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("cineai")

# ── Global model state ────────────────────────────────────────────────────────
_data    = None   # {"ratings": df, "movies": df, "users": df}
_cf      = None   # CollaborativeFilteringModel
_cb      = None   # ContentBasedModel
_engine  = None   # HybridRecommender
_ready   = False
_err_msg = None
_started_at = datetime.utcnow().isoformat() + "Z"


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_models() -> None:
    global _data, _cf, _cb, _engine, _ready, _err_msg
    try:
        log.info("Loading dataset ...")
        _data = load_all(verbose=False)

        log.info("Loading Collaborative Filtering model ...")
        _cf = CollaborativeFilteringModel()
        _cf.load()

        log.info("Loading Content-Based model ...")
        _cb = ContentBasedModel.load_from_disk()

        log.info("Assembling Hybrid engine ...")
        _engine = HybridRecommender(alpha=0.7, cold_start_threshold=10)
        _engine.set_models(_cf, _cb)

        _ready = True
        log.info("All models loaded successfully — server ready.")
    except FileNotFoundError as e:
        _err_msg = (
            f"Trained models not found: {e}. "
            "Run:  python src/main.py  to train first."
        )
        log.error(_err_msg)
    except Exception as e:
        _err_msg = str(e)
        log.exception("Failed to load models")


# ── Middleware ────────────────────────────────────────────────────────────────

@app.before_request
def _before():
    g.t0 = time.time()


@app.after_request
def _after(response):
    # CORS — allow any origin (fine for a portfolio / local demo)
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["X-Powered-By"]                 = "CineAI / Flask"
    # Per-request timing log
    elapsed = round((time.time() - g.t0) * 1000, 1)
    if not request.path.startswith("/static"):
        log.info(f"{request.method} {request.path} -> {response.status_code} ({elapsed} ms)")
    return response


@app.route("/", methods=["OPTIONS"])
@app.route("/api/<path:p>", methods=["OPTIONS"])
def _options(p=""):
    return "", 204


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_ready():
    """Abort with 503 if models are not loaded."""
    if not _ready:
        abort(503, description=_err_msg or "Models not loaded. Run: python src/main.py first.")


def _df_to_records(df, cols=None):
    """DataFrame → JSON-safe list of dicts."""
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    return json.loads(df.to_json(orient="records", force_ascii=False))


def _safe_int(val, default, lo=1, hi=1000):
    try:
        return max(lo, min(hi, int(val)))
    except (TypeError, ValueError):
        return default


def _safe_float(val, default, lo=0.0, hi=1.0):
    try:
        return max(lo, min(hi, float(val)))
    except (TypeError, ValueError):
        return default


def _enrich_with_year(recs: list) -> list:
    """Add 'year' field to recommendation dicts from movies data."""
    if _data is None:
        return recs
    year_map = (
        _data["movies"].set_index("movie_id")["year"].to_dict()
        if "year" in _data["movies"].columns else {}
    )
    for r in recs:
        r.setdefault("year", year_map.get(r.get("movie_id")))
    return recs


# ── UI Route ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── Health ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({
        "status":       "ok" if _ready else "initializing",
        "models_loaded": _ready,
        "started_at":   _started_at,
        "error":        _err_msg,
        "dataset": {
            "users":   int(_data["ratings"]["user_id"].nunique()) if _ready else None,
            "movies":  int(_data["ratings"]["movie_id"].nunique()) if _ready else None,
            "ratings": int(len(_data["ratings"]))                  if _ready else None,
        } if _ready else None,
    })


# ── Dataset Statistics ────────────────────────────────────────────────────────

@app.route("/api/stats")
def stats():
    _require_ready()
    ratings = _data["ratings"]
    movies  = _data["movies"]

    genre_counts: dict[str, int] = {}
    for g_str in movies["genres"].dropna():
        for g in str(g_str).split("|"):
            g = g.strip()
            if g:
                genre_counts[g] = genre_counts.get(g, 0) + 1

    rating_dist = ratings["rating"].value_counts().sort_index()

    top_rated = (
        ratings.groupby("movie_id")
        .agg(count=("rating", "count"), avg=("rating", "mean"))
        .reset_index()
        .merge(movies[["movie_id", "title", "genres"]], on="movie_id")
        .sort_values("count", ascending=False)
        .head(10)
    )
    top_rated["avg"] = top_rated["avg"].round(2)

    n_users  = int(ratings["user_id"].nunique())
    n_movies = int(ratings["movie_id"].nunique())

    return jsonify({
        "total_ratings":   int(len(ratings)),
        "total_users":     n_users,
        "total_movies":    n_movies,
        "avg_rating":      round(float(ratings["rating"].mean()), 4),
        "sparsity":        round(float(1 - len(ratings) / (n_users * n_movies)), 6),
        "rating_distribution": {str(k): int(v) for k, v in rating_dist.items()},
        "genre_counts":    genre_counts,
        "top_rated_movies": _df_to_records(
            top_rated, ["movie_id", "title", "genres", "count", "avg"]),
    })


# ── Genres ────────────────────────────────────────────────────────────────────

@app.route("/api/genres")
def genres():
    """Return all unique genres sorted by movie count."""
    _require_ready()
    genre_counts: dict[str, int] = {}
    for g_str in _data["movies"]["genres"].dropna():
        for g in str(g_str).split("|"):
            g = g.strip()
            if g:
                genre_counts[g] = genre_counts.get(g, 0) + 1
    sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
    return jsonify({
        "count":  len(sorted_genres),
        "genres": [{"genre": g, "n_movies": n} for g, n in sorted_genres],
    })


# ── Movie Catalogue (paginated) ───────────────────────────────────────────────

@app.route("/api/movies")
def movies_list():
    """
    Paginated movie catalogue.
    Query params: page (1-based), per_page (1-200), genre (filter), sort (rating|count|title)
    """
    _require_ready()
    page     = _safe_int(request.args.get("page"),     1,  lo=1,  hi=10000)
    per_page = _safe_int(request.args.get("per_page"), 20, lo=1,  hi=200)
    genre    = (request.args.get("genre") or "").strip()
    sort_by  = request.args.get("sort", "count")   # rating | count | title

    movies  = _data["movies"].copy()
    ratings = _data["ratings"]

    # Genre filter
    if genre:
        movies = movies[movies["genres"].str.contains(genre, case=False, na=False)]

    # Join rating stats
    stats_df = (
        ratings.groupby("movie_id")
        .agg(avg_rating=("rating", "mean"), n_ratings=("rating", "count"))
        .reset_index()
    )
    movies = movies.merge(stats_df, on="movie_id", how="left")
    movies["avg_rating"] = movies["avg_rating"].round(2).fillna(0)
    movies["n_ratings"]  = movies["n_ratings"].fillna(0).astype(int)

    # Sort
    if sort_by == "rating":
        movies = movies.sort_values("avg_rating", ascending=False)
    elif sort_by == "title":
        movies = movies.sort_values("title")
    else:
        movies = movies.sort_values("n_ratings", ascending=False)

    total = len(movies)
    start = (page - 1) * per_page
    end   = start + per_page
    page_movies = movies.iloc[start:end]

    return jsonify({
        "page":       page,
        "per_page":   per_page,
        "total":      total,
        "total_pages": (total + per_page - 1) // per_page,
        "genre_filter": genre or None,
        "sort":       sort_by,
        "movies":     _df_to_records(
            page_movies, ["movie_id","title","genres","year","avg_rating","n_ratings"]),
    })


# ── Movie Detail ──────────────────────────────────────────────────────────────

@app.route("/api/movies/<int:movie_id>")
def movie_detail(movie_id):
    _require_ready()
    rows = _data["movies"][_data["movies"]["movie_id"] == movie_id]
    if rows.empty:
        abort(404, description=f"Movie {movie_id} not found.")

    movie = rows.iloc[0].to_dict()
    mr    = _data["ratings"][_data["ratings"]["movie_id"] == movie_id]["rating"]

    # Rating distribution
    rating_dist = {str(r): int((mr == r).sum()) for r in [1, 2, 3, 4, 5]}

    return jsonify({
        "movie_id":   movie_id,
        "title":      movie.get("title", ""),
        "genres":     movie.get("genres", ""),
        "year":       movie.get("year"),
        "avg_rating": round(float(mr.mean()), 2) if len(mr) else 0,
        "n_ratings":  int(len(mr)),
        "rating_distribution": rating_dist,
    })


# ── Search ────────────────────────────────────────────────────────────────────

@app.route("/api/search")
def search():
    _require_ready()
    q    = (request.args.get("q") or "").strip()
    n    = _safe_int(request.args.get("n"), 20, lo=1, hi=100)
    genre = (request.args.get("genre") or "").strip()

    if not q and not genre:
        return jsonify({"results": [], "query": q, "count": 0})

    movies = _data["movies"].copy()

    if q:
        movies = movies[movies["title"].str.contains(q, case=False, na=False)]
    if genre:
        movies = movies[movies["genres"].str.contains(genre, case=False, na=False)]

    movies = movies.head(n).copy()

    stats_df = (
        _data["ratings"].groupby("movie_id")
        .agg(avg_rating=("rating", "mean"), n_ratings=("rating", "count"))
        .reset_index()
    )
    movies = movies.merge(stats_df, on="movie_id", how="left")
    movies["avg_rating"] = movies["avg_rating"].round(2).fillna(0)
    movies["n_ratings"]  = movies["n_ratings"].fillna(0).astype(int)

    return jsonify({
        "query":  q,
        "genre":  genre or None,
        "count":  len(movies),
        "results": _df_to_records(
            movies, ["movie_id","title","genres","year","avg_rating","n_ratings"]),
    })


# ── Popular ───────────────────────────────────────────────────────────────────

@app.route("/api/popular")
def popular():
    _require_ready()
    n   = _safe_int(request.args.get("n"), 10, lo=1, hi=50)
    genre = (request.args.get("genre") or "").strip()

    pop = HybridRecommender.popularity_fallback(
        _data["movies"], _data["ratings"], n=n * 3 if genre else n)

    if genre:
        pop = pop[pop["genres"].str.contains(genre, case=False, na=False)].head(n)

    recs = _df_to_records(pop)
    recs = _enrich_with_year(recs)
    return jsonify({"type": "popular", "count": len(recs), "recommendations": recs})


# ── Predict Single Rating ─────────────────────────────────────────────────────

@app.route("/api/predict")
def predict_rating():
    """
    Predict the rating a specific user would give a specific movie.
    Query params: user=<id>, movie=<id>
    """
    _require_ready()
    user_id  = _safe_int(request.args.get("user"),  None, lo=1, hi=99999)
    movie_id = _safe_int(request.args.get("movie"), None, lo=1, hi=99999)

    if user_id is None or movie_id is None:
        return jsonify({"error": "Both 'user' and 'movie' query parameters are required."}), 400

    if user_id not in _data["ratings"]["user_id"].values:
        abort(404, description=f"User {user_id} not found.")

    movie_rows = _data["movies"][_data["movies"]["movie_id"] == movie_id]
    if movie_rows.empty:
        abort(404, description=f"Movie {movie_id} not found.")

    predicted = _cf.predict_rating(user_id, movie_id)
    movie     = movie_rows.iloc[0]
    already_rated = bool(
        len(_data["ratings"][
            (_data["ratings"]["user_id"]  == user_id) &
            (_data["ratings"]["movie_id"] == movie_id)
        ]) > 0
    )

    return jsonify({
        "user_id":       user_id,
        "movie_id":      movie_id,
        "title":         movie.get("title", ""),
        "genres":        movie.get("genres", ""),
        "predicted_rating": predicted,
        "already_rated": already_rated,
        "scale":         "1.0 – 5.0",
    })


# ── Hybrid Recommendations ────────────────────────────────────────────────────

@app.route("/api/recommend/<int:user_id>")
def recommend(user_id):
    _require_ready()
    n     = _safe_int(request.args.get("n"),    10, lo=1, hi=50)
    alpha = _safe_float(request.args.get("alpha"), 0.7)

    if user_id not in _data["ratings"]["user_id"].values:
        abort(404, description=f"User {user_id} not found.")

    _engine.alpha = alpha
    t0   = time.time()
    recs = _engine.recommend(user_id, _data["movies"], _data["ratings"], n=n)
    elapsed = round(time.time() - t0, 3)

    rec_list = _df_to_records(recs)
    rec_list = _enrich_with_year(rec_list)

    n_rated = int(len(_data["ratings"][_data["ratings"]["user_id"] == user_id]))
    return jsonify({
        "type":          "hybrid",
        "user_id":       user_id,
        "alpha":         alpha,
        "n_user_ratings": n_rated,
        "cold_start":    n_rated < _engine.cold_start_threshold,
        "elapsed_sec":   elapsed,
        "count":         len(rec_list),
        "recommendations": rec_list,
    })


# ── Collaborative Filtering Only ──────────────────────────────────────────────

@app.route("/api/cf/<int:user_id>")
def cf_recommend(user_id):
    _require_ready()
    n = _safe_int(request.args.get("n"), 10, lo=1, hi=50)

    if user_id not in _data["ratings"]["user_id"].values:
        abort(404, description=f"User {user_id} not found.")

    recs = _cf.get_top_n_recommendations(
        user_id, _data["movies"], _data["ratings"], n=n)
    rec_list = _df_to_records(recs)
    rec_list = _enrich_with_year(rec_list)

    return jsonify({
        "type":            "collaborative_filtering",
        "user_id":         user_id,
        "count":           len(rec_list),
        "recommendations": rec_list,
    })


# ── Content-Based Only ────────────────────────────────────────────────────────

@app.route("/api/cb/<int:user_id>")
def cb_recommend(user_id):
    _require_ready()
    n = _safe_int(request.args.get("n"), 10, lo=1, hi=50)

    if user_id not in _data["ratings"]["user_id"].values:
        abort(404, description=f"User {user_id} not found.")

    recs = _cb.get_user_profile_recommendations(
        user_id, _data["ratings"], n=n)
    rec_list = _df_to_records(recs)
    rec_list = _enrich_with_year(rec_list)

    return jsonify({
        "type":            "content_based",
        "user_id":         user_id,
        "count":           len(rec_list),
        "recommendations": rec_list,
    })


# ── Similar Movies ────────────────────────────────────────────────────────────

@app.route("/api/similar/<int:movie_id>")
def similar(movie_id):
    _require_ready()
    n = _safe_int(request.args.get("n"), 10, lo=1, hi=50)

    if movie_id not in _cb.movie_index:
        abort(404, description=f"Movie {movie_id} not in content model.")

    sims    = _cb.get_similar_movies(movie_id, n=n)
    src_row = _data["movies"][_data["movies"]["movie_id"] == movie_id]
    src     = src_row.iloc[0].to_dict() if not src_row.empty else {}

    sim_list = _df_to_records(sims)
    sim_list = _enrich_with_year(sim_list)

    return jsonify({
        "type":   "similar",
        "source": {
            "movie_id": movie_id,
            "title":    src.get("title", ""),
            "genres":   src.get("genres", ""),
            "year":     src.get("year"),
        },
        "count":  len(sim_list),
        "similar": sim_list,
    })


# ── User History ──────────────────────────────────────────────────────────────

@app.route("/api/user/<int:user_id>/history")
def user_history(user_id):
    _require_ready()
    n   = _safe_int(request.args.get("n"),  20, lo=1, hi=200)
    sort = request.args.get("sort", "rating")   # rating | title | movie_id

    if user_id not in _data["ratings"]["user_id"].values:
        abort(404, description=f"User {user_id} not found.")

    user_r = (
        _data["ratings"][_data["ratings"]["user_id"] == user_id]
        .merge(_data["movies"][["movie_id", "title", "genres", "year"]]
               if "year" in _data["movies"].columns
               else _data["movies"][["movie_id", "title", "genres"]],
               on="movie_id")
    )

    if sort == "title":
        user_r = user_r.sort_values("title")
    else:
        user_r = user_r.sort_values("rating", ascending=False)

    user_r = user_r.head(n)

    u_info = _data["users"][_data["users"]["user_id"] == user_id]
    user_info = u_info.iloc[0].to_dict() if not u_info.empty else {}

    return jsonify({
        "user_id":   user_id,
        "user_info": user_info,
        "n_total":   int(len(_data["ratings"][_data["ratings"]["user_id"] == user_id])),
        "sort":      sort,
        "history":   _df_to_records(user_r, ["movie_id","title","genres","year","rating"]),
    })


# ── User Statistics ───────────────────────────────────────────────────────────

@app.route("/api/user/<int:user_id>/stats")
def user_stats(user_id):
    """Per-user analytics: rating breakdown, genre preferences, activity."""
    _require_ready()

    if user_id not in _data["ratings"]["user_id"].values:
        abort(404, description=f"User {user_id} not found.")

    user_r = _data["ratings"][_data["ratings"]["user_id"] == user_id].copy()
    user_r = user_r.merge(_data["movies"][["movie_id","genres"]], on="movie_id", how="left")

    # Rating distribution
    rating_dist = {str(r): int((user_r["rating"] == r).sum()) for r in [1,2,3,4,5]}

    # Genre preferences
    genre_scores: dict[str, list] = {}
    for _, row in user_r.iterrows():
        for g in str(row.get("genres","")).split("|"):
            g = g.strip()
            if g:
                genre_scores.setdefault(g, []).append(row["rating"])

    genre_prefs = [
        {"genre": g, "avg_rating": round(sum(v)/len(v), 2), "n_rated": len(v)}
        for g, v in genre_scores.items()
    ]
    genre_prefs.sort(key=lambda x: x["avg_rating"], reverse=True)

    u_info = _data["users"][_data["users"]["user_id"] == user_id]
    user_info = u_info.iloc[0].to_dict() if not u_info.empty else {}

    return jsonify({
        "user_id":          user_id,
        "user_info":        user_info,
        "n_ratings":        int(len(user_r)),
        "avg_rating":       round(float(user_r["rating"].mean()), 2),
        "rating_distribution": rating_dist,
        "genre_preferences": genre_prefs[:10],
        "cold_start":       len(user_r) < 10,
    })


# ── Users List ────────────────────────────────────────────────────────────────

@app.route("/api/users")
def users_list():
    _require_ready()
    limit = _safe_int(request.args.get("limit"), 50, lo=1, hi=200)

    uc = (
        _data["ratings"].groupby("user_id")["rating"]
        .agg(n_ratings="count", avg_rating="mean")
        .reset_index()
        .sort_values("n_ratings", ascending=False)
        .head(limit)
    )
    uc["avg_rating"] = uc["avg_rating"].round(2)

    return jsonify({
        "total_users": int(_data["ratings"]["user_id"].nunique()),
        "shown":       len(uc),
        "users":       _df_to_records(uc),
    })


# ── Error Handlers ────────────────────────────────────────────────────────────

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad Request", "message": str(e)}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not Found", "message": str(e)}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method Not Allowed", "message": str(e)}), 405

@app.errorhandler(503)
def unavailable(e):
    return jsonify({"error": "Service Unavailable", "message": str(e)}), 503

@app.errorhandler(500)
def server_error(e):
    log.exception("Internal server error")
    return jsonify({"error": "Internal Server Error", "message": str(e)}), 500


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _load_models()
    app.run(host="0.0.0.0", port=5000, debug=False)
