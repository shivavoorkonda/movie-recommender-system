# -*- coding: utf-8 -*-
"""
app.py - Flask REST API for the Hybrid Movie Recommender System
Endpoints:
  GET /                             -> Web UI
  GET /health                       -> Health check
  GET /api/stats                    -> Dataset statistics
  GET /api/popular?n=10             -> Popular movies
  GET /api/recommend/<user_id>?n=10 -> Hybrid recommendations
  GET /api/cf/<user_id>?n=10        -> Collaborative Filtering only
  GET /api/cb/<user_id>?n=10        -> Content-Based only
  GET /api/similar/<movie_id>?n=10  -> Similar movies
  GET /api/search?q=<query>         -> Search by title
  GET /api/user/<user_id>/history   -> User rating history
  GET /api/movies/<movie_id>        -> Movie details
  GET /api/users                    -> Top active users
"""

import sys, os, time, json, logging
from pathlib import Path
from flask import Flask, jsonify, request, render_template, abort

BASE = Path(__file__).resolve().parent.parent
SRC  = BASE / "src"
sys.path.insert(0, str(SRC))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

from data_loader             import load_all
from collaborative_filtering import CollaborativeFilteringModel
from content_based           import ContentBasedModel
from hybrid_engine           import HybridRecommender

app = Flask(__name__, template_folder="templates", static_folder="static")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

_data = _cf = _cb = _engine = None
_ready = False
_err_msg = None


def _load_models():
    global _data, _cf, _cb, _engine, _ready, _err_msg
    try:
        log.info("Loading dataset...")
        _data = load_all(verbose=False)
        log.info("Loading CF model...")
        _cf = CollaborativeFilteringModel()
        _cf.load()
        log.info("Loading CB model...")
        _cb = ContentBasedModel.load_from_disk()
        log.info("Assembling hybrid engine...")
        _engine = HybridRecommender(alpha=0.7, cold_start_threshold=10)
        _engine.set_models(_cf, _cb)
        _ready = True
        log.info("All models loaded.")
    except Exception as e:
        _err_msg = str(e)
        log.error(f"Failed: {e}")


def _require_ready():
    if not _ready:
        abort(503, description=_err_msg or "Models not loaded. Run: python src/main.py first.")


def _df_to_records(df, cols=None):
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    return json.loads(df.to_json(orient="records", force_ascii=False))


def _safe_int(val, default, lo=1, hi=100):
    try:
        return max(lo, min(hi, int(val)))
    except:
        return default


def _safe_float(val, default, lo=0.0, hi=1.0):
    try:
        return max(lo, min(hi, float(val)))
    except:
        return default


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok" if _ready else "initializing", "models_loaded": _ready, "error": _err_msg})


@app.route("/api/stats")
def stats():
    _require_ready()
    ratings = _data["ratings"]
    movies  = _data["movies"]

    genre_counts = {}
    for g_str in movies["genres"].dropna():
        for g in g_str.split("|"):
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

    return jsonify({
        "total_ratings": int(len(ratings)),
        "total_users":   int(ratings["user_id"].nunique()),
        "total_movies":  int(ratings["movie_id"].nunique()),
        "avg_rating":    round(float(ratings["rating"].mean()), 4),
        "sparsity":      round(float(1 - len(ratings) / (ratings["user_id"].nunique() * ratings["movie_id"].nunique())), 6),
        "rating_distribution": {str(k): int(v) for k, v in rating_dist.items()},
        "genre_counts":  genre_counts,
        "top_rated_movies": _df_to_records(top_rated, ["movie_id", "title", "genres", "count", "avg"]),
    })


@app.route("/api/popular")
def popular():
    _require_ready()
    n   = _safe_int(request.args.get("n"), 10, 1, 50)
    pop = HybridRecommender.popularity_fallback(_data["movies"], _data["ratings"], n=n)
    return jsonify({"type": "popular", "count": len(pop), "recommendations": _df_to_records(pop)})


@app.route("/api/recommend/<int:user_id>")
def recommend(user_id):
    _require_ready()
    n     = _safe_int(request.args.get("n"), 10, 1, 50)
    alpha = _safe_float(request.args.get("alpha"), 0.7)

    if user_id not in _data["ratings"]["user_id"].values:
        abort(404, description=f"User {user_id} not found.")

    _engine.alpha = alpha
    t0   = time.time()
    recs = _engine.recommend(user_id, _data["movies"], _data["ratings"], n=n)
    return jsonify({
        "type": "hybrid", "user_id": user_id, "alpha": alpha,
        "elapsed_sec": round(time.time() - t0, 3),
        "count": len(recs),
        "recommendations": _df_to_records(recs),
    })


@app.route("/api/cf/<int:user_id>")
def cf_recommend(user_id):
    _require_ready()
    n = _safe_int(request.args.get("n"), 10, 1, 50)
    if user_id not in _data["ratings"]["user_id"].values:
        abort(404, description=f"User {user_id} not found.")
    recs = _cf.get_top_n_recommendations(user_id, _data["movies"], _data["ratings"], n=n)
    return jsonify({"type": "collaborative_filtering", "user_id": user_id, "count": len(recs), "recommendations": _df_to_records(recs)})


@app.route("/api/cb/<int:user_id>")
def cb_recommend(user_id):
    _require_ready()
    n = _safe_int(request.args.get("n"), 10, 1, 50)
    if user_id not in _data["ratings"]["user_id"].values:
        abort(404, description=f"User {user_id} not found.")
    recs = _cb.get_user_profile_recommendations(user_id, _data["ratings"], n=n)
    return jsonify({"type": "content_based", "user_id": user_id, "count": len(recs), "recommendations": _df_to_records(recs)})


@app.route("/api/similar/<int:movie_id>")
def similar(movie_id):
    _require_ready()
    n = _safe_int(request.args.get("n"), 10, 1, 50)
    if movie_id not in _cb.movie_index:
        abort(404, description=f"Movie {movie_id} not found.")
    sims = _cb.get_similar_movies(movie_id, n=n)
    src_row = _data["movies"][_data["movies"]["movie_id"] == movie_id]
    src = src_row.iloc[0].to_dict() if not src_row.empty else {}
    return jsonify({
        "type": "similar",
        "source": {"movie_id": movie_id, "title": src.get("title", ""), "genres": src.get("genres", "")},
        "count": len(sims), "similar": _df_to_records(sims),
    })


@app.route("/api/search")
def search():
    _require_ready()
    q = (request.args.get("q") or "").strip()
    n = _safe_int(request.args.get("n"), 20, 1, 100)
    if not q:
        return jsonify({"results": [], "query": q, "count": 0})

    mask    = _data["movies"]["title"].str.contains(q, case=False, na=False)
    results = _data["movies"][mask].head(n).copy()
    stats_df = _data["ratings"].groupby("movie_id").agg(
        avg_rating=("rating", "mean"), n_ratings=("rating", "count")
    ).reset_index()
    results = results.merge(stats_df, on="movie_id", how="left")
    results["avg_rating"] = results["avg_rating"].round(2).fillna(0)
    results["n_ratings"]  = results["n_ratings"].fillna(0).astype(int)
    return jsonify({"query": q, "count": len(results),
                    "results": _df_to_records(results, ["movie_id","title","genres","year","avg_rating","n_ratings"])})


@app.route("/api/user/<int:user_id>/history")
def user_history(user_id):
    _require_ready()
    n = _safe_int(request.args.get("n"), 20, 1, 200)
    if user_id not in _data["ratings"]["user_id"].values:
        abort(404, description=f"User {user_id} not found.")
    user_r = (
        _data["ratings"][_data["ratings"]["user_id"] == user_id]
        .merge(_data["movies"][["movie_id","title","genres"]], on="movie_id")
        .sort_values("rating", ascending=False).head(n)
    )
    u_info = _data["users"][_data["users"]["user_id"] == user_id]
    user_info = u_info.iloc[0].to_dict() if not u_info.empty else {}
    return jsonify({
        "user_id": user_id, "user_info": user_info,
        "n_total": int(len(_data["ratings"][_data["ratings"]["user_id"] == user_id])),
        "history": _df_to_records(user_r, ["movie_id","title","genres","rating"]),
    })


@app.route("/api/movies/<int:movie_id>")
def movie_detail(movie_id):
    _require_ready()
    rows = _data["movies"][_data["movies"]["movie_id"] == movie_id]
    if rows.empty:
        abort(404, description=f"Movie {movie_id} not found.")
    movie = rows.iloc[0].to_dict()
    mr    = _data["ratings"][_data["ratings"]["movie_id"] == movie_id]["rating"]
    return jsonify({
        "movie_id": movie_id, "title": movie.get("title",""),
        "genres": movie.get("genres",""), "year": movie.get("year"),
        "avg_rating": round(float(mr.mean()), 2) if len(mr) else 0,
        "n_ratings": int(len(mr)),
        "rating_distribution": {str(r): int((mr==r).sum()) for r in [1,2,3,4,5]},
    })


@app.route("/api/users")
def users_list():
    _require_ready()
    uc = (_data["ratings"].groupby("user_id")["rating"].count()
          .reset_index().rename(columns={"rating":"n_ratings"})
          .sort_values("n_ratings", ascending=False).head(50))
    return jsonify({"total_users": int(_data["ratings"]["user_id"].nunique()), "users": _df_to_records(uc)})


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not Found", "message": str(e)}), 404

@app.errorhandler(503)
def unavailable(e):
    return jsonify({"error": "Service Unavailable", "message": str(e)}), 503

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal Server Error", "message": str(e)}), 500


if __name__ == "__main__":
    _load_models()
    app.run(host="0.0.0.0", port=5000, debug=False)
