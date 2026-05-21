# -*- coding: utf-8 -*-
"""
app.py — Flask REST API for the Hybrid Movie Recommender
=========================================================
Serves the frontend SPA and exposes recommendation endpoints.

Endpoints
---------
  GET  /                              → SPA (index.html)
  GET  /health                        → Server health + model status
  GET  /api/stats                     → Dataset stats + genre counts
  GET  /api/genres                    → All genre names with counts
  GET  /api/movies?page=1&per_page=20 → Paginated movie catalogue
  GET  /api/movies/<id>               → Single movie detail
  GET  /api/search?q=toy&genre=Comedy → Search movies
  GET  /api/popular?n=10              → Bayesian popularity
  GET  /api/predict?user=1&movie=50   → Predict single rating (CF)
  GET  /api/recommend/<user_id>       → Hybrid recommendations
  GET  /api/cf/<user_id>              → CF-only recommendations
  GET  /api/als/<user_id>             → ALS-only recommendations
  GET  /api/cb/<user_id>              → CB-only recommendations
  GET  /api/similar/<movie_id>        → Content-based similar movies
  GET  /api/user/<id>/history         → User rating history
  GET  /api/user/<id>/stats           → Per-user analytics
  GET  /api/users                     → Top active users
  GET  /api/compare/<user_id>         → Side-by-side model comparison
  GET  /api/fairness                  → Pre-computed bias metrics
  GET  /api/confidence?user=1&movie=1 → Prediction + confidence interval
"""

import sys, os, time, json, logging, mimetypes
from pathlib import Path
from datetime import datetime, timezone
from functools import lru_cache

# Windows registry workaround for CSS/JS mime types
mimetypes.add_type('text/css', '.css')
mimetypes.add_type('application/javascript', '.js')

from flask import Flask, jsonify, request, render_template, abort, g

# ── Path setup ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
SRC  = BASE / "src"
sys.path.insert(0, str(SRC))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8",       "1")

from data_loader             import load_all
from collaborative_filtering import CollaborativeFilteringModel, ALSModel
from content_based           import ContentBasedModel
from hybrid_engine           import HybridRecommender

# ── App factory ───────────────────────────────────────────────────────────────
app = Flask(__name__,
            template_folder=str(BASE / "web" / "templates"),
            static_folder=str(BASE / "web" / "static"))
app.config["JSON_SORT_KEYS"] = False

_started_at = datetime.now(timezone.utc).isoformat()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("cineai")

# ── Globals (filled by _load_models) ──────────────────────────────────────────
_ratings = _movies = _users = _merged = None
_cf  = None   # SGD model
_als = None   # ALS model
_cb  = None
_engine = None
_user_features = None
_movie_features = None
_fairness_cache = None   # cached bias report


def _load_models():
    global _ratings, _movies, _users, _merged, _cf, _als, _cb, _engine
    global _user_features, _movie_features

    log.info("Loading dataset ...")
    data     = load_all(verbose=False)
    _ratings = data["ratings"]
    _movies  = data["movies"]
    _users   = data["users"]
    _merged  = data["merged"]
    _user_features  = data.get("user_features")
    _movie_features = data.get("movie_features")

    log.info("Loading Collaborative Filtering (SGD) model ...")
    _cf = CollaborativeFilteringModel()
    _cf.load()

    log.info("Loading Collaborative Filtering (ALS) model ...")
    _als = ALSModel()
    try:
        _als.load()
    except FileNotFoundError:
        log.warning("ALS model not found — ALS endpoints will use SGD fallback.")
        _als = None

    log.info("Loading Content-Based model ...")
    _cb = ContentBasedModel.load_from_disk()

    log.info("Assembling Hybrid engine ...")
    _engine = HybridRecommender(alpha=0.7, cold_start_threshold=10, mmr_lambda=0.8)
    _engine.set_models(_cf, _cb, _als)

    log.info("All models loaded successfully — server ready.")


_load_models()


# ── Middleware ────────────────────────────────────────────────────────────────
@app.before_request
def _before():
    g.t0 = time.time()

@app.after_request
def _after(resp):
    resp.headers["Access-Control-Allow-Origin"]  = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    ms = (time.time() - g.t0) * 1000
    log.info(f"{request.method} {request.path} → {resp.status_code} ({ms:.0f}ms)")
    return resp


# ── Error handlers ────────────────────────────────────────────────────────────
@app.errorhandler(400)
def bad_request(e):
    return jsonify(error="Bad request", detail=str(e)), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify(error="Not found"), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify(error="Method not allowed"), 405

@app.errorhandler(500)
def internal_error(e):
    log.exception("Internal server error")
    return jsonify(error="Internal server error"), 500

@app.errorhandler(503)
def unavailable(e):
    return jsonify(error="Models not loaded yet"), 503


# ── Helpers ───────────────────────────────────────────────────────────────────
def _movie_to_dict(row, ratings_df=None):
    """Convert a movie DataFrame row to a JSON-friendly dict."""
    d = {
        "movie_id": int(row["movie_id"]),
        "title":    str(row["title"]),
        "genres":   str(row.get("genres", "")),
        "year":     int(row["year"]) if "year" in row and not (row.get("year") != row.get("year")) else None,
    }
    if ratings_df is not None:
        mr = ratings_df[ratings_df["movie_id"] == d["movie_id"]]["rating"]
        d["avg_rating"] = round(float(mr.mean()), 2) if len(mr) else None
        d["n_ratings"]  = int(len(mr))
    return d


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(
        status="ok",
        started_at=_started_at,
        models={
            "cf_sgd": _cf is not None and _cf._fitted,
            "cf_als": _als is not None and _als._fitted,
            "cb":     _cb is not None and _cb._fitted,
            "hybrid": _engine is not None and _engine._fitted,
        },
        dataset={
            "ratings": len(_ratings) if _ratings is not None else 0,
            "movies":  len(_movies)  if _movies  is not None else 0,
            "users":   len(_users)   if _users   is not None else 0,
        },
    )


# ── Dataset Stats ─────────────────────────────────────────────────────────────
@app.route("/api/stats")
def api_stats():
    r, m, u = _ratings, _movies, _users
    genre_counts = {}
    for g_str in m["genres"].dropna():
        for g in g_str.split("|"):
            if g != "unknown":
                genre_counts[g] = genre_counts.get(g, 0) + 1
    genre_counts = dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True))

    rating_dist = r["rating"].value_counts().sort_index().to_dict()
    rating_dist = {str(int(k)): int(v) for k, v in rating_dist.items()}

    # Top rated movies
    movie_stats = (r.groupby("movie_id")
                   .agg(avg=("rating", "mean"), count=("rating", "count"))
                   .reset_index())
    movie_stats = movie_stats[movie_stats["count"] >= 50]
    top10 = movie_stats.sort_values("avg", ascending=False).head(10)
    top_rated = []
    for _, row in top10.iterrows():
        mrow = m[m["movie_id"] == row["movie_id"]]
        if not mrow.empty:
            mr = mrow.iloc[0]
            top_rated.append({
                "movie_id": int(row["movie_id"]),
                "title": str(mr["title"]),
                "genres": str(mr["genres"]),
                "year": int(mr["year"]) if not (mr["year"] != mr["year"]) else None,
                "avg": round(float(row["avg"]), 2),
                "count": int(row["count"]),
            })

    return jsonify(
        n_ratings=int(len(r)),
        n_users=int(r["user_id"].nunique()),
        n_movies=int(r["movie_id"].nunique()),
        avg_rating=round(float(r["rating"].mean()), 3),
        sparsity=round(1 - len(r) / (r["user_id"].nunique() * r["movie_id"].nunique()), 4),
        rating_distribution=rating_dist,
        genre_counts=genre_counts,
        top_rated_movies=top_rated,
    )


@app.route("/api/genres")
def api_genres():
    genre_counts = {}
    for g_str in _movies["genres"].dropna():
        for g in g_str.split("|"):
            if g != "unknown":
                genre_counts[g] = genre_counts.get(g, 0) + 1
    result = [{"genre": g, "count": c} for g, c in
              sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)]
    return jsonify(genres=result)


# ── Movie Catalogue ──────────────────────────────────────────────────────────
@app.route("/api/movies")
def api_movies():
    page     = max(1, request.args.get("page", 1, type=int))
    per_page = min(100, max(1, request.args.get("per_page", 20, type=int)))
    genre    = request.args.get("genre", "").strip()
    sort     = request.args.get("sort", "title")

    df = _movies.copy()
    if genre:
        df = df[df["genres"].str.contains(genre, case=False, na=False)]

    if sort == "year":
        df = df.sort_values("year", ascending=False, na_position="last")
    else:
        df = df.sort_values("title")

    total = len(df)
    start = (page - 1) * per_page
    page_df = df.iloc[start:start + per_page]

    items = [_movie_to_dict(row, _ratings) for _, row in page_df.iterrows()]
    return jsonify(movies=items, page=page, per_page=per_page,
                   total=total, pages=(total + per_page - 1) // per_page)


@app.route("/api/movies/<int:movie_id>")
def api_movie_detail(movie_id):
    row = _movies[_movies["movie_id"] == movie_id]
    if row.empty:
        abort(404)
    row = row.iloc[0]
    d = _movie_to_dict(row, _ratings)
    mr = _ratings[_ratings["movie_id"] == movie_id]["rating"]
    d["rating_distribution"] = {str(int(k)): int(v) for k, v in
                                mr.value_counts().sort_index().items()}
    return jsonify(d)


# ── Search ────────────────────────────────────────────────────────────────────
@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify(results=[], query="")
    mask = _movies["title"].str.contains(q, case=False, na=False)
    genre = request.args.get("genre", "").strip()
    if genre:
        mask &= _movies["genres"].str.contains(genre, case=False, na=False)
    hits = _movies[mask].head(25)
    results = [_movie_to_dict(row, _ratings) for _, row in hits.iterrows()]
    return jsonify(results=results, query=q, total=int(mask.sum()))


# ── Popular ───────────────────────────────────────────────────────────────────
@app.route("/api/popular")
def api_popular():
    n = min(50, max(1, request.args.get("n", 12, type=int)))
    pop = HybridRecommender.popularity_fallback(_movies, _ratings, n=n)
    items = []
    for _, row in pop.iterrows():
        d = {"movie_id": int(row["movie_id"]), "title": str(row["title"]),
             "genres": str(row["genres"]),
             "avg_rating": round(float(row["avg_rating"]), 2),
             "n_ratings": int(row["n_ratings"]),
             "weighted_score": round(float(row["weighted_score"]), 3)}
        mr = _movies[_movies["movie_id"] == d["movie_id"]]
        d["year"] = int(mr.iloc[0]["year"]) if not mr.empty and not (mr.iloc[0]["year"] != mr.iloc[0]["year"]) else None
        items.append(d)
    return jsonify(movies=items)


# ── Predict ───────────────────────────────────────────────────────────────────
@app.route("/api/predict")
def api_predict():
    uid = request.args.get("user", type=int)
    mid = request.args.get("movie", type=int)
    if uid is None or mid is None:
        abort(400)
    pred = _cf.predict_rating(uid, mid)
    return jsonify(user_id=uid, movie_id=mid, predicted_rating=pred, model="SGD")


# ── Recommendations ──────────────────────────────────────────────────────────
@app.route("/api/recommend/<int:user_id>")
def api_recommend(user_id):
    n     = min(50, max(1, request.args.get("n", 10, type=int)))
    alpha = request.args.get("alpha", type=float)

    eng = _engine
    if alpha is not None:
        eng = HybridRecommender(alpha=alpha, cold_start_threshold=10, mmr_lambda=0.8)
        eng.set_models(_cf, _cb, _als)

    recs = eng.recommend(user_id, _movies, _ratings, n=n)
    items = []
    for _, row in recs.iterrows():
        d = {"movie_id": int(row["movie_id"]), "title": str(row["title"]),
             "genres": str(row["genres"]),
             "cf_score": round(float(row.get("cf_score", 0)), 3),
             "als_score": round(float(row.get("als_score", 0)), 3),
             "cb_score": round(float(row.get("cb_score", 0)), 3),
             "hybrid_score": round(float(row.get("hybrid_score", 0)), 3)}
        mr = _movies[_movies["movie_id"] == d["movie_id"]]
        d["year"] = int(mr.iloc[0]["year"]) if not mr.empty and not (mr.iloc[0]["year"] != mr.iloc[0]["year"]) else None
        items.append(d)
    return jsonify(user_id=user_id, model="hybrid", n=n, recommendations=items)


@app.route("/api/cf/<int:user_id>")
def api_cf(user_id):
    n    = min(50, max(1, request.args.get("n", 10, type=int)))
    recs = _cf.get_top_n_recommendations(user_id, _movies, _ratings, n=n)
    items = []
    for _, row in recs.iterrows():
        d = {"movie_id": int(row["movie_id"]), "title": str(row["title"]),
             "genres": str(row["genres"]),
             "predicted_rating": round(float(row["predicted_rating"]), 3)}
        mr = _movies[_movies["movie_id"] == d["movie_id"]]
        d["year"] = int(mr.iloc[0]["year"]) if not mr.empty and not (mr.iloc[0]["year"] != mr.iloc[0]["year"]) else None
        items.append(d)
    return jsonify(user_id=user_id, model="cf_sgd", n=n, recommendations=items)


@app.route("/api/als/<int:user_id>")
def api_als(user_id):
    n = min(50, max(1, request.args.get("n", 10, type=int)))
    model = _als if _als else _cf
    recs  = model.get_top_n_recommendations(user_id, _movies, _ratings, n=n)
    items = []
    for _, row in recs.iterrows():
        d = {"movie_id": int(row["movie_id"]), "title": str(row["title"]),
             "genres": str(row["genres"]),
             "predicted_rating": round(float(row["predicted_rating"]), 3)}
        mr = _movies[_movies["movie_id"] == d["movie_id"]]
        d["year"] = int(mr.iloc[0]["year"]) if not mr.empty and not (mr.iloc[0]["year"] != mr.iloc[0]["year"]) else None
        items.append(d)
    return jsonify(user_id=user_id, model="cf_als" if _als else "cf_sgd_fallback", n=n, recommendations=items)


@app.route("/api/cb/<int:user_id>")
def api_cb(user_id):
    n    = min(50, max(1, request.args.get("n", 10, type=int)))
    recs = _cb.get_user_profile_recommendations(user_id, _ratings, n=n)
    items = []
    for _, row in recs.iterrows():
        d = {"movie_id": int(row["movie_id"]), "title": str(row["title"]),
             "genres": str(row["genres"]),
             "similarity_score": round(float(row["similarity_score"]), 3)}
        mr = _movies[_movies["movie_id"] == d["movie_id"]]
        d["year"] = int(mr.iloc[0]["year"]) if not mr.empty and not (mr.iloc[0]["year"] != mr.iloc[0]["year"]) else None
        items.append(d)
    return jsonify(user_id=user_id, model="content_based", n=n, recommendations=items)


@app.route("/api/similar/<int:movie_id>")
def api_similar(movie_id):
    n    = min(30, max(1, request.args.get("n", 10, type=int)))
    sims = _cb.get_similar_movies(movie_id, n=n)
    items = []
    for _, row in sims.iterrows():
        d = {"movie_id": int(row["movie_id"]), "title": str(row["title"]),
             "genres": str(row["genres"]),
             "similarity_score": round(float(row["similarity_score"]), 3)}
        mr = _movies[_movies["movie_id"] == d["movie_id"]]
        d["year"] = int(mr.iloc[0]["year"]) if not mr.empty and not (mr.iloc[0]["year"] != mr.iloc[0]["year"]) else None
        items.append(d)
    return jsonify(movie_id=movie_id, similar=items)


# ── User Endpoints ───────────────────────────────────────────────────────────
@app.route("/api/user/<int:user_id>/history")
def api_user_history(user_id):
    ur = _ratings[_ratings["user_id"] == user_id]
    if ur.empty:
        abort(404)
    merged = ur.merge(_movies, on="movie_id").sort_values("rating", ascending=False)
    items = []
    for _, row in merged.iterrows():
        d = _movie_to_dict(row)
        d["user_rating"] = int(row["rating"])
        items.append(d)
    demo = _users[_users["user_id"] == user_id]
    user_info = {}
    if not demo.empty:
        u = demo.iloc[0]
        user_info = {"age": int(u["age"]), "gender": str(u["gender"]),
                     "occupation": str(u["occupation"])}
    return jsonify(user_id=user_id, demographics=user_info,
                   n_ratings=len(items), history=items)


@app.route("/api/user/<int:user_id>/stats")
def api_user_stats(user_id):
    ur = _ratings[_ratings["user_id"] == user_id]
    if ur.empty:
        abort(404)
    merged = ur.merge(_movies, on="movie_id")
    rating_dist = {str(int(k)): int(v) for k, v in
                   merged["rating"].value_counts().sort_index().items()}
    genre_prefs = {}
    for _, row in merged.iterrows():
        for g in str(row["genres"]).split("|"):
            if g != "unknown":
                genre_prefs.setdefault(g, []).append(float(row["rating"]))
    genre_avg = {g: round(float(sum(rs))/len(rs), 2) for g, rs in genre_prefs.items()}
    genre_avg = dict(sorted(genre_avg.items(), key=lambda x: x[1], reverse=True))

    demo = _users[_users["user_id"] == user_id]
    user_info = {}
    if not demo.empty:
        u = demo.iloc[0]
        user_info = {"age": int(u["age"]), "gender": str(u["gender"]),
                     "occupation": str(u["occupation"])}

    return jsonify(
        user_id=user_id,
        demographics=user_info,
        n_ratings=int(len(ur)),
        avg_rating=round(float(ur["rating"].mean()), 3),
        std_rating=round(float(ur["rating"].std()), 3),
        rating_distribution=rating_dist,
        genre_preferences=genre_avg,
    )


@app.route("/api/users")
def api_users():
    n = min(50, request.args.get("n", 20, type=int))
    top = (_ratings.groupby("user_id")["rating"]
           .agg(n_ratings="count", avg_rating="mean")
           .sort_values("n_ratings", ascending=False)
           .head(n).reset_index())
    items = []
    for _, row in top.iterrows():
        d = {"user_id": int(row["user_id"]),
             "n_ratings": int(row["n_ratings"]),
             "avg_rating": round(float(row["avg_rating"]), 2)}
        demo = _users[_users["user_id"] == d["user_id"]]
        if not demo.empty:
            u = demo.iloc[0]
            d.update({"age": int(u["age"]), "gender": str(u["gender"]),
                      "occupation": str(u["occupation"])})
        items.append(d)
    return jsonify(users=items)


# ── Model Comparison ──────────────────────────────────────────────────────────
@app.route("/api/compare/<int:user_id>")
def api_compare(user_id):
    n = min(20, max(1, request.args.get("n", 10, type=int)))

    # Get recs from each model
    hybrid_recs = _engine.recommend(user_id, _movies, _ratings, n=n)
    sgd_recs    = _cf.get_top_n_recommendations(user_id, _movies, _ratings, n=n)
    cb_recs     = _cb.get_user_profile_recommendations(user_id, _ratings, n=n)

    def recs_to_list(df, score_col):
        items = []
        for _, row in df.iterrows():
            d = {"movie_id": int(row["movie_id"]), "title": str(row["title"]),
                 "genres": str(row["genres"]),
                 "score": round(float(row.get(score_col, 0)), 3)}
            items.append(d)
        return items

    result = {
        "user_id": user_id,
        "hybrid": recs_to_list(hybrid_recs, "hybrid_score"),
        "sgd":    recs_to_list(sgd_recs,    "predicted_rating"),
        "cb":     recs_to_list(cb_recs,     "similarity_score"),
    }

    if _als:
        als_recs = _als.get_top_n_recommendations(user_id, _movies, _ratings, n=n)
        result["als"] = recs_to_list(als_recs, "predicted_rating")

    return jsonify(result)


# ── Confidence ────────────────────────────────────────────────────────────────
@app.route("/api/confidence")
def api_confidence():
    uid = request.args.get("user", type=int)
    mid = request.args.get("movie", type=int)
    if uid is None or mid is None:
        abort(400)
    conf = _cf.predict_with_confidence(uid, mid)
    return jsonify(user_id=uid, movie_id=mid, **conf)


# ── Fairness ──────────────────────────────────────────────────────────────────
@app.route("/api/fairness")
def api_fairness():
    global _fairness_cache
    if _fairness_cache is not None:
        return jsonify(_fairness_cache)

    # Compute on first request, then cache
    try:
        from analysis import popularity_bias_analysis, demographic_fairness, calibration_analysis
        pop = popularity_bias_analysis(_cf, _movies, _ratings, n_users=50)
        pop_clean = {k: v for k, v in pop.items() if k != "rec_counts"}
        demo = demographic_fairness(_cf, _ratings, _users)
        calib = calibration_analysis(_cf, _ratings)
        _fairness_cache = {
            "popularity_bias": pop_clean,
            "demographic_fairness": demo,
            "calibration": calib,
        }
    except Exception as e:
        _fairness_cache = {"error": str(e)}

    return jsonify(_fairness_cache)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
