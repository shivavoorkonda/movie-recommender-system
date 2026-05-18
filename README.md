# CineAI — Hybrid Movie Recommender System

> **Full-Stack AI project** · Flask REST API + Dark-mode SPA Frontend · MovieLens 100K · Matrix Factorisation SGD + TF-IDF

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python) ![Flask](https://img.shields.io/badge/Flask-3.x-green?logo=flask) ![ML](https://img.shields.io/badge/ML-Matrix%20Factorisation-purple) ![Tests](https://img.shields.io/badge/Tests-31%20passed-success)

---

## What This Project Does

CineAI is a **hybrid movie recommender system** that combines two complementary ML techniques — just like Netflix does — to deliver personalised movie recommendations:

| Method | Algorithm | What it learns |
|--------|-----------|----------------|
| **Collaborative Filtering** | Matrix Factorisation (SGD) | User taste from shared rating patterns |
| **Content-Based** | TF-IDF + Cosine Similarity | Movie genre/feature similarity |
| **Hybrid Fusion** | Weighted blend α·CF + (1-α)·CB | Best of both, with cold-start handling |

---

## Live Demo

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (downloads dataset automatically)
python src/main.py --no-plots

# 3. Launch web app
start_web.bat          # Windows
# OR
python web/app.py      # Any OS
```

Open **http://localhost:5000** — the full UI loads instantly.

---

## Project Structure

```
movie_recommender_system/
│
├── src/                             # Core ML pipeline
│   ├── data_loader.py               #   Download & preprocess MovieLens 100K
│   ├── collaborative_filtering.py   #   Matrix Factorisation via SGD (NumPy)
│   ├── content_based.py             #   TF-IDF genres + Cosine Similarity
│   ├── hybrid_engine.py             #   Weighted fusion + cold-start handling
│   ├── evaluator.py                 #   RMSE, MAE, Precision@K, Recall@K, ILD
│   ├── utils.py                     #   Pretty printing & helpers
│   └── main.py                      #   CLI pipeline entry point
│
├── web/                             # Full-stack web application
│   ├── app.py                       #   Flask REST API (12 endpoints)
│   ├── templates/index.html         #   Dark-mode SPA (single-page app)
│   └── static/
│       ├── css/style.css            #   Netflix-style dark UI
│       └── js/app.js                #   Vanilla JS frontend logic
│
├── tests/
│   └── test_recommender.py          # 31 pytest unit tests
│
├── data/processed/                  # Cleaned CSVs (auto-generated)
├── models/saved/                    # Trained models — .pkl (auto-generated)
├── outputs/plots/                   # Evaluation charts (auto-generated)
│
├── requirements.txt
├── .gitignore
├── start_web.bat                    # Windows: launch web server
├── run.bat                          # Windows: run CLI pipeline
└── run_tests.bat                    # Windows: run all tests
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI |
| GET | `/health` | Server health check |
| GET | `/api/stats` | Dataset statistics |
| GET | `/api/popular?n=10` | Popularity-ranked movies |
| GET | `/api/recommend/<user_id>?n=10&alpha=0.7` | **Hybrid** recommendations |
| GET | `/api/cf/<user_id>?n=10` | Collaborative filtering only |
| GET | `/api/cb/<user_id>?n=10` | Content-based only |
| GET | `/api/similar/<movie_id>?n=10` | Similar movies |
| GET | `/api/search?q=star&n=20` | Search by title |
| GET | `/api/user/<user_id>/history` | User rating history |
| GET | `/api/movies/<movie_id>` | Movie details + stats |
| GET | `/api/users` | Top active users |

---

## Model Performance

| Metric | Value |
|--------|-------|
| Test RMSE | **0.9361** |
| Test MAE | **0.7372** |
| Precision@10 | **57.6%** |
| Recall@10 | **71.4%** |
| F1@10 | **63.8%** |
| Intra-List Diversity | 0.210 |
| Training time | ~14 seconds |
| Test suite | **31 / 31 passed** |

---

## How the Hybrid Engine Works

```
Rating Matrix R (users × movies)
         ↓  Matrix Factorisation (SGD)
R ≈ P · Qᵀ + bias_user + bias_item + global_mean

Movie genres → TF-IDF vectors → Cosine Similarity matrix

hybrid_score = α × norm(CF_predicted_rating)
             + (1-α) × norm(CB_similarity_score)

Cold-start: if user has < 10 ratings → α is scaled down
            so content-based contributes more
```

---

## Dataset

**MovieLens 100K** — [GroupLens Research](https://grouplens.org/datasets/movielens/100k/)
- 100,000 ratings · 943 users · 1,682 movies
- Rating scale: 1–5 stars · Matrix sparsity: 93.7%
- **Auto-downloaded** on first run — no manual download needed

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Core | NumPy (custom SGD), scikit-learn (TF-IDF, cosine sim) |
| Data | Pandas, Matplotlib, Seaborn |
| Backend | Flask, joblib |
| Frontend | Vanilla JS, CSS3 (dark mode, animations) |
| Testing | pytest (31 tests) |
| Dataset | MovieLens 100K (GroupLens) |

---

## Running CLI Pipeline

```bash
# Full training + evaluation + plots
set PYTHONIOENCODING=utf-8 && python src/main.py

# Fast re-run with saved models
python src/main.py --skip-train --user 42

# Get movies similar to a title
python src/main.py --movie "Star Wars" --skip-train

# Top movies for new user (no history)
python src/main.py --new-user --skip-train

# Run tests
python -m pytest tests/ -v
```

---

*Built as a portfolio project demonstrating end-to-end ML engineering — from data preprocessing through model training, REST API design, and full-stack web development.*
