# CineAI — Advanced Hybrid Movie Recommender System

> **Portfolio-Grade ML Engineering Showcase** · Multi-Model Ensemble (SGD + ALS + TF-IDF) · Fairness & Popularity Bias Auditing · Interactive Model Comparison Suite · Flask REST API + Custom Dark-Mode Vanilla SPA

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![Flask](https://img.shields.io/badge/Flask-3.x-green?logo=flask) ![ML](https://img.shields.io/badge/Algorithms-Ensemble%20%7C%20SGD%20%7C%20ALS%20%7C%20MMR-purple) ![License](https://img.shields.io/badge/License-MIT-orange)

---

## 🌐 Live Production Deployment

The project is deployed across a high-performance **Split Architecture**:
*   🎨 **Interactive Frontend SPA (Vercel):** [https://movie-recommender-system-delta-rouge.vercel.app/](https://movie-recommender-system-delta-rouge.vercel.app/)
*   🧠 **Advanced REST API Backend (Render):** [https://cineai-backend-f3sc.onrender.com](https://cineai-backend-f3sc.onrender.com)

---

## 🌟 Overview & Core Engineering

CineAI is an advanced **hybrid movie recommender system** designed to solve classic information retrieval and machine learning challenges in recommendation systems:
*   **Cold Start Mitigation:** Incorporates dynamic demographic grouping and alpha weighting fallbacks when user profile sparsity is high.
*   **Accuracy-Diversity Tradeoff:** Employs Maximal Marginal Relevance (MMR) to diversify list rankings and counter echo-chamber recommendations.
*   **Algorithmic Auditing:** Audits popularity bias (Gini coefficients), demographically stratified evaluation (RMSE differences), and calibration checks.
*   **Model Ensembling:** Blends Matrix Factorization trained via Stochastic Gradient Descent (SGD) and Alternating Least Squares (ALS) alongside TF-IDF Content-Based filtering.

---

## 📦 Project Architecture

```
movie_recommender_system/
│
├── src/                             # Core ML & Recommendation Pipeline
│   ├── data_loader.py               #   Dataset loading, demographic feature engineering
│   ├── collaborative_filtering.py   #   SGD Matrix Factorization & ALS Matrix Factorization
│   ├── content_based.py             #   TF-IDF Title/Genres vectorization + Cosine Similarity
│   ├── hybrid_engine.py             #   Dynamic ensembling & MMR re-ranking
│   ├── evaluator.py                 #   Advanced metrics (NDCG@K, MAP@K, HR@K, ILD, coverage)
│   ├── tuner.py                     #   Random search Bayesian hyperparameter optimization
│   ├── analysis.py                  #   Fairness, bias, calibration, & demographic audit engine
│   └── main.py                      #   Unified CLI entrypoint for orchestrating pipelines
│
├── web/                             # Full-Stack Web Application
│   ├── app.py                       #   Flask REST API serving predictions & live audits
│   ├── templates/index.html         #   Glassmorphism Single-Page Application (SPA) HTML5
│   └── static/
│       ├── css/style.css            #   Netflix-style dark mode styling, skeleton animations
│       └── js/app.js                #   Reactive SPA controller handling state & dynamic comparisons
│
├── tests/
│   └── test_recommender.py          #   Pytest suite covering models, metrics, and tuning
│
├── requirements.txt                 #   Project dependencies
├── start_web.bat                    #   Windows script to launch Flask web app
├── run.bat                          #   Windows script to run the full training pipeline
└── run_tests.bat                    #   Windows script to trigger test suite
```

---

## 🛠️ Installation & Getting Started

### 1. Clone & Setup Workspace
Ensure Python 3.10+ is installed:
```bash
# Install required libraries
pip install -r requirements.txt
```

### 2. Run the Pipelines (CLI)
The unified pipeline in `src/main.py` handles model training, parameter tuning, metric evaluation, and fairness analysis:

```bash
# Run full pipeline (Train, tune hyperparameters, audit fairness, write evaluation reports)
python src/main.py --full

# Train models and run evaluations on a specific user
python src/main.py --user 42

# Execute Hyperparameter Optimization (Random Search Cross-Validation)
python src/main.py --tune

# Audit system fairness, bias, and calibration errors
python src/main.py --analyse
```

### 3. Spin Up Web App
```bash
# Spin up Flask app
python web/app.py
```

---

## 📊 Evaluation & Metrics Glossary

CineAI evaluates recommendation quality using multiple ranking and diversity metrics:

*   **NDCG@K (Normalized Discounted Cumulative Gain):** Measures ranking quality, penalizing relevant items placed lower in recommendation lists.
*   **MAP@K (Mean Average Precision):** Evaluates overall precision of recommendations up to cut-off point $K$.
*   **Hit Rate@K:** Proportion of test items successfully predicted within the top $K$ suggestions.
*   **ILD (Intra-List Diversity):** Average genre distance between recommended items to prevent lists from containing duplicate themes.
*   **Catalogue Coverage:** Percentage of catalog movies recommended to at least one user to audit "rich-get-richer" effects.
*   **Novelty Score:** Self-information based metric evaluating the surprise factor of suggestions.

---

## ⚖️ Fairness & Bias Auditing

To ensure recommendations are fair and calibrated, the `analysis.py` module evaluates:

1.  **Demographic Equity:** Stratified RMSE validation across age ranges and gender splits to check if the model favors specific demographic subsets.
2.  **Popularity Domination (Gini Coefficient):** Computes inequality curves on movie recommendation frequencies, proving the effectiveness of popularity-penalty tuning.
3.  **Calibration Discrepancy:** Segments predicted vs. actual user rating averages into buckets to check if the system misrepresents user expectations.

---

## 🔌 API Documentation

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **GET** | `/api/stats` | Dataset statistics (counts, distributions) |
| **GET** | `/api/popular?n=N` | Top $N$ popularity-ranked movies |
| **GET** | `/api/recommend/<uid>?n=N&alpha=A` | Hybrid recommendations for user `uid` |
| **GET** | `/api/cf/<uid>?n=N` | Collaborative filtering recommendations (SGD) |
| **GET** | `/api/als/<uid>?n=N` | Collaborative filtering recommendations (ALS) |
| **GET** | `/api/cb/<uid>?n=N` | Content-based recommendations (TF-IDF similarity) |
| **GET** | `/api/compare/<uid>?n=N` | Side-by-side comparative list across all 4 modes |
| **GET** | `/api/similar/<mid>?n=N` | Similar item retrieval using Cosine Similarity |
| **GET** | `/api/user/<uid>/history` | Historical logs of rated movies for profile debugging |
| **GET** | `/api/confidence?user=U&movie=M` | Bootstrap prediction uncertainty estimation & confidence interval |
| **GET** | `/api/fairness` | System-wide demographic RMSE, Gini inequality, and calibration metrics |

---

## 🧪 Testing

The project maintains a unit test suite testing recommendation logic, matrix factorization shapes, metrics calculation, and parameter optimization.

```bash
# Run tests via pytest
python -m pytest tests/ -v
```

---

*CineAI was built as an end-to-end recommendation systems portfolio project, showcasing multi-model ensembling, algorithmic fairness auditing, statistical uncertainty estimation, and responsive UI layout design.*
