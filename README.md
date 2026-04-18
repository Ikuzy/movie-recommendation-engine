# Movie Recommendation Engine (Kaggle X ALX)

> Recommendation system built in Python on 5M records using scikit-surprise — ranked **48th out of 266 teams (top 17%)** with a final RMSE of **0.81** on Kaggle.

---

## Description

This project tackles the challenge of building an intelligent movie recommendation system — the kind of algorithm that powers Netflix, Amazon Prime, Disney+, and similar platforms to suggest content users are likely to enjoy.

The goal was to predict how a user would rate a movie they have not yet seen, based on their historical viewing and rating preferences. This is a classic **collaborative filtering** problem evaluated using **Root Mean Square Error (RMSE)**.

> *"Ever wondered how Netflix somehow knows what to recommend to you? It's not just a guess drawn out of a hat. There is an algorithm behind it."*

---

## Challenge Overview

| Detail | Value |
|--------|-------|
| Platform | Kaggle |
| Type | Collaborative Filtering / Recommendation System |
| Dataset size | ~5 million records |
| Evaluation metric | RMSE (Root Mean Square Error) |
| Final RMSE score | **0.81** |
| Final ranking | **48th out of 266 participants (top 17%)** |

---

## Model Progression

This project followed an iterative approach — starting with simpler models and progressively improving performance:

| # | Model | RMSE | Notes |
|---|-------|------|-------|
| 1 | KNN Item-based (run 1) | 0.90 | Baseline — default hyperparameters |
| 2 | KNN Item-based (run 2) | 0.87 | Tuned similarity metric and neighbors |
| 3 | SVD++ | 0.85 | Matrix factorization with implicit feedback |
| 4 | **SVD (final)** | **0.81** | Best result — submitted to Kaggle |

Each iteration brought a meaningful improvement, with SVD delivering the best generalization on unseen user-movie pairs.

---

## Approach

### 1. Data Loading & Preprocessing
- Loaded ~5M user-movie rating records from CSV using Pandas
- Extracted rating scale dynamically from training data
- Built full trainset using scikit-surprise's `Dataset` and `Reader`

### 2. Model Exploration

**KNN Item-based (Baseline)**
Started with item-based KNN collaborative filtering to establish a baseline. Tuned the similarity metric and number of neighbors across two runs, improving RMSE from 0.90 to 0.87.

**SVD++**
Moved to SVD++ which incorporates both explicit ratings and implicit feedback signals. Configured with:
- `n_factors = 100`
- `n_epochs = 20`
- `lr_all = 0.005`
- `reg_all = 0.02`

This improved RMSE to 0.85.

**SVD (Final Model)**
Standard SVD with optimized hyperparameters delivered the best RMSE of **0.81** and was used for the final Kaggle submission.

### 3. Cold-Start Handling
A key challenge in recommendation systems is handling users or movies with no prior history. A fallback prediction strategy was implemented:

```python
# Priority order for cold-start fallback:
# 1. Average of user mean + movie mean
# 2. User mean only (if movie unknown)
# 3. Movie mean only (if user unknown)
# 4. Global mean (complete cold start)
```

### 4. Memory-Efficient Prediction
Given the scale of the test set (5M records), predictions were generated in **chunks of 100,000 rows** to avoid memory overflow:

```python
reader = pd.read_csv(TEST_PATH, chunksize=100_000)
for chunk in reader:
    preds = predict_chunk(chunk)
```

### 5. Submission Generation
Output was written directly to CSV in Kaggle's required format:

```csv
Id,rating
1_2011,3.5
42_318,4.2
```

Where `Id = userID_movieID`.

---

## Tech Stack

| Tool | Usage |
|------|-------|
| Python | Core language |
| Pandas | Data loading, preprocessing, chunked I/O |
| NumPy | Numerical computations, fallback handling |
| scikit-surprise | SVD, SVD++, KNN collaborative filtering models |
| Jupyter Notebook | Development and experimentation |

---

## Results

| Metric | Score |
|--------|-------|
| RMSE | **0.81** |
| Rank | **48 / 266** |
| Percentile | **Top 17%** |

---

## Project Structure

```
movie-recommendation-engine/
├── data/                        # Dataset files (not tracked in git)
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── EDA.ipynb                # Exploratory data analysis
├── recommender.py               # Final model script (SVD)
├── submissions/
│   └── submission_svd.csv       # Final Kaggle submission
├── requirements.txt
└── README.md
```

---

## Installation & Usage

### Requirements
- Python 3.8+
- pip

### Setup

```bash
git clone https://github.com/ikuzy/movie-recommendation-engine.git
cd movie-recommendation-engine
pip install -r requirements.txt
```

### Run

```bash
python recommender.py
```

This will:
1. Load `train.csv` and `test.csv` from the current directory
2. Train the SVD model on the full training set
3. Generate predictions with cold-start fallback
4. Write `submission_svd.csv` ready for Kaggle upload

---

## Key Technical Decisions

**Why SVD over SVD++?**
SVD++ uses implicit feedback in addition to explicit ratings, which theoretically makes it more powerful. However, on this dataset SVD generalized better — likely because the implicit feedback signal was too sparse to provide a meaningful advantage, while adding noise that hurt RMSE.

**Why chunked prediction?**
Loading 5M test records into memory at once causes memory overflow on standard hardware. Processing in 100K-row chunks keeps memory usage stable without sacrificing speed.

**Why a cold-start fallback?**
Some users or movies in the test set had no training history. Without a fallback, these would default to the global mean, which is suboptimal. The hierarchical fallback (user mean → movie mean → global mean) produces more accurate estimates for sparse cases.

---

## Resources

- [Kaggle Profile — oussamazouine](https://www.kaggle.com/oussamazouine)
- [scikit-surprise Documentation](https://surprise.readthedocs.io/en/stable/)
- [SVD for Collaborative Filtering — Simon Funk](https://sifter.org/simon/journal/20061211.html)
- [Matrix Factorization Techniques — Koren et al.](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf)
- [KNN for Collaborative Filtering](https://surprise.readthedocs.io/en/stable/knn_inspired.html)

---

## Author

| Login | GitHub | Kaggle |
|-------|--------|--------|
| Ozouine | [@ikuzy](https://github.com/ikuzy) | [oussamazouine](https://www.kaggle.com/oussamazouine) |
