# Movie Recommendation Engine (Kaggle X ALX)

> Collaborative filtering recommendation system built in Python on 5M records using scikit-surprise — ranked **48th out of 266 teams (top 17%)** with a final RMSE of **0.81** on Kaggle.

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

Each iteration brought a meaningful improvement. SVD with tuned hyperparameters delivered the best generalization on unseen user-movie pairs.

---

## Approach

### 1. Data Loading & Preprocessing

Both scripts load the training data efficiently using typed columns to minimize memory footprint:

```python
df_train = pd.read_csv(
    args.train_path,
    usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': np.int32, 'movieId': np.int32, 'rating': np.float32},
)
```

The rating scale is extracted dynamically from the data and passed to scikit-surprise's `Reader`.

---

### 2. Model Exploration

#### KNN Item-based (Baseline)
Started with item-based KNN collaborative filtering to establish a baseline. Tuned the similarity metric and number of neighbors across two runs, improving RMSE from **0.90 → 0.87**.

---

#### SVD++ (`recommender.py`)
SVD++ incorporates both explicit ratings and implicit feedback signals (items a user has interacted with, regardless of rating).

**Hyperparameters:**
```python
SVDpp(
    n_factors=100,
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    verbose=True
)
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `n_factors` | 100 | Number of latent factors |
| `n_epochs` | 20 | Training iterations |
| `lr_all` | 0.005 | Learning rate |
| `reg_all` | 0.02 | Regularization to prevent overfitting |

Achieved RMSE: **0.85**

---

#### SVD — Final Model (`svd_recommender.py`)
Standard biased SVD with stronger hyperparameters and a cleaner, production-ready implementation. This was the final submitted model.

**Hyperparameters:**
```python
SVD(
    n_factors=150,
    n_epochs=30,
    lr_all=0.005,
    reg_all=0.02,
    biased=True,
    random_state=42,
    verbose=True
)
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `n_factors` | 150 | More latent factors than SVD++ |
| `n_epochs` | 30 | More training iterations |
| `biased` | True | Includes user and item bias terms |
| `random_state` | 42 | Reproducibility |

Achieved RMSE: **0.81** ✅

---

### 3. Cold-Start Handling

A hierarchical fallback strategy handles users or movies with no training history:

```python
# Priority order:
# 1. Average of user mean + movie mean (both known)
# 2. User mean only (movie unknown)
# 3. Movie mean only (user unknown)
# 4. Global mean (complete cold start)
```

The SVD script additionally clips predictions to the valid rating range:

```python
pred = float(np.clip(pred, min_rating, max_rating))
```

This prevents out-of-range predictions that would artificially inflate RMSE.

---

### 4. Memory-Efficient Chunked Prediction

Given the scale of the test set (~5M records), predictions are generated in chunks of 100,000 rows to avoid memory overflow:

```python
test_reader = pd.read_csv(TEST_PATH, chunksize=100_000)
with open(submission_path, 'w') as f:
    f.write('Id,rating\n')
    for chunk in test_reader:
        preds = predict_chunk(chunk)
        f.write('\n'.join(preds) + '\n')
```

---

### 5. CLI Support (SVD script)

The final SVD script supports full command-line configuration — no need to edit the source code to change hyperparameters:

```bash
python svd_recommender.py \
  --train-path train.csv \
  --test-path test.csv \
  --n-factors 150 \
  --n-epochs 30 \
  --submission-path submission_svd.csv
```

---

## Tech Stack

| Tool | Usage |
|------|-------|
| Python | Core language |
| Pandas | Data loading, preprocessing, chunked I/O |
| NumPy | Numerical computations, dtype optimization, clipping |
| scikit-surprise | SVD, SVD++, KNN collaborative filtering models |
| argparse | CLI interface for the final SVD script |
| Jupyter Notebook | Exploration and EDA |

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
├── recommender.py               # SVD++ model script
├── svd_recommender.py           # SVD final model script (best result)
├── submissions/
│   ├── submission_svdpp.csv     # SVD++ submission
│   └── submission_svd.csv      # Final SVD submission (RMSE 0.81)
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

### Run SVD++ model

```bash
python recommender.py
```

### Run SVD final model (best result)

```bash
python svd_recommender.py
```

With custom hyperparameters:

```bash
python svd_recommender.py --n-factors 150 --n-epochs 30 --submission-path my_submission.csv
```

---

## Key Technical Decisions

**Why SVD over SVD++?**
SVD++ incorporates implicit feedback alongside explicit ratings, making it theoretically more powerful. However, on this dataset standard SVD generalized better — the implicit feedback signal was too sparse to provide a meaningful advantage and introduced noise that hurt RMSE. SVD with more factors (150 vs 100) and more epochs (30 vs 20) compensated for the lack of implicit signals.

**Why chunked prediction?**
Loading 5M test records into memory at once causes memory overflow on standard hardware. Processing in 100K-row chunks keeps memory stable without sacrificing speed or accuracy.

**Why clip predictions?**
Matrix factorization can occasionally produce predictions outside the valid rating range. Clipping to `[min_rating, max_rating]` ensures all outputs are valid, reducing RMSE on edge cases.

**Why typed columns?**
Using `np.int32` and `np.float32` instead of default Python types reduces memory usage by roughly 50% — critical when working with 5M records.

---

## Resources

- [Kaggle Profile — oussamazouine](https://www.kaggle.com/oussamazouine)
- [scikit-surprise Documentation](https://surprise.readthedocs.io/en/stable/)
- [SVD for Collaborative Filtering — Simon Funk](https://sifter.org/simon/journal/20061211.html)
- [Matrix Factorization Techniques — Koren et al.](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf)
- [KNN Collaborative Filtering — surprise docs](https://surprise.readthedocs.io/en/stable/knn_inspired.html)
- [SVD++ Paper — Koren 2008](https://dl.acm.org/doi/10.1145/1401890.1401944)

---

## Author

| Login | GitHub | Kaggle |
|-------|--------|--------|
| Ozouine | [@ikuzy](https://github.com/ikuzy) | [oussamazouine](https://www.kaggle.com/oussamazouine) |
