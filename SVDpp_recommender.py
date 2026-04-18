import pandas as pd
from surprise import Dataset, Reader, SVDpp
import numpy as np

TRAIN_PATH = 'train.csv'
TEST_PATH = 'test.csv'
SUBMISSION_PATH = 'submission_svdpp.csv'
CHUNKSIZE = 100_000

print('Loading training data...')
df_train = pd.read_csv(TRAIN_PATH)
reader = Reader(rating_scale=(df_train['rating'].min(), df_train['rating'].max()))
data = Dataset.load_from_df(df_train[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

print('Training SVD++ model...')
svdpp = SVDpp(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, verbose=True)
svdpp.fit(trainset)

user_means = df_train.groupby('userId')['rating'].mean()
movie_means = df_train.groupby('movieId')['rating'].mean()
global_mean = df_train['rating'].mean()

def predict_chunk(chunk):
    preds = []
    for idx, row in chunk.iterrows():
        user = row['userId']
        movie = row['movieId']
        key = f"{user}_{movie}"
        try:
            pred = svdpp.predict(user, movie).est
        except Exception:
            user_mean = user_means.get(user, np.nan)
            movie_mean = movie_means.get(movie, np.nan)
            if not np.isnan(user_mean) and not np.isnan(movie_mean):
                pred = (user_mean + movie_mean) / 2
            elif not np.isnan(user_mean):
                pred = user_mean
            elif not np.isnan(movie_mean):
                pred = movie_mean
            else:
                pred = global_mean
        preds.append(f"{key},{pred:.4f}")
    return preds

print('Predicting and writing submission...')
reader = pd.read_csv(TEST_PATH, chunksize=CHUNKSIZE)
with open(SUBMISSION_PATH, 'w') as f:
    f.write('Id,rating\n')
    for chunk in reader:
        preds = predict_chunk(chunk)
        f.write('\n'.join(preds) + '\n')

print('Done! Submission written to', SUBMISSION_PATH)
