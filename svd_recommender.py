import argparse

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Movie recommender using Surprise SVD (non-SVD++).')
    parser.add_argument('--train-path', default='train.csv')
    parser.add_argument('--test-path', default='test.csv')
    parser.add_argument('--submission-path', default='submission_svd.csv')
    parser.add_argument('--chunksize', type=int, default=100_000)
    parser.add_argument('--n-factors', type=int, default=150)
    parser.add_argument('--n-epochs', type=int, default=30)
    parser.add_argument('--lr-all', type=float, default=0.005)
    parser.add_argument('--reg-all', type=float, default=0.02)
    parser.add_argument('--random-state', type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print('Loading training data...')
    df_train = pd.read_csv(
        args.train_path,
        usecols=['userId', 'movieId', 'rating'],
        dtype={'userId': np.int32, 'movieId': np.int32, 'rating': np.float32},
    )

    min_rating = float(df_train['rating'].min())
    max_rating = float(df_train['rating'].max())
    reader = Reader(rating_scale=(min_rating, max_rating))

    data = Dataset.load_from_df(df_train[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()

    print('Training SVD model...')
    svd = SVD(
        n_factors=args.n_factors,
        n_epochs=args.n_epochs,
        lr_all=args.lr_all,
        reg_all=args.reg_all,
        biased=True,
        random_state=args.random_state,
        verbose=True,
    )
    svd.fit(trainset)

    user_means = df_train.groupby('userId')['rating'].mean()
    movie_means = df_train.groupby('movieId')['rating'].mean()
    global_mean = float(df_train['rating'].mean())

    def predict_chunk(chunk: pd.DataFrame) -> list[str]:
        preds = []
        for _, row in chunk.iterrows():
            user = int(row['userId'])
            movie = int(row['movieId'])
            key = f'{user}_{movie}'

            pred = svd.predict(user, movie).est
            if np.isnan(pred):
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

            pred = float(np.clip(pred, min_rating, max_rating))
            preds.append(f'{key},{pred:.4f}')
        return preds

    print('Predicting and writing submission...')
    test_reader = pd.read_csv(
        args.test_path,
        usecols=['userId', 'movieId'],
        dtype={'userId': np.int32, 'movieId': np.int32},
        chunksize=args.chunksize,
    )

    with open(args.submission_path, 'w') as f:
        f.write('Id,rating\n')
        for chunk in test_reader:
            preds = predict_chunk(chunk)
            f.write('\n'.join(preds) + '\n')

    print('Done! Submission written to', args.submission_path)


if __name__ == '__main__':
    main()
