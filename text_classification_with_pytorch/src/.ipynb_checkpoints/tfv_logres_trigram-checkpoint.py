"""
Both CountVectorizer and TfidfVectorizer implementations of scikit-learn offers ngrams
by ngram_range parameter, which has a minimum and maximum limit. By default the limit is (1, 1)

Here we have included the 3-gram combination of words with TfidfVectorizer as it was giving 
better result than CountVectorizer

"""

import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model


def run(fold):

    df = pd.read_csv("../inputs/IMDB_Dataset-folds.csv")

    df_train = df[df.kfold != fold].reset_index(drop=True)

    df_valid = df[df.kfold == fold].reset_index(drop=True)

    tfv_trigram = TfidfVectorizer(
        tokenizer=word_tokenize,
        token_pattern=None,
        ngram_range=(1, 3)
    )

    tfv_trigram.fit(df_train.review)

    x_train = tfv_trigram.transform(df_train.review)

    x_valid = tfv_trigram.transform(df_valid.review)

    model = linear_model.LogisticRegression()

    model.fit(x_train, df_train.sentiment.values)

    yhat = model.predict(x_valid)

    acc = metrics.accuracy_score(yhat, df_valid.sentiment.values)

    print(f"Fold: {fold}, Accuracy: {acc}")


if __name__ == "__main__":
    for fold in range(5):
        run(fold)


