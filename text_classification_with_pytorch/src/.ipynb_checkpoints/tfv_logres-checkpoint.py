import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

def run(fold):
    
    df = pd.read_csv("../inputs/IMDB_Dataset-folds.csv")
    
    df_train = df[df.kfold!= fold].reset_index(drop=True)
    
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    tfv = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)
    
    tfv.fit(df_train.review)
    
    x_train = tfv.transform(df_train.review)
    
    x_valid = tfv.transform(df_valid.review)
    
    model = linear_model.LogisticRegression()
    
    model.fit(x_train, df_train.sentiment.values)
    
    yhat = model.predict(x_valid)
    
    acc = metrics.accuracy_score(yhat, df_valid.sentiment.values)
    
    print(f"Fold: {fold}, Accuracy:{acc}")
    
if __name__ == "__main__":
    for fold in range(5):
        run(fold)