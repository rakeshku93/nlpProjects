import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import naive_bayes
from sklearn.feature_extraction.text import CountVectorizer 

def run(fold):
    
    df = pd.read_csv("../inputs/IMDB_Dataset-folds.csv")
    
    df_train = df[df.kfold != fold].reset_index(drop=True)
    
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    count_vec = CountVectorizer(tokenizer=word_tokenize, token_pattern=None)
    
    count_vec.fit(df_train.review)
    
    x_train = count_vec.transform(df_train.review)
    
    x_valid = count_vec.transform(df_valid.review)
    
    model = naive_bayes.MultinomialNB()
    
    model.fit(x_train, df_train.sentiment.values)
    
    acc = model.score(x_valid, df_valid.sentiment.values)

    print(f"Fold: {fold}, Accuracy: {acc}")
    
if __name__ == "__main__":
    for fold in range(5):
        run(fold)
    