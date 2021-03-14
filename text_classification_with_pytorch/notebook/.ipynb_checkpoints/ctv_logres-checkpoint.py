import pandas as pd
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize


def run(fold):
    # read the training data
    df = pd.read_csv("../inputs/IMDB_Dataset-folds.csv")
    
    # get training data using folds
    df_train = df[df.kfold !=fold].reset_index(drop=True)
    
    # get validation data using folds
    df_valid = df[df.kfold ==fold].reset_index(drop=True)
    
    # initialize CountVectorizer with NLTK's word_tokenize function as tokenizer
    count_vec = CountVectorizer(tokenizer=word_tokenize, token_pattern=None)  
    
    # fit count_vec on training data reviews
    count_vec.fit(df_train.review)
    
    x_train = count_vec.transform(df_train.review)
    
    
    x_valid = count_vec.transform(df_valid.review) 
    
    # initalize LogisticRegression model
    model = linear_model.LogisticRegression()
    
    model.fit(x_train, df_train.sentiment.values)
    
    accuracy = model.score(x_valid, df_valid.sentiment.values)
    
     # print auc at each fold
    print(f"Fold = {fold}, Accuracy = {accuracy}")
    
    
if __name__ == "__main__":
    for fold in range(5):
        run(fold)