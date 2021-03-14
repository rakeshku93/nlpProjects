import pandas as pd
from sklearn import model_selection

def create_folds(path):
    
    data = pd.read_csv(path)
    
    # create a new column "kfold" & fill it with -1    
    data["kfold"] = -1
    
    # randomized the dataset
    data = data.sample(frac=1).reset_index(drop=True)
    
    # map positive to 1 and negative to 0
    data.sentiment = data.sentiment.apply(
        lambda x: 1 if x=="positive" else 0  
    )
    
    # fetch the target 
    y = data.sentiment.values
        
    stratified_kf = model_selection.StratifiedKFold(n_splits=5)
    
    for fold, (t_, v_) in enumerate(stratified_kf.split(X=data, y=y)):
        data.loc[v_, "kfold"] = fold
        
    return data.to_csv("../inputs/IMDB_Dataset-folds.csv",
                       index=False)
    
if __name__ == "__main__":
    path = "../inputs/IMDB Dataset.csv"
    create_folds(path)