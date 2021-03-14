import io
import torch

import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn import metrics
from torch import nuclear_norm

import config
import dataset
import engine
import lstm

# This function will return embedding_dict with words as key & embedding vectors as value


def load_vectors(fname):

    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

# embedding_dict from fast-text word embeddings


def create_embedding_matrix(word_index, embedding_dict):
    """
    This function creates the embedding matrix.
    * * :param word_index: a dictionary with word:index_value -- {"was": 101, "are": 89,}
    ??  :param embedding_dict: a dictionary with word : embedding_vector
        :return: a numpy array with embedding vectors for all known words 
    """

    # initialize matrix with zeros
    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    # loop over all the words
    for word, idx in word_index.items():
        # if word is found in pre-trained embeddings, update the matrix, else the vector is zeros!
        if word in embedding_dict:
            embedding_matrix[idx] = embedding_dict[word]

    return embedding_matrix


def run(df, fold):
    """
    Run training and validation for a given fold & dataset

    :param df: pandas dataframe with kfold column
    :param fold: current fold, int
    """
    # fetch training dataframe
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # fetch validation dataframe

    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    
    tokenizer.fit_on_texts(df_train.review.values)
    
    x_train = tokenizer.texts_to_sequences(df_train.review.values)
        
    x_valid = tokenizer.texts_to_sequences(df_valid.review.values)  
    
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=config.MAXLEN)  
    
    x_valid = tf.keras.preprocessing.sequence.pad_sequences(x_valid, maxlen=config.MAXLEN)
    
    #* embedding_dict: dictionary with word:embedding_vectors
    embedding_dict = load_vectors("../input/wiki-news-300d-1M.vec/wiki-news-300d-1M.vec")
    
    #* word_index: dictionary with word:idx --  {'the': 1, 'cat': 2, 'sat': 3, 'on': 4}
    word_index = tokenizer.word_index  
        
    #* embedding matrix: a dictionary with idx:embedding_vector 
    embedding_matrix = create_embedding_matrix(
                            word_index, 
                            embedding_dict
            )
    
    model = lstm.LSTM(embedding_matrix)

    optimizer = torch.optim.Adam(model.parameters, lr=1e-3)

    # check if GPU is available else run on CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    
    train_dataset = dataset.IMDBDataset(
        reviews = x_train,
        targets = df_train.sentiment.values)

    valid_dataset = dataset.IMDBDataset(
        reviews = x_valid,
        targets = df_valid.sentiment.values)

    train_data_loader = torch.utils.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    valid_data_loader = torch.utils.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=2
    )

    best_accuracy = 0
    early_stopping_counter = 0
    for epoch in range(config.EPOCHS):

        engine.train(train_data_loader, model, optimizer, device)

        preds, targets = engine.evaluate(
            valid_data_loader, model, optimizer, device)

        print(f"preds----{preds}")

        preds = np.array(preds) >= 0.5

        accuracy = metrics.accuracy_score(preds, targets)

        print(f"Fold:{fold}, Epoch: {epoch},  Accuracy: {accuracy}")

        # simple early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy

        else:
            early_stopping_counter += 1

        if early_stopping_counter > 2:
            break


if __name__ == "__main__":
    for fold in range(5):
        print(f"Running on fold: {fold}")
        df = pd.read_csv("../input/IMDB_Dataset-folds.csv")
        run(df, fold)