import torch
from torch import avg_pool1d, max_pool1d
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, embedding_matrix) -> None:
        """
        : param embedding_matrix : numpy array with vectors for all words
        
        """
        super(LSTM, self).__init__()
        
        # number of words = number of rows in embedding_matrix 
        num_words = embedding_matrix.shape[0]
        
        # embedding dimension = number of columns in embedding_matrix
        embed_dim = embedding_matrix.shape[1]
        
        self.embedding = nn.Embedding(
                num_embeddings = num_words,
                embedding_dim = embed_dim,
        )
                
        self.embedding.weight = nn.Parameter(
            torch.tensor(
                embedding_matrix,
                dtype=torch.float32,
            )
        )
        
        # don't want to train the pre-trained embedding 
        self.embedding.weight.requires_grad = False
        
        # define a bi-directional LSTM layer
        self.lstm = nn.LSTM(
                input_size=embed_dim,
                hidden_size=128,
                bidirectional=True,
                batch_first=True
        )
        
        # define a output layer
        self.out = nn.Linear(512, 1)
        
        
    def forward(self, x):
        
        x = self.embedding(x)
        
        x, _ = self.lstm(x)
        
        avg_pool = torch.mean(x, 1)
        max_pool = torch.max(x, 1)
        
        out = torch.cat((avg_pool, max_pool), 1)
        
        print(f"output shape---{type(out)}--{out}")
        
        # pass through the output layer & return the output
        out = self.out(out)
        
        # return linear output
        return out
        
        
        
        
        
        
        

