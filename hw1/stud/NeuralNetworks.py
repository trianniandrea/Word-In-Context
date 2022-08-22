# This module contains the classes that define the various models and architectures used in homework.
# The first class implements a simple MLP architecture, the second uses Bidirectional LSTM cells.
# Each of these classes is saved as a field in the StudentModel object defined in Implementation.py.
# It is a choice that makes the code structure more flexible. So I can detach and attach models simply by editing a field in the StudentModel object.


import torch
import torch.nn as nn
from typing import *


class MLPClassifier(nn.Module):

    def __init__(self, input_dim: int, embedding_dim: int, vocab_size: int,
                 padding_idx: int, pretrained_embedding: List[List], freeze: bool=True):

        super(MLPClassifier, self).__init__()

        self.emb = nn.Embedding( embedding_dim=embedding_dim, num_embeddings=vocab_size, 
                                 padding_idx=padding_idx,_weight=pretrained_embedding)

        self.fc1 = nn.Linear(in_features=input_dim*2, out_features=input_dim//2)
        self.fc2 = nn.Linear(in_features=input_dim//2, out_features=input_dim//16)
        self.fc3 = nn.Linear(in_features=input_dim//16, out_features=1)

        self.norm1 = nn.BatchNorm1d(input_dim//2)
        self.norm2 = nn.BatchNorm1d(input_dim//16)
        self.dropout = nn.Dropout(p=0.25)

        if freeze:  self.emb.weight.requires_grad = False


    # Define the forward pass. Takes as argument:
    # - input tensor (shape: batch_size * max_sentence_len * embedding_dim)
    # - tensor of lengths of the sentences
    def forward(self, x_in: torch.Tensor, x_len: torch.Tensor=None, apply_sigmoid: bool=False) -> torch.Tensor:

        # Retrieve the embedding
        x_emb_1 = self.emb(x_in[:,0,:].long()).float()
        x_emb_2 = self.emb(x_in[:,1,:].long()).float()

        # Mean between the embedding dimensions + concatenation
        x_emb_1 = torch.mean(x_emb_1,1)
        x_emb_2 = torch.mean(x_emb_2,1)
        x_emb = torch.cat((x_emb_1, x_emb_2),dim=1)

        # 1st liner layer + ReLU + Normalization + Dropout
        y_int = torch.relu(self.fc1(x_emb))
        y_int = self.dropout(self.norm1(y_int))

        # 2nd liner layer + ReLU + Normalization + Dropout
        y_int = torch.relu(self.fc2(y_int))
        y_int = self.dropout(self.norm2(y_int))

        # 3rd liner layer + Sigmoid
        y_out = self.fc3(y_int).squeeze(dim=1)
        if apply_sigmoid:
            y_out = torch.sigmoid(y_out)

        return y_out


    # Returns the associated Loss Function
    def get_loss_fun(self) -> Callable:
        return nn.BCEWithLogitsLoss()


    # Returns the associated Optimizer
    def get_optimizer(self, learning_rate: float):
        return torch.optim.Adam(self.parameters(), lr=learning_rate,  eps=1e-08, weight_decay=1e-04)


    # Return the associated scheduler
    def get_scheduler(self,optimizer, verbose: bool) -> torch.optim.lr_scheduler:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,verbose=verbose)


    # Reset all the wieghts of the model (used during grid search in notebook)
    def weight_reset(self) -> None:
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()


    # Returns a string rapresentation of the model
    def __str__(self):
      return "MLP_"+str(self.emb.embedding_dim)+"D_"+str(self.emb.num_embeddings)+"TOK"






class LSTMClassifier(nn.Module):

    def __init__(self, input_dim: int, embedding_dim: int, vocab_size: int, padding_idx: int,
                 pretrained_embedding: List[List], freeze: bool=True):

        super(LSTMClassifier, self).__init__()
        
        self.emb = nn.Embedding( embedding_dim=embedding_dim, num_embeddings=vocab_size, 
                                padding_idx=padding_idx,_weight=pretrained_embedding)

        self.lstm1 = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=1, dropout=0.0, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(in_features=embedding_dim*2, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=1)

        self.norm1 = nn.BatchNorm1d(embedding_dim*2)
        self.norm2 = nn.BatchNorm1d(10)
        self.dropout = nn.Dropout(p=0.25)

        if freeze:  self.emb.weight.requires_grad = False


    # Define the forward pass. Takes as argument:
    # - input tensor (shape: batch_size * max_sentence_len * embedding_dim)
    # - tensor of lengths of the sentences
    def forward(self, x_in: torch.Tensor, x_len: torch.Tensor, apply_sigmoid: bool=False) -> torch.Tensor:

        # Retrieve the embedding
        x_emb_1 = self.emb(x_in[:,0,:].long()).float()
        x_emb_2 = self.emb(x_in[:,1,:].long()).float()

        # Pack the padded sequence to avoid useless computation
        packed_embedded1 = nn.utils.rnn.pack_padded_sequence(x_emb_1, x_len[:,0] , batch_first=True, enforce_sorted=False)
        packed_embedded2 = nn.utils.rnn.pack_padded_sequence(x_emb_2, x_len[:,1], batch_first=True, enforce_sorted=False)

        # Take the last hidden state for each sentence (individually)
        y_out1, (h_t1, h_c) = self.lstm1(packed_embedded1)
        y_out2, (h_t2, h_c) = self.lstm1(packed_embedded2)

        # Take the mean between the 2 direction of the lstm cell (forward and backward)
        h_t1 = torch.mean(h_t1,0)
        h_t2 = torch.mean(h_t2,0)

        # Concatenate the 2 hidden state + normalization + dropout
        h_cat = torch.cat((h_t1, h_t2),dim=1)
        h_cat = self.dropout(self.norm1(h_cat))

        # 1st liner layer + ReLU + Normalization + Dropout
        y_int = torch.relu(self.fc1(h_cat))
        y_int = self.dropout(self.norm2(y_int))

        # 2nd liner layer + Sigmoid
        y_out = self.fc2(y_int).squeeze(dim=1)
        if apply_sigmoid:
            y_out = torch.sigmoid(y_out)

        return y_out


    # Returns the associated Loss Function.
    def get_loss_fun(self) -> Callable:
        return nn.BCEWithLogitsLoss()


    # Returns the associated Optimizer.
    def get_optimizer(self, learning_rate: float):
        return torch.optim.Adam(self.parameters(), lr=learning_rate,  eps=1e-08, weight_decay=1e-03)


    # Returns the associated Scheduler.
    def get_scheduler(self,optimizer,verbose: bool=True) -> torch.optim.lr_scheduler:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,verbose=verbose)


    # Reset all the wieghts of the model (used during grid search in notebook)
    def weight_reset(self) -> None:
        self.lstm1.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


    # Returns a string rapresentation of the model
    def __str__(self):
      return "LSTM_"+str(self.emb.embedding_dim)+"D_"+str(self.emb.num_embeddings)+"TOK"