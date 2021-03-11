import torch
from torch import nn
import os
from torch._C import device
import config

#USING BIDIRECTIONAL LSTM
class Model(nn.Module):
    def __init__(self , batch_size , name_length , vocab_size):
        super().__init__()
        self.device = config.DEVICE
        self.to(self.device)
        self.batch_size = config.BATCH_SIZE
        self.max_len = config.NAME_LEN
        self.vocab_size = config.VOCAB_SIZE
        self.hidden_size = config.LSTM_HIDDEN_SIZE
        self.p = config.DROPOUT_PROBABILITY
        self.num_layers = config.NUM_OF_LSTM_LAYERS

        #bidirectional LSTM layer
        self.lstm = nn.LSTM(
                    self.vocab_size , self.hidden_size ,
                    bidirectional = True, dropout = self.p ,
                    num_layers = self.num_layers , batch_first = True
                )
        #dense layer 1 
        self.fc = nn.Linear(self.hidden_size*2 , 1)

    def _init_zeroes(self):
        h0 =  torch.zeros(self.num_layers*2,
                    self.batch_size, self.hidden_size , device = self.device
                )
        c0 =  torch.zeros(self.num_layers*2,
                    self.batch_size, self.hidden_size , device=  self.device
                )
        return (h0 , c0)


    def forward(self, x):
        opt, ( hn , cn ) = self.lstm(x ,self._init_zeroes())
        x = opt[:, -1, :]
        x = self.fc(x)
        return x 

    def save(self , name):
        torch.save(self.state_dict() , "weights\weight.pth".format())