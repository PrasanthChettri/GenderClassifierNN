import torch
from torch import nn
import config
import pytorch_lightning as pl
from typing import List
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import random
import numpy as np
 

#USING BIDIRECTIONAL LSTM
class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.batch_size = config.BATCH_SIZE
        self.max_len = config.NAME_LEN
        self.vocab = config.VOCAB
        self.vocab_size = len(self.vocab)
        self.hidden_size = config.LSTM_HIDDEN_SIZE
        self.p = config.DROPOUT_PROBABILITY
        self.num_layers = config.NUM_OF_LSTM_LAYERS
        self.lr = config.LEARNING_RATE

        self.b_loss = nn.BCEWithLogitsLoss()
        #bidirectional LSTM layer
        self.lstm = nn.LSTM(
                    self.vocab_size ,
                    self.hidden_size ,
                    bidirectional = True,
                    dropout = self.p ,
                    num_layers = self.num_layers,
                    batch_first = True
                )
        #dense layer 1 
        self.fc = nn.Linear(self.hidden_size*2 , 1)
        self._init_token()

    def _init_token(self):
        '''
            init token intialises to 
            the dictionary for the vocab
        '''
        vectors = []
        for i in range(self.vocab_size):
            vectors.append(
                [0]*i + [1] + [0]*(self.vocab_size-i-1)
            )
        self.v_dict = dict( zip(self.vocab , vectors) )


    def tokenize(self, name):
        name = name.lower()
        '''
            filter out symbols not in vocab
        '''
        name = list(
                    filter(lambda word : word in self.vocab , name)
                )
        len_name = len(name)
        '''
            padding the tokens with zero at the front
            if len_name bigger than len supported 
            we snipp it
        '''
        if len_name < self.max_len : 
            token = [[0] * self.vocab_size] * (self.max_len - len_name)
        else : 
            name = name[:self.max_len]
            token = []

        for alpha in name : 
            token.append(self.v_dict[alpha])

        return token

    def _get_sample(self):
        '''
                function returns lists of shuffled ids given of range_len 
                one for training, valid and test each
                
        '''
        len_dset = len(self.dataset)
        range_list = list(range( len_dset ))

        random.shuffle(range_list)

        split_ratio = config.SPLIT_RATIO
        train_end = int(split_ratio[0]*len_dset) 
        valid_end = int(split_ratio[1]*len_dset) 

        self.train_set = range_list[0 : train_end]
        self.valid_idx = range_list[train_end : valid_end]
        self.test_idx =  range_list[valid_end : -1]
        

    def prepare_data(self):
        data  = pd.concat([pd.read_csv("gender_refine-csv.csv").dropna() , 
                        pd.read_csv("gender_refine-csv2.csv").dropna()],
                        sort = False)
        train_sample, valid_sample, test_sample = self._get_sample(data)

        fet = data['name'].map(lambda x : self.tokenize(x)).values.tolist()
        label = data['gender'].replace({3 : 0.5}).values.tolist()

        fet = torch.Tensor(fet).type(dtype=torch.float32).cuda()
        label = torch.Tensor(label).type(dtype=torch.float32).cuda()

        self.dataset = TensorDataset(fet, label)

        self._get_sample(self)

    def train_dataloader(self):
        return DataLoader(self.dataset, 
                    sampler = self.train_sample, batch_size = self.batch_size,
                    drop_last = True
                )

    def val_dataloader(self):
        return DataLoader(self.dataset, 
                    sampler = self.valid_sample, batch_size = self.batch_size,
                    drop_last = True
                )

    def test_dataloader(self):
        return DataLoader(self.dataset, 
                    sampler = self.test_sample, batch_size = self.batch_size,
                    drop_last = True
                )
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters() , lr = self.lr)

    def training_step(self, train_batch, batch_idx):
        x , y = train_batch
        logits = self.forward(x)
        loss = self.b_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log('val_loss', loss)

    def forward(self, name : str):
        exit()
        opt, ( hn , cn ) = self.lstm(x ,self._init_zeroes())
        x = opt[:, -1, :]
        x = self.fc(x)
        return x 

if __name__ == "__main__":
    a = Model()
    a.forward("abcd")
