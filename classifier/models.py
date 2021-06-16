import torch
from torch import nn
from classifier import config
from typing import List
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
import pandas as pd
import random



#USING BIDIRECTIONAL LSTM
class Model(LightningModule):
    def __init__(self , batch_size):
        """
        initialise all the params from the conf file
        """
        super().__init__()
        self.batch_size = batch_size
        self.max_len = config.NAME_LEN
        self.vocab = config.VOCAB
        self.vocab_size = len(self.vocab)
        self.hidden_size = config.LSTM_HIDDEN_SIZE
        self.p = config.DROPOUT_PROBABILITY
        self.num_layers = config.NUM_OF_LSTM_LAYERS
        self.lr = config.LEARNING_RATE
        self.n_workers = config.NUM_WORKERS

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
        self.val_confusion = pl.metrics.classification\
                .ConfusionMatrix(num_classes  = 2)

    def _init_token(self):
        """
        initialises the vocab dictionary
        key = alphabet
        value = one hot encoding
        the dictionary for the vocab
        """
        vectors = []
        for i in range(self.vocab_size):
            vectors.append(
                [0]*i + [1] + [0]*(self.vocab_size-i-1)
            )
        self.vocab_dict = dict( zip(self.vocab , vectors) )

    def tokenize(self, name):
        """
        one hot encoding the a name
        filter out symbols not in vocab
        padding the tokens with zero at the front
        if len_name bigger than len supported, snipp it
        """
        name = name.split(' ')[0]
        name = list(
            filter(lambda word : word in self.vocab , name.lower())
        )
        len_name = len(name)

        if len_name < self.max_len :
            token = [[0] * self.vocab_size] * (self.max_len - len_name)
        else :
            name = name[:self.max_len]
            token = []

        for alpha in name :
            token.append(self.vocab_dict[alpha])
        return token

    def _get_sample(self):
        """
        function returns lists of sampler of shuffled indices
        one for training, valid and test each
        """
        len_dset = len(self.dataset)
        range_list = list(range( len_dset ))
        random.shuffle(range_list)
        split_ratio = config.SPLIT_RATIO

        train_end = int(split_ratio[0]*len_dset)
        valid_end = int(split_ratio[1]*len_dset)

        self.train_set = range_list[0 : train_end]
        self.valid_set = range_list[train_end : valid_end]
        self.test_set =  range_list[valid_end : -1]

    def test_step(self, batch, batch_idx):
        """
        test step- compute loss, accuracy
        """
        x, y = batch
        y_hat = self(x).flatten()
        loss = self.b_loss(y_hat, y)
        self.log('test_loss' , loss)
        probab_pred = torch.sigmoid(y_hat)
        pred = torch.round(probab_pred).flatten().int()
        self.val_confusion.update(pred, y.int())
        return loss

    def test_epoch_end(self, outputs):
        """
        computes accuracy -(true negatives + true positive)/total
        from the confusion matrix
        """
        metric_val = self.val_confusion.compute().flatten()
        tn , fp , fn , tp = metric_val
        total = torch.sum(metric_val)
        accuracy = (tn + tp)/total
        return {
            'accuracy' : accuracy
        }


    def prepare_data(self):
        """ prepare data from the dataset """
        data  = pd.concat([pd.read_csv("classifier/dataset/gender_refine-csv.csv").dropna() ,
                        pd.read_csv("classifier/dataset/gender_refine-csv2.csv").dropna()],
                        sort = False)

        fet = data['name'].map(lambda x : self.tokenize(x)).values.tolist()
        label = data['gender'].replace({3 : 0.5}).values.tolist()
        self.dataset = TensorDataset(
                torch.Tensor(fet).type(dtype=torch.float32) ,
                torch.Tensor(label).type(dtype=torch.float32)
            )
        self._get_sample()

    def train_dataloader(self):
        return DataLoader(self.dataset,
                sampler = self.train_set, batch_size = self.batch_size,
                drop_last = True, num_workers = self.n_workers

              )

    def val_dataloader(self):
        return DataLoader(self.dataset,
                sampler = self.valid_set, batch_size = self.batch_size,
                drop_last = True , num_workers = self.n_workers
              )

    def test_dataloader(self):
        return DataLoader(self.dataset,
                sampler = self.test_set, batch_size = self.batch_size,
                drop_last = True, num_workers = self.n_workers
            )

    def configure_optimizers(self):
        optimizer =  torch.optim.Adam(self.parameters() , lr = self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        """training step """
        x , y = train_batch
        logits = self.forward(x).flatten()
        loss = self.b_loss(logits, y)
        self.log('training_loss' , loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        """validation step"""
        x, y = val_batch
        logits = self.forward(x).flatten()
        loss = self.b_loss(logits, y)
        self.log('val_loss' , loss)
        return loss

    def training_epoch_end(self, outputs):
        """compute mean loss for the training epoch"""
        losses = list( map(lambda loss_dict : loss_dict['loss'] , outputs) )
        avg_loss = torch.stack(losses).mean()
        self.log('avg training loss' , avg_loss)

    def validation_epoch_end(self, outputs):
        """compute mean loss for the validation epoch"""
        avg_loss = torch.stack(outputs).mean()
        self.log('avg validation loss', avg_loss)

    def forward(self, x):
        """
        @param : x - input of dimensions:(batch_size , name_len , vocab_size)
        run a forward step
        """
        opt, ( hn , cn ) = self.lstm(x ,self._init_zeroes())
        #take the last hidden state
        x = opt[:, -1, :]
        #run through a dense layer
        x = self.fc(x)
        return x

    def _init_zeroes(self):
        h0 =  torch.zeros(self.num_layers*2,
                    self.batch_size, self.hidden_size , device = self.device)
        c0 =  torch.zeros(self.num_layers*2,
                    self.batch_size, self.hidden_size , device=  self.device)
        return (h0 , c0)
