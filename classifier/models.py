import torch
from torch import nn

class mod(nn.Module):

    #I hope this model is good
    #Conv2d with one hot encoding does better than Conv1d
    def __init__(self):
        super().__init__()
        #input :2d(one hot) : 53*20
        self.c1 = nn.Conv2d(1 , 3 , 4 , stride = 1 , padding = 0)
        #c1 = 50 * 17 , after maxpool = 12  , 4
        self.c2 = nn.Conv2d(3 , 6 , 2 , stride = 1 , padding = 1)
        #c2 = 13* 5  , after maxpool = 6  , 2
        self.c3 = nn.Conv2d(6 , 15 , 3 , stride = 3 , padding = 1 ) #c2 = 2* 1  
        self.sig = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(4 , 4)
        self.maxpool2 = nn.MaxPool2d(2 , 2)
        self.l1 = nn.Linear(30 , 10)
        self.l2 = nn.Linear(10 ,  2)

    def forward(self, x):
        x =  self.sig(self.c1(x))
        x = self.maxpool(x)
        x = self.sig(self.c2(x))
        x = self.maxpool2(x)
        x = self.sig(self.c3(x))
        x = x.view(-1 ,30 )
        x = self.sig(self.l1(x))
        x = self.sig(self.l2(x))

        return  (x)

#USING BIDIRECTIONAL LSTM
class mod2(nn.Module):
    def __init__(self , batch_size , name_length , vocab_size):
        super().__init__()
        self.batch_size = batch_size
        self.max_len = name_length
        self.vocab_size = vocab_size
        self.hidden_lstm = 2
        self.p = 0.2 

        input_size = self.hidden_lstm*2

        #bidirectional LSTM layer
        self.lstm = nn.LSTM(
                    self.vocab_size , self.hidden_lstm ,
                    bidirectional = True, dropout = self.p
                )

        #dense layer 1 
        self.l1 = nn.Linear(input_size, 1)
        self.drpout = nn.Dropout(p = self.p)

    def forward(self, x):
        x = x.view(self.max_len , self.batch_size , self.vocab_size)
        h2, ( [ h1 ,h2 ], [ c1, c2 ]) = self.lstm(x)

        #h1 = output for LSTM layer 1 (forward)
        #h2 = output for LSTM layer 2 (reverse)

        x = torch.cat((h1 , h2))
        h1 = h1.view([1 , * h1.shape])
        h2 = h2.view([1 , * h2.shape])
        x = torch.cat((h1 , h2))
        print(x.shape)
        x = self.drpout(x)
        exit()
        x = x.view(self.bath_size, 1 , -1)
        x = self.l1(x)
        exit()
        return x 
