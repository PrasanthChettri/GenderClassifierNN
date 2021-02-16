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
    def __init__(self , batch_size , name_length):
        super().__init__()
        self.batch_size = batch_size
        self.max_len = name_length

        self.lstm = nn.LSTM(27 , 4 , bidirectional = True)
        self.l1 = nn.Linear(96 , 36)
        self.l2 = nn.Linear(36 , 10)
        self.l3 = nn.Linear(10 , 2)
        self.act = nn.Sigmoid()
        self.drpout = nn.Dropout(0.169)
        self.inneract = nn.Tanh()

    def forward(self, x):
        x = x.view(self.max_len , self.batch_size , 27)
        x , y = self.lstm(x)
        x = x.view(self.batch_size, 1 , -1)
        x = self.drpout(self.inneract(self.l1(x)))
        x = self.drpout(self.inneract(self.l2(x)))
        return self.drpout(self.l3(x))
