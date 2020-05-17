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
        self.nl = name_length

        self.lstm = nn.LSTM(27 , 4 , bidirectional = True)
        self.l1 = nn.Linear(160 , 80)
        self.l2 = nn.Linear(80 , 40)
        self.l3 = nn.Linear(40 , 10)
        self.l4  = nn.Linear(10 , 2)
        self.act = nn.Sigmoid()
        self.inneract = nn.Tanh()

    def forward(self, x):
        x = x.view(self.nl , self.batch_size , 27)
        x , y = self.lstm(x) 
        x = x.view(1 , -1)
        x = self.inneract(self.l1(x))
        x = self.inneract(self.l2(x))
        x = self.inneract(self.l3(x))
        x = self.l4(x)
        return x
