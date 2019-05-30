import torch
from torch import nn

class mod(nn.Module):

    #I hope this model is good
    #Conv2d with one hot encoding does better than Conv1d

    def __init__(self):

        super().__init__()
        #input :2d(one hot) : 53*53
        self.c1 = nn.Conv2d(1 , 3 , 4 , stride = 1 , padding = 0)
        #c1 = 50 * 50 , after maxpool = 12  , 12
        self.c2 = nn.Conv2d(3 , 6 , 2 , stride = 1 , padding = 1)
        #c2 = 7 * 7  , after maxpool = 3  , 3
        self.relu = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(4 , 4)
        self.maxpool2 = nn.MaxPool2d(2 , 2)
        self.l1 = nn.Linear(18 , 9)
        self.l2 = nn.Linear(9 , 4)
        self.l3 = nn.Linear(4 , 2)

    def forward(self, x):

        x =  self.relu(self.c1(x))
        x = self.maxpool(x)
        x = self.relu(self.c2(x))
        x = self.maxpool(x)
        
        x = x.view(-1 ,18 )

        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))

        return  (x)
