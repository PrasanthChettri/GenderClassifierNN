import torch
from torch import nn

class mod(nn.Module):
    #I hope this model is good
    #NO its not

    def __init__(self):
        super().__init__()
        #input :2d(one hot) : 18*18 
        self.c1 = nn.Conv2d(1 , 3 , 4 , stride = 1 , padding = 3)
        #c1 = 15 * 15 , after maxpool = 5 , 5
        self.c2 = nn.Conv2d(3 , 6 , 2 , stride = 1 , padding = 1)
        #c2 = 3 * 3
        self.relu = nn.Tanh()
        self.maxpool = nn.MaxPool2d(3 , 3)
        self.l1 = nn.Linear(12 , 6)
        self.l2 = nn.Linear(6 , 2)

    def forward(self, x):

        x =  self.relu(self.c1(x))
        x = self.maxpool(self.batch(x))

        x = self.relu(self.c2(x))
        x = self.maxpool(self.batch2(x))

        x = x.view(-1 ,12 )
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))

        return  (x)
