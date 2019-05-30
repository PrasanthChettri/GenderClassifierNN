import model
import pandas as pd
import torch
from torch import nn , optim
from random import shuffle

class train:

    def __init__(self):
        self.data  = pd.read_csv("gender_refine-csv.csv").dropna()
        self.l = self.data['gender'].values.tolist()
        self.f =self.data['name'].values.tolist()
        self.getter = self.dat()

    def dat(self):
        al =  list(' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.dict = {}
        for i , j  in enumerate(al):
           self.dict[j]  = i 
        ind = list(range(len(self.f)))
        shuffle(ind)
        for i in ind :
            try :
                yield [self.dict[ii] for ii in self.f[i]]  , self.l[i]
            except Exception:
                pass


    def get(self, batch= 4):
        ret_f  =  []
        ret_l  =  []
        for i in range(batch):
            tret_f  , tret_l = next(self.getter)
            ret_f.append(tret_f)
            ret_l.append(tret_l)
        return ret_f , ret_l

    @staticmethod
    def pad(x):
        for i in range(len(x)):
            assert len(x[i]) < 20 , "nope" 
            x[i]+= [0]*(20-len(x[i]))
            x[i] = list(train.one_hot(x[i]))

        return (torch.Tensor(x))
    
    @staticmethod
    def one_hot (x):
        for i in x :
            z = [0]*53
            z[i] = 1
            yield z   

    def traindat(self, epochs = 15):
        tnn = model.mod()
        tnn.train()
        criterion = nn.CrossEntropyLoss()
        opt = optim.Adam(tnn.parameters() , lr = 0.01)
        sof = nn.Softmax(dim = 1)
        print ("Training...")
        #main training loop

        for i in range(epochs):
            trainloss = 0 
            z = 0

            for i in range(700):
                fet , label =  self.get()
                fet = train.pad(fet).float().unsqueeze(dim = 1)
                label = torch.Tensor(label).long()
                #label like = tensor([1 , 0 , 0 , 0])
                output = tnn.forward(fet)

                try :
                    loss = criterion(output , label)
                except Exception :
                    opt.zero_grad()
                    continue 

                loss.backward()
                z += 1
                opt.step()
                opt.zero_grad()
                trainloss += loss.item()
                opt.zero_grad()

            print ("LOSS --> " , (trainloss/z)*4)
            print (z)
        if input("save") == 'y':
            torch.save(tnn.state_dict() , 'E35,700S.pth')



if __name__  =='__main__':
    z= train()
    z.traindat(epochs = 35)
    print ("FINISHED")
