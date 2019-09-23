#TODO : USE LSTMS 
import model
import pandas as pd
import torch
from torch import nn , optim
from random import shuffle

def pad(x):
    for i in range(len(x)):
        assert len(x[i]) < 15 , "nope" 
        x[i] += [0]*(15-len(x[i]))
        x[i] = list(one_hot(x[i]))
    return (torch.Tensor(x))

def one_hot (x):
    for i in x :
        z = [0]*27
        if i:
            z[i] = 1
        yield z 

class train:
    def __init__(self):
        self.data  = pd.read_csv("gender_refine-csv.csv").dropna()
        self.l = self.data['gender'].values.tolist()
        self.f =self.data['name'].values.tolist()
        ind = list(range(len(self.f)))
        shuffle(ind)
        split = 0.3
        split = int(split*len(self.l))

        al =  list(' abcdefghijklmnopqrstuvwxyz')
        self.dict = {}
        for i , j  in enumerate(al):
           self.dict[j]  = i 
        self.t_getter = self.dat(ind[:split])
        #making the split small because my computer is a piece of shit
        self.v_getter = self.dat(ind[split:split*2])

    def dat(self, ind , batch = 1 , train = True):
        ind = iter(ind)
        k = []
        b_i = batch
        while ind:
            batch = b_i
            while batch:
                try :
                    ai = next(i)
                    m = [[self.dict[ii] for ii in self.f[ai].lower()], self.l[ai]]
                    k.append(m)
                    batch -= 1
                except Exception:
                    pass
            yield k

    def traindat(self, epochs = 15):
        tnn = model.mod2()
        tnn.train()
        criterion = nn.CrossEntropyLoss()
        opt = optim.Adam(tnn.parameters() , lr = 0.001)
        sof = nn.Softmax(dim = 1)
        print ("Training...")
        #main training loop
        old_valid_value = float('inf')
        o_vval = float('inf')
        k = 0 

        while True:
            k += 1
            train_loss = 0 
            valid_loss = 0 
            z_t = 0
            z_v = 0
            tnn.train()

            for fet , label  in self.t_getter:
                opt.zero_grad()
                fet = pad(fet).float()
                label = torch.Tensor(label).long()
                #label like = tensor([1 , 0 , 0 , 0])
                output = tnn.forward(fet)

                try :
                    t_loss = criterion(output , label)
                except Exception as ep :
                    opt.zero_grad()
                    continue 

                t_loss.backward()
                z_t += 1
                opt.step()
                train_loss +=t_loss.item()
            print ("LOSS --> " , (train_loss/z_t))

            #validation process 
            with torch.no_grad():
                for fet , label in self.v_getter():
                    fet = pad(fet).float()
                    label = torch.Tensor(label).long()
                    #label like = tensor([1 , 0 , 0 , 0])
                    output = tnn.forward(fet)
                    try :
                        v_loss = criterion(output , label)
                    except Exception as ep :
                        continue 
                    valid_loss += v_loss.item()
                    z_v += 1

                print ("valid_loss --->" , valid_loss/z_v)
                print ("epoch no---> " , k)
            print ("-------------------------------------------")

            if o_vval < valid_loss :
                break
            o_vval = valid_loss 
        torch.save(tnn.state_dict() , "LATEST6")

if __name__  =='__main__':
    z= train()
    z.traindat()
