#TODO : USE LSTMS 
import model
import pandas as pd
import torch
from torch import nn , optim
from random import shuffle

def pad(x):
    for i in range(len(x)):
        assert len(x[i]) < 20 , "nope" 
        x[i]+= [0]*(20-len(x[i]))
        x[i] = list(one_hot(x[i]))

    return (torch.Tensor(x))

def one_hot (x):
    for i in x :
        z = [0]*53
        z[i] = 1
        yield z   


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


    def get(self, batch= 1):
        ret_f  =  []
        ret_l  =  []
        for i in range(batch):
            tret_f  , tret_l = next(self.getter)
            ret_f.append(tret_f)
            ret_l.append(tret_l)
        return ret_f , ret_l

    def traindat(self, epochs = 15):
        
        tnn = model.mod2()
        tnn.train()
        criterion = nn.CrossEntropyLoss()
        opt = optim.Adam(tnn.parameters() , lr = 0.001)
        sof = nn.Softmax(dim = 1)
        print ("Training...")
        #main training loop
        valid_matrix = []
        old_valid_value = float('inf')

        while True :
            train_loss = 0 
            valid_loss = 0 
            z_t = 0
            z_v = 0
            tnn.train()

            for i in range(2800):
                opt.zero_grad()
                fet , label =  self.get()
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
            print ("-------------------------------------------")

            #validation process 
            with torch.no_grad():
                for i in range(200):
                    fet , label =  self.get()
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

            valid_matrix.append(valid_loss)

            if len(valid_matrix) >= 3:
                if sum(valid_matrix) > old_valid_value:
                    print ("Breaking Loop")
                    break 
                else :
                    old_valid_value = sum(valid_matrix)
                    print ("Valid aggregate ----> " , old_valid_value)
                    valid_matrix = []


        if input("save") == 'y':
            s = input("name?")
            torch.save(tnn.state_dict() , s)


if __name__  =='__main__':
    z= train()
    z.traindat()
    print ("FINISHED")
