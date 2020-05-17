from tokenizer import tokenizer
from random import shuffle 
import torch
import pandas as pd

class Dataset: 
    def __init__(self, sp , batch_size , name_len):
        data  = pd.read_csv("gender_refine-csv.csv").dropna()
        self.len_d = len(data)
        self.l = data['gender'].values.tolist()
        self.f = data['name'].values.tolist()
        #split betweeen Training AND validation
        self.split = int(sp*len(self.l))
        self.t_obj = tokenizer(name_len)
        self.batch = batch_size

        self.ind = list(range(self.len_d))
        shuffle(self.ind)

    def t_get(self):
        t_ind = self.ind[:self.split]
        t_ind = iter(t_ind)
        return self.dat(t_ind)

    def v_get(self):
        v_ind = self.ind[self.split:2*self.split]
        v_ind = iter(v_ind)
        return self.dat(v_ind) 

    def dat(self, ind):
        while ind :
            fandl = []
            i = 0
            while i < self.batch : 
                try : 
                    ai = next(ind)
                except StopIteration :
                    return

                # no unisex names
                lab = self.l[ai]
                if lab == 3 :
                    continue 

                lab = torch.Tensor([round(lab)]).long()
                fet = torch.Tensor(self.t_obj.tkniz(self.f[ai])).float()
                m = [fet , lab]
                fandl.append(m)
                i += 1
            yield fandl

if __name__ == "__main__":
    obj = Dataset(0.3)
    print(next(obj.t_get()))
