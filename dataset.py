from tokenizer import tokenizer
from random import shuffle 
import torch
import pandas as pd

class Dataset: 
    def __init__(self, sp_t ,batch_size , name_len):
        data  = pd.read_csv("gender_refine-csv.csv").dropna()
        self.len_d = len(data)
        self.l = data['gender'].values.tolist()
        self.f = data['name'].values.tolist()
        #split betweeen Training AND validation
        self.t_obj = tokenizer(name_len)
        self.batch = batch_size
        self.split = int(sp_t*self.len_d)
        self.load()

    def load(self):
        ind = list(range(self.len_d))
        shuffle(ind)

        t_ind = ind[:self.split]
        t_data , self.t_pop = self.dat(iter(t_ind))

        v_ind = ind[self.split:]
        v_data , self.v_pop = self.dat(iter(v_ind))

        return t_data , v_data

    def dat(self, ind):
        all_data = []
        fandl = []
        counter = 0 
        all_counter = 0
        for ai in ind : 
            lab = self.l[ai]
            if lab == 3 :
                continue 
            lab = torch.Tensor([round(lab)]).long().cuda()
            fet = torch.Tensor(self.t_obj.tkniz(self.f[ai])).float().cuda()
            fandl.append([fet , lab])
            counter += 1
            if counter == self.batch :
                all_data.append(fandl)
                all_counter += counter
                fandl = []
                counter =0 

        return all_data , all_counter

if __name__ == "__main__":
    obj = Dataset(0.3 ,1 ,  20)
    print(obj.load()[0][0])
