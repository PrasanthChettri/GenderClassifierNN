from tokenizer import tokenizer
from random import shuffle 
import torch
import pandas as pd
import numpy as np
import torch.utils.data  as data_utils

def getdata(batch_size , name_len , split ):
        t_obj = tokenizer(name_len)
        data  = pd.concat([pd.read_csv("gender_refine-csv.csv").dropna() , 
                            pd.read_csv("gender_refine-csv2.csv").dropna()]) 
        split = int(len(data)*split) 
        fet = data['name'].map(lambda x : t_obj.tkniz(x)).values.tolist()
        label = data['gender'].replace({3 : 0.5})
        label = np.array(label.map(lambda x : [x , 1-x]).values.tolist()).astype(np.float32)
        fet = torch.Tensor(fet).cuda()
        label = torch.from_numpy(label).cuda()
        train_loader =data_utils.TensorDataset(fet[:split] , label[:split]) 
        train_dataset = data_utils.DataLoader(dataset = train_loader, batch_size = batch_size, shuffle = True)
        valid_loader =data_utils.TensorDataset(fet[split:] , label[split:]) 
        valid_dataset = data_utils.DataLoader(dataset = valid_loader, batch_size = batch_size, shuffle = True)
        return train_dataset , valid_dataset , len(data)


if __name__ == "__main__":
    getdata(10 , 12 , 0.8)
