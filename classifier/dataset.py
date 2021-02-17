from random import shuffle 
import torch
import pandas as pd
import numpy as np
import torch.utils.data  as data_utils
from typing import Tuple

from tokenizer import Tokenizer

def getdata(batch_size  : int,
        name_len : int,
        split_ratio : float
)->Tuple[data_utils.DataLoader , data_utils.DataLoader , int]:
        '''
                Standart data cleaning
                Merges data from the two CSVs that are datasets
                returns processed data
        '''
        t_obj = Tokenizer(name_len)
        data  = pd.concat([pd.read_csv("gender_refine-csv.csv").dropna() , 
                            pd.read_csv("gender_refine-csv2.csv").dropna()]) 

        split = int(len(data)*split_ratio)

        fet = data['name'].map(lambda x : t_obj.tkniz(x)).values.tolist()
        label = data['gender'].replace({3 : 0.5})
        label = np.array(label.map(lambda x : [x , 1-x]).values.tolist()).astype(np.float32)

        '''
                load them into cuda 
                if cuda 
        '''
        fet = torch.Tensor(fet).cuda()
        label = torch.from_numpy(label).cuda()
        train_loader =data_utils.TensorDataset(fet[:split] , label[:split]) 
        train_dataset = data_utils.DataLoader(dataset = train_loader, batch_size = batch_size, shuffle = True)
        valid_loader =data_utils.TensorDataset(fet[split:] , label[split:]) 
        valid_dataset = data_utils.DataLoader(dataset = valid_loader, batch_size = batch_size, shuffle = True)
        return train_dataset , valid_dataset , len(data)


if __name__ == "__main__":
        '''
                can be used to fetch or set up a 
                'piplene' of sorts to save and then train data
        '''
        getdata(10 , 12 , 0.8)
