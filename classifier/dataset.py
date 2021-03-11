from random import shuffle 
import torch
import pandas as pd
import numpy as np
from torch._C import device
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Tuple , List
import random
import config

from tokenizer import Tokenizer

def get_sample(
        range_len : int , split_ratio : List[float]
)->Tuple[SubsetRandomSampler , SubsetRandomSampler , SubsetRandomSampler]:
        '''
                function returns lists of shuffled ids given of range_len 
                one for training, valid and test each
                
        '''
        range_list = list( range(range_len) )
        len_dataset = len(range_list)
        random.shuffle(range_list)
        split_len = list( map(lambda x : int(len_dataset*x) , split_ratio) )

        train_idx = [0 , split_len[0]]
        valid_idx = [split_len[0] , split_len[0] + split_len[1]]
        test_idx =  [split_len[0] + split_len[1] , -1]

        return (
                range_list[train_idx[0] : train_idx[1]] ,  # samples for training
                range_list[valid_idx[0] : valid_idx[1]] ,  # samples for validation
                range_list[test_idx[0] : test_idx[1]]      # samples for testing 
        )  


def getdata(batch_size  : int,
)->Tuple[DataLoader , DataLoader , DataLoader]:
        '''
                Standart data cleaning
                Merges data from the two CSVs that are datasets
                returns processed data
        '''
        split_ratio = config.SPLIT_RAIO
        name_len = config.NAME_LEN

        assert len(split_ratio) == 3 and sum(split_ratio) == 1

        t_obj = Tokenizer(name_len)
        data  = pd.concat([pd.read_csv("gender_refine-csv.csv").dropna() , 
                        pd.read_csv("gender_refine-csv2.csv").dropna()],
                        sort =False)

        train_sample ,  valid_sample , test_sample  = get_sample(len(data) , split_ratio)
        fet = data['name'].map(lambda x : t_obj.tkniz(x)).values.tolist()
        label = data['gender'].replace({3 : 0.5}).values.tolist()
        fet = torch.Tensor(fet).type(dtype=torch.float32).cuda()
        label = torch.Tensor(label).type(dtype=torch.float32).cuda()
        dataset  = TensorDataset(fet , label) 

        dataset_params = {
                'dataset'   :  dataset,
                'batch_size': config.BATCH_SIZE,
                'drop_last' : True ,
                'device'    : config.DEVICE
        }

        return (
                DataLoader(sampler = train_sample, **dataset_params) ,
                DataLoader(sampler = valid_sample , **dataset_params) ,
                DataLoader(sampler = test_sample , **dataset_params)
        )
