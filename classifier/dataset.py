from random import shuffle 
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Tuple , List
import random

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
                range_list[test_idx[0] : test_idx[1]]
        )  # samples for testing 


def getdata(batch_size  : int,
        name_len : int,
        split_ratio : List[float]
)->Tuple[DataLoader , DataLoader , DataLoader]:
        '''
                Standart data cleaning
                Merges data from the two CSVs that are datasets
                returns processed data
        '''
        assert len(split_ratio) == 3 and sum(split_ratio) == 1
        t_obj = Tokenizer(name_len)
        data  = pd.concat([pd.read_csv("gender_refine-csv.csv").dropna() , 
                            pd.read_csv("gender_refine-csv2.csv").dropna()]) 


        train_sample ,  valid_sample , test_sample  = get_sample(len(data) , split_ratio)
        fet = data['name'].map(lambda x : t_obj.tkniz(x)).values.tolist()
        label = data['gender'].replace({3 : 0.5}).values.tolist()


        fet = torch.Tensor(fet).type(dtype=torch.float32).cuda()
        label = torch.Tensor(label).type(dtype=torch.float32).cuda()
        
        dataset  = TensorDataset(fet , label) 
        train_dataset = DataLoader( dataset = dataset,
                        batch_size = batch_size, sampler=train_sample
                )


        valid_dataset = DataLoader(dataset = dataset,
                        batch_size = batch_size , sampler=valid_sample
                )

        test_dataset = DataLoader(dataset = dataset,
                        batch_size = batch_size, sampler=test_sample
                )
        return train_dataset , valid_dataset , test_dataset