import torch 
from torch.nn import functional as F
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
import pandas as pd
import seaborn as sns
import numpy as np
from typing import Dict , List , Tuple
from dataset import getdata
import models
from tokenizer import Tokenizer
import config

class Trainer:
    def __init__(self , model , epochs):
        self.model  =  model
        self.epochs = epochs
        self._init_dataset()

    def plot_chart(self):
        '''
            Training and accuracy charts
        '''
        epoch_list = list( range(1 , self.epochs + 1) )
        plt.plot(epoch_list, self.train_loss_list)
        plt.plot(epoch_list, self.valid_loss_list)
        plt.show()

    def plt_confusion_matrix(self):
        figsize = (7,5)
        fontsize= 10
        df_cm = pd.DataFrame(
            self.conf_matrix , index = config.CLASSES , columns = config.CLASSES
        )
        fig = plt.figure(figsize=figsize)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
        heatmap.yaxis.set_ticklabels(
                            heatmap.yaxis.get_ticklabels(), rotation=0,
                            ha='right', fontsize=fontsize
                        )
        heatmap.xaxis.set_ticklabels(
                            heatmap.xaxis.get_ticklabels(),
                            rotation=45, ha='right', fontsize=fontsize
                        )
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def _init_dataset(self):
        self.train_set, self.valid_set, self.test_set = getdata()
        self.valid_data_size = len(self.valid_set)
        self.train_data_size = len(self.train_set)
        self.test_data_size = len(self.test_set)

    def test(self):
        test_data_size = len(self.test_set)
        test_size = test_data_size*config.BATCH_SIZE
        y_pred = np.array([] , dtype = int)
        y_true = np.array([] , dtype = int)
        with torch.no_grad():
            for fet , label in self.test_set :
                output = self.model.forward(fet)
                pred = torch.sigmoid(output).flatten()
                y_true = np.concatenate(
                            (y_true , torch.round(label).int().cpu().numpy()) , axis =  None 
                        )
                y_pred = np.concatenate(
                            (y_pred , torch.round(pred).int().cpu().numpy()) , axis =  None 
                        )
        self.cm = confusion_matrix(y_true, y_pred)
        tn , fp , fn , tp = self.cm.ravel()
        print("Accuracy {}".format(100*(tn + tp)/test_size))

    def train(self):

        criterion = config.CRITERION()
        opt = config.OPTIMZER(
            self.model.parameters() , lr = config.LEARNING_RATE
        )
        '''
            func trains the model
            and saves the data
        '''
        self.train_loss_list = []
        self.valid_loss_list = []

        for epoch_number in range(1 , self.epochs + 1):
            train_loss = 0 
            valid_loss = 0 
            self.model.train()

            for fet , label in self.train_set:
                opt.zero_grad()
                output = self.model.forward(fet).flatten()
                t_loss = criterion(output , label)
                opt.zero_grad()
                t_loss.backward()
                opt.step()
                train_loss += t_loss.item()

            #validation process 
            with torch.no_grad():
                for fet , label in self.valid_set:
                    output = self.model.forward(fet)
                    output = torch.sigmoid(output).flatten()
                    v_loss = criterion(output , label)
                    valid_loss += v_loss.item()

            overall_t_loss = train_loss/self.train_data_size
            overall_v_loss = valid_loss/self.valid_data_size
            self.train_loss_list.append(overall_t_loss)
            self.valid_loss_list.append(overall_v_loss)

            print(f"epoch number = {epoch_number}, "
                f"train_loss = {overall_t_loss}, " 
                f"valid_loss = {overall_v_loss}")

def main():

if __name__ == "__main__":
    main()
