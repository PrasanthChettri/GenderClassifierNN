import torch 
from torch import nn , optim
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


def plot_chart(train_loss_list : List[float] , valid_loss_list = List[float]):
    '''
        Training and accuracy charts
    '''
    epoch_list = list( range(1 , config.EPOCHS + 1) )
    plt.plot(epoch_list, train_loss_list)
    plt.plot(epoch_list, valid_loss_list)
    plt.show()

def plt_confusion_matrix(conf_matrix):
    figsize = (7,5)
    fontsize= 10
    df_cm = pd.DataFrame(
        conf_matrix , index = config.CLASSES , columns = config.CLASSES
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


def test(model, test_set):
    test_data_size = len(test_set)
    test_size = test_data_size*config.BATCH_SIZE
    y_pred = np.array([] , dtype = int)
    y_true = np.array([] , dtype = int)
    with torch.no_grad():
        for fet , label in test_set :
            output = model.forward(fet)
            pred = torch.sigmoid(output).flatten()
            y_true = np.concatenate(
                        (y_true , torch.round(label).int().cpu().numpy()) , axis =  None 
                    )
            y_pred = np.concatenate(
                        (y_pred , torch.round(pred).int().cpu().numpy()) , axis =  None 
                    )
    cm = confusion_matrix(y_true, y_pred)
    tn , fp , fn , tp = cm.ravel()
    print("Accuracy {}".format(100*(tn + tp)/test_size))
    return cm

def train(
    model, criterion, opt , train_set, valid_set
)->Tuple[List[float] , List[float]]:
    
    '''
        func trains the model
        and saves the data
    '''
    valid_data_size = len(valid_set)
    train_data_size = len(train_set)
    train_loss_list = []
    valid_loss_list = []

    for epoch_number in range(1 , config.EPOCHS + 1):
        train_loss = 0 
        valid_loss = 0 
        model.train()

        for fet , label in train_set:
            opt.zero_grad()
            output = model.forward(fet).flatten()
            t_loss = criterion(output , label)
            opt.zero_grad()
            t_loss.backward()
            opt.step()
            train_loss += t_loss.item()

        #validation process 
        with torch.no_grad():
            for fet , label in valid_set:
                output = model.forward(fet)
                output = torch.sigmoid(output).flatten()
                v_loss = criterion(output , label)
                valid_loss += v_loss.item()

        overall_t_loss = train_loss/train_data_size
        overall_v_loss = valid_loss/valid_data_size
        train_loss_list.append(overall_t_loss)
        valid_loss_list.append(overall_v_loss)

        print(f"epoch number = {epoch_number}, "
            f"train_loss = {overall_t_loss}, " 
            f"valid_loss = {overall_v_loss}")
    return train_loss_list, valid_loss_list

def main():
    train_set , test_set, valid_set = getdata()
    model = models.Model()
    criterion = nn.BCEWithLogitsLoss()
    opt = optim.Adam(
        model.parameters() , lr = config.LEARNING_RATE
    )
    model, train_loss , valid_loss = train(
        criterion = criterion, opt = opt,
        train_set = train_set, test_set = test_set
    )

if __name__ == "__main__":
    main()