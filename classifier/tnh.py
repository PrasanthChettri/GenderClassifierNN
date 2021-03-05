import torch 
from torch import nn , optim
from torch.nn import functional as F
import os
from tokenizer import Tokenizer

from dataset import getdata
import models



def main():
    '''
        main func trains the model
        and saves the data
    '''

    #Batch_Size & lenght of a namevector
    batch_size =  102
    vocab_size = Tokenizer.VOCAB_SIZE
    name_len = 14
    split_ratio = [0.85 , 0.11 , 0.04]
    '''
        we wait for the model if validation accuracy is going down
        before early stopping
    '''
    EPOCHS = 50

    #Loading feautres and labels training and validation
    train_set , valid_set , test_set = getdata(batch_size , name_len , split_ratio)
    '''
        Populate testing and validation with this many 
        items
    '''

    model = models.mod2(batch_size , name_len , vocab_size).cuda()
    criterion = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters() , lr = 0.00169)
    '''
        training phase
    '''
    print("Training  .. ")
    
    for epoch_number in range(1 , EPOCHS + 1):
        train_loss = 0 
        valid_loss = 0 
        model.train()
        for fet , label in train_set:
            opt.zero_grad()
            #dicard leftover (test_data % batch_size)
            if fet.shape[0] != batch_size : break
            output = model.forward(fet).flatten()
            t_loss = criterion(output , label)
            opt.zero_grad()
            t_loss.backward()
            opt.step()
            train_loss += t_loss.item()


        #validation process 
        with torch.no_grad():
            for fet , label in valid_set:
                if fet.shape[0] != batch_size : break
                output = model.forward(fet).flatten()
                v_loss = criterion(output , label)
                valid_loss += v_loss.item()

        print(f"epoch number = {epoch_number} , train_loss = {train_loss} , valid_loss = {valid_loss}")


    os.chdir('..')
    torch.save(model.state_dict(), "weights\weights.pth".format())
    '''
        testing phase
    '''


if __name__  == '__main__':
    main()