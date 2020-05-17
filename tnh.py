import model
import torch
from torch import nn , optim
from dataset import Dataset

def main():
    #Batch_Size & lenght of a namevector
    batch_size = 1
    name_len = 20 

    #Loading feautres and labels training and validation
    dset = Dataset(0.3 , batch_size, name_len)
    t_getter = dset.t_get()
    v_getter = dset.v_get()
    l_of_split = dset.split

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu').upper()
    tnn = model.mod2(batch_size , name_len)
    tnn.train()
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(tnn.parameters() , lr = 0.001)
    sof = nn.Softmax(dim = 1)
    o_vval  = 1

    print("TRAINING ON {} :::::".format(device))

    #WRONG
    while o_vval > valid_loss:
        train_loss = 0 
        valid_loss = 0 
        tnn.train()
        k = 0 

        for dat in t_getter:
            fet , label = dat[0]
            opt.zero_grad()
            output = tnn.forward(fet)
            t_loss = criterion(output , label)
            opt.zero_grad()
            t_loss.backward()
            opt.step()
            train_loss +=t_loss.item()

        print ("LOSS --> " , (train_loss/l_of_split))

        #validation process 
        with torch.no_grad():
            for dat in v_getter:
                fet , label = dat[0]
                output = tnn.forward(fet)
                v_loss = criterion(output , label)
                valid_loss += v_loss.item()

        print ("valid_loss --->" , valid_loss/l_of_split)
        print ("epoch no---> " , k)
        k += 1
        o_vval = valid_loss 

    torch.save(tnn.state_dict() , "{}".format(o_vval))

if __name__  == '__main__':
    main()
