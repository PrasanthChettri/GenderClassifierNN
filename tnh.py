import model
import torch
from torch import nn , optim
from dataset import Dataset

def main():
    #Batch_Size & lenght of a namevector
    batch_size = 1 
    name_len = 20

    #Loading feautres and labels training and validation
    dset = Dataset(0.8, batch_size, name_len)
    t_getter , v_getter = dset.load()
    l_of_split = dset.split

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tnn = model.mod2(batch_size , name_len).cuda()
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(tnn.parameters() , lr = 0.001)
    sof = nn.Softmax(dim = 1)
    old_value = float('inf')

    print("training on {} :::::".format(device))
    print("training_population {}".format(dset.t_pop))
    print("validation_population {}".format(dset.v_pop))
    k = 0

    while True :
        train_loss = 0 
        valid_loss = 0 
        tnn.train()

        for dat in t_getter:
            fet , label = dat[0]
            opt.zero_grad()
            output = tnn.forward(fet)
            t_loss = criterion(output , label)
            opt.zero_grad()
            t_loss.backward()
            opt.step()
            train_loss +=t_loss.item()

        print ("\nLOSS --> " , (train_loss/l_of_split))

        #validation process 
        with torch.no_grad():
            for dat in v_getter:
                fet , label = dat[0]
                output = tnn.forward(fet)
                v_loss = criterion(output , label)
                valid_loss += v_loss.item()

        print ("valid_loss --->" , valid_loss/l_of_split)
        print ("epoch no---> " , k)
        print()

        k += 1
        if old_value < valid_loss :
            break
        old_value = valid_loss 
    torch.save(tnn.state_dict() , "weights:{}.pth".format(int(old_value)))

if __name__  == '__main__':
    main()
