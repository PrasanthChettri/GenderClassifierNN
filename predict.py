import torch
from torch.nn import functional as F
from sys import argv
import  model  
from tnh import train as pr

to_p= argv[1:]
print (to_p)

al =  list(' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
dicte = {}

for i , j  in enumerate(al):
   dicte[j]  = i 
arr = []

for i in to_p:
    k = [[dicte[ii] for ii in i]]
    print (k)
    arr.append(pr.pad(k))

tnn = model.mod()
tnn.eval()
tnn.load_state_dict(torch.load("E35,700S.pth"))

with torch.no_grad():

    for i in arr:
        output = tnn.forward(i.unsqueeze(1))
        output = F.softmax(output , dim = 1)
        male , female = output[0] 
        print ("female : {} % probabilty \nmale : {} % probabilty ".format(male.item() , female.item()))

