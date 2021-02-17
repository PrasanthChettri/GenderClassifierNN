from fastapi import FastAPI
import torch
from torch.nn import functional as F
from typing import Dict
import os


from classifier import tokenizer , model
import schemas 

app = FastAPI()
NAME_LEN = 12

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/predict") #, response_model = Dict[str, float])
def predict(names : schemas.model_in):
    '''
        name_len maximum we support
    '''
    t_obj = tokenizer.Tokenizer(NAME_LEN)
    batch_size = 1  # process One by One
    tnn = model.mod2(batch_size , NAME_LEN)
    tnn.eval()
    tnn.load_state_dict(torch.load("weights_48.pth", map_location=torch.device('cpu')))

    result = {}

    with torch.no_grad():
        for name in names.name:
            name_token = torch.Tensor(t_obj.tkniz(name)).float()
            output = tnn.forward(name_token.unsqueeze(1))
            output = F.softmax(output[0] , dim = 1)
            '''
                get the output in the normal dataype form
            '''
            male , female = map(lambda x : x.item() , output[0])
            result.update({
                        name : {
                            'male'   : male , 
                            'female' : female
                            }
                    })
    return result