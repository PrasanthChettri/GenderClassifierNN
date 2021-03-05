from fastapi import FastAPI
import torch
from torch.nn import functional as F
from typing import Dict
import os


from classifier import tokenizer  , models
import schemas 

app = FastAPI()
NAME_LEN = 14
VOCAB_SIZE = tokenizer.Tokenizer.VOCAB_SIZE

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
    model = models.mod2(batch_size , NAME_LEN , VOCAB_SIZE)
    model.load_state_dict(torch.load("weights\\weights_532.87.pth"))
    model.eval()

    result = {}

    with torch.no_grad():
        for name in names.name:
            name_token = torch.Tensor(t_obj.tkniz(name)).float()
            output = model.forward(name_token)
            '''
                get the output in the normal dataype form
            '''
            return torch.sigmoid(output).item()
            return output
            result.update({
                        name : {
                            'male'   : male , 
                            'female' : female
                            }
                    })
    return result
