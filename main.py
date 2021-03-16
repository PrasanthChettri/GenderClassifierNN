from fastapi import FastAPI
import torch
from torch.nn import functional as F
from typing import Dict
import os
from typing import List
from classifier.predict import Classifier
import schemas 

app = FastAPI()
NAME_LEN = 14
@app.get("/")
def root():
    return {"message": "Hello World"}



@app.post("/predict", response_model = List[schemas.prediction])
def predict_gender(names : schemas.model_in):
    prediction_list = []
    classifier = Classifier()
    for name in names.names : 
        prediction_list.append(
            classifier.predict(name)
        )
    return prediction_list


