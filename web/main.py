import fastapi
from fastapi import FastAPI , Request
from fastapi import templating
from typing import Dict, List
from classifier.predict import Classifier
from web import schemas 

app = FastAPI()
templates = templating.Jinja2Templates(directory="web/templates")
@app.get("/")
def root(request : Request):
    return templates.TemplateResponse("index.html" , {'request' : request})

@app.get("/predict" , response_model= schemas.prediction)
def predict(request : Request, name : str):
    classifier = Classifier()
    return classifier.predict(name)

@app.post("/bulk_predict", response_model = List[schemas.prediction])
def bulk_predict(names : schemas.model_in):
    prediction_list = []
    classifier = Classifier()
    for name in names.names : 
        prediction_list.append(
            classifier.predict(name)
        )
    return prediction_list
