import fastapi
from fastapi import FastAPI , Request, Query
from fastapi import templating
from typing import Dict, List 
from classifier.predict import Classifier

app = FastAPI()
templates = templating.Jinja2Templates(directory="web/templates")

from pydantic import BaseModel

#####################
# PYDANTIC SCHEMAS
#####################
class prediction_out(BaseModel):
    name : str
    percentage_female : float
    percentage_male : float


@app.get("/")
def root(request : Request):
    """
    returns an html front-end
    """
    return templates.TemplateResponse("index.html" , {'request' : request})

@app.get("/predict" , response_model= prediction_out)
def predict(request : Request, name : str = Query(default = '' , title='Name')):
    """
    Runs the classifier once to predict a given name
    Parameters
    -----------
    name := single name that has to be predicted 

    Returns
    -----------
    - Data as 
    {
        name
        confidence/probabilty of being male
        confidence/probabilty of being female
    }
    """
    classifier = Classifier()
    return classifier.predict(name)

@app.post("/bulk_predict", response_model = List[prediction_out])
def bulk_predict(names : List[str] = Query(default = [] , title="List of Names")):
    """
    predict more than one name at a time, runs the classifier to predict more than one name 

    Parameters
    -----------
    - name := list of names that has to be predicted 

    Returns
    -----------
    - Data as 
    List of 
    {
        name
        confidence/probabilty of being male
        confidence/probabilty of being female
    }
    """
    prediction_list = []
    classifier = Classifier()
    for name in names : 
        prediction_list.append(
            classifier.predict(name)
        )
    return prediction_list