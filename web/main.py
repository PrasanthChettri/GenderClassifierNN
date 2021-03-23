import fastapi
from fastapi import FastAPI , Request, Query
from fastapi import templating
from typing import Dict, List 
from classifier.classify import Classifier

app = FastAPI()
templates = templating.Jinja2Templates(directory="web/templates")

from pydantic import BaseModel

#####################
# PYDANTIC SCHEMA
#####################
class classification_out(BaseModel):
    name : str
    percentage_female : float
    percentage_male : float



@app.get("/")
def root(request : Request):
    """
    returns an html front-end
    """
    return templates.TemplateResponse("index.html" , {'request' : request})

@app.get("/classify" , response_model= classification_out)
def classify(
    request : Request, name : str = Query(default = '' , title='Name')
):
    """
    Runs the classifier once to classify a given name
    Parameters
    -----------
    name := single name that has to be classifyed 

    Returns
    -----------
    - Data as 
    {
        name
        confidence/probabilty of being male
        confidence/probabilty of being female
    }
    """
    classifier = Classifier(classifications=1)
    response =  classifier.classify([name])[-1]
    return response 

@app.post("/bulk_classify", response_model = List[classification_out])
def bulk_classify(
    names : List[str] = Query(default = [] , title="List of Names")
):
    """
    classify more than one name at a time, runs the classifier to classify more than one name 

    Parameters
    -----------
    - name := list of names that has to be classifyed 

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
    num_of_names = len(names)
    classifier = Classifier(classifications= num_of_names)
    response = classifier.classify(names)
    return response