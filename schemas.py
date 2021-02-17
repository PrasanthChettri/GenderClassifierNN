from pydantic import BaseModel , validator
from typing import List

class model_in(BaseModel):
    name : List[str]
    @validator('name', each_item=True)
    def validate_name(cls , v):
        
        if ' ' in v : 
            '''
                model takes in only the first name
            '''
            raise ValueError(f"Only provide the first name for {v}")
        elif not v.isalpha():
            '''
                model only takes alphabets
            '''
            raise ValueError(f"Name should not contain special characters for name {v}")
        else :
            return v
