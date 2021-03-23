from classifier.models import Model
import os
from classifier import  config
import torch
from typing import Dict , Union , List

class Classifier : 
	def __init__(self , classifications : int):
		'''
		intialises the model 
		@param : classifications - number of names to classify
		'''
		self.classifications =  classifications
		self.model = Model(batch_size = classifications)
		model_path = os.path.join('classifier', 'trained' , config.MODEL_NAME)
		self.model.load_state_dict(torch.load(model_path))

	def classify(self, names : List[str]) -> List[Dict[str, Union[float ,str]]]:
		'''
		initialises 
		@param : classifications - the list of names to classify
		@returns : output - the list of classifications 
		'''
		assert len(names) == self.classifications, "lenght of the list provided does not match the num of classifications"
		word_vector = list( map(self.model.tokenize, names) ) 
		logit = self.model.forward(torch.Tensor(word_vector))
		confidences = torch.sigmoid(logit).flatten().tolist() #returns uncertainty of being male
		output = []
		for name, confidence in zip(names , confidences) : 
			output.append({
			'name': name ,
			'percentage_female' : round(1 - confidence , 2),
			'percentage_male' : round(confidence , 2),
		})
		return output 