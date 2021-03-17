from classifier.models import Model
import os
from classifier import  config
import torch
from typing import Dict , Union

class Classifier : 
	def __init__(self):
		self.model = Model(batch_size = 1)
		model_path = os.path.join('classifier', 'trained' , config.MODEL_NAME)
		self.model.load_state_dict(torch.load(model_path))

	def predict(self, name : str) -> Dict[str, Union[int ,str]]:
		word_vector = torch.Tensor(self.model.tokenize(name))
		logit = self.model.forward(
					word_vector.view(1 , *word_vector.shape)
				)
		probab = torch.sigmoid(logit).item()
		return {
			'name': name ,
			'percentage_female' : round(1 - probab , 2),
			'percentage_male' : round(probab , 2),
		}