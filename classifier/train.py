import pytorch_lightning as pl
from models import Model
import torch
import config


model = Model()
trainer = pl.Trainer(max_epochs = 20)
trainer.fit(model)