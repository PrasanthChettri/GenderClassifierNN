import torch
from torch import nn , optim

BATCH_SIZE : int =  64
VOCAB = list(' abcdefghijklmnopqrstuvwxyz')
VOCAB_SIZE : int = len(VOCAB)
NAME_LEN  : int = 14
EPOCHS : int = 1
SPLIT_RATIO = [0.11, 0.15 , 1]
CLASSES = ['male' , 'female']
NUM_CLASSES = len(CLASSES)
LEARNING_RATE = 0.001
LSTM_HIDDEN_SIZE = 6
DROPOUT_PROBABILITY = 0.2
NUM_OF_LSTM_LAYERS = 2
CRITERION = nn.BCEWithLogitsLoss
OPTIMIZER = optim.Adam
MODEL_NAME = 'ltsm_model.pth'