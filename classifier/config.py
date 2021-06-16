from torch import nn , optim

EPOCHS : int = 20 # Number of epochs to train on
BATCH_SIZE : int =  64 #batch_size for training
VOCAB = list('abcdefghijklmnopqrstuvwxyz') #vocab
VOCAB_SIZE : int = len(VOCAB) #len of the vocab
NAME_LEN  : int = 14 #lenth of input
SPLIT_RATIO = [0.90, 0.96 , 1] #elem 0 = training ,elem 1 - elem 0  = valid, 1 - elem 1 = testing
CLASSES = ['male' , 'female'] #classes for classification
NUM_CLASSES = len(CLASSES) # number of classes
LEARNING_RATE = 0.0016 #learning rate for the optimizer
LSTM_HIDDEN_SIZE = 6 #size of hidden and cell state
DROPOUT_PROBABILITY = 0.2 # p(Dropout)
NUM_OF_LSTM_LAYERS = 2 # number of layers in a stacked LSTM
CRITERION = nn.BCEWithLogitsLoss # Loss function
OPTIMIZER = optim.Adam # Optimizer
MODEL_NAME = 'lstm_model.pth' # name for saved the weights after training
NUM_WORKERS = 12
