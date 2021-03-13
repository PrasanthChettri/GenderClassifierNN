import pytorch_lightning as pl
from models import Model
from dataset import get_data
import config

def main():
    model = Model()
    train_d, valid_d ,test_d = get_data(config.BATCH_SIZE) 
    trainer = pl.Trainer()
    trainer.fit(Model, train_d, valid_d)

if __name__ == "__main__":
    main()
