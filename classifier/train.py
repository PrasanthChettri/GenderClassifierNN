import pytorch_lightning as pl
from classifier.models import Model
import torch
from classifier import config
import os


def main():
    model = Model(batch_size = config.BATCH_SIZE)
    trainer = pl.Trainer(max_epochs = 30,
                gpus = 1, default_root_dir = 'classifier')
    trainer.fit(model)
    torch.save(model.state_dict(),
            os.path.join('classifier', 'trained' , config.MODEL_NAME)
        )
    trainer.test(model)

if __name__ == "__main__":
    main()