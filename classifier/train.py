import pytorch_lightning as pl
from classifier.models import Model
import torch
from classifier import config
import os


def main():
    model = Model(batch_size = config.BATCH_SIZE)
    trainer = pl.Trainer(max_epochs = config.EPOCHS, default_root_dir = 'classifier')
    trainer.fit(model)
    train_dir = os.path.join('classifier', 'trained')
    if not os.path.exists(train_dir) :
        os.mkdir(train_dir)

    torch.save(model.state_dict(),
        os.path.join(train_dir, config.MODEL_NAME)
    )
    trainer.test(model)

if __name__ == "__main__":
    main()
