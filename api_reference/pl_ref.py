import pytorch_lightning as pl
import torch


class NowcastingModel(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        # make simple 1x1 convolution model
        self.model = torch.nn.Sequential(torch.nn.Conv2d(1, 1, kernel_size=1), torch.nn.ReLU())

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        return loss


class NowcastingDataset(torch.Dataset):
    def __init__(self, path):
        # self.data = load_data(path)
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def main():
    train_dataset = NowcastingDataset(path="train_data")
    train_dataloader = torch.data.DataLoader(train_dataset, batch_size=32)
    model = NowcastingModel(learning_rate=0.001)

    trainer = pl.Trainer()
    trainer.fit(model, train_dataloader)

    eval_dataloader = ...  # as above
    trainer.predict(model, eval_dataloader)
