import abc

import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr

import mlcast


class NowcastingModelBase(abc.ABC):
    ds_scaling_params: xr.Dataset
    PytorchLightningModuleClass: pl.LightningModule
    timestep_length: np.timedelta64

    def __init__(self):
        self.pl_module = self.PytorchLightningModuleClass()

    @abc.abstractmethod
    def save(self, path):
        pass

    @abc.abstractmethod
    def load(self, path):
        pass

    @abc.abstractmethod
    def fit(self, da_rr):
        pass

    @abc.abstractmethod
    def predict(self, da_rr, duration):
        pass


class DummyNowcastingModule(pl.LightningModule):
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


class DummyNowcastingModel(NowcastingModelBase):
    PytorchLightningModuleClass = DummyNowcastingModule

    def fit(self, da_rr):
        dataset = mlcast.data.SingleScalarDataset(da_rr)
        data_loader = torch.data.DataLoader(dataset, batch_size=32)
        trainer = pl.Trainer()
        trainer.fit(self.pl_module, data_loader)
        # set the timestep length and scaling parameters from the dataset
        self.timestep_length = da_rr.time.diff("time")[0]
        self.ds_scaling_params = dataset.scaling_params

    def predict(self, da_rr, duration):
        dataset = mlcast.data.SingleScalarDataset(da_rr)
        data_loader = torch.data.DataLoader(dataset, batch_size=32)
        prediction_tensor = self.pl_module.predict(data_loader)

        n_timesteps = int(duration / self.timestep_length)

        dims = da_rr.dims
        dims.append("elapsed_time")
        coords = {dim: da_rr[dim] for dim in dims}
        coords["elapsed_time"] = [self.timestep_length * i for i in range(n_timesteps)]

        da_prediction = xr.DataArray(prediction_tensor, coords=da_rr.coords)
        return da_prediction


def main():
    da_rr = xr.open_dataset("radar_rainrate.nc")

    da_rr_train = da_rr.sel(time=slice("2019-01-01", "2020-01-01"))

    # training a nowcasting model from scratch
    nowcasting_model = mlcast.models.LDCast(learning_rate=0.001)
    nowcasting_model.fit(da_rr_train)

    # loading a pre-trained nowcasting model and making predictions
    da_rr_init = da_rr.sel(time=slice("2020-01-01", "2020-02-01"))
    nowcasting_model = mlcast.models.LDCast.load(path="model.pth")
    nowcasting_model.predict(da_rr_init, duration="PT1H")
