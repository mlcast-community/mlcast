# new file with respect to original code

from src.mlcast.models.base import NowcastingModelBase
import pytorch_lightning as L
from src.mlcast.models.ldcast.data import LatentDataset
from torch.utils.data import DataLoader


class LDCast(NowcastingModelBase):
    def __init__(self, ldm_lightning, autoencoder):
        super().__init__()
        self.ldm_lightning = ldm_lightning
        self.autoencoder = autoencoder
    
    def fit(self, dataset):
        '''dataset should contains pairs of (inputs, true), with
        inputs.shape = (batch_size, 1, 4, 256, 256)
        true.shape = (batch_size, 1, 20, 256, 256)
        '''
        self.fit_autoencoder(dataset)
        self.fit_ldm(dataset)

    def fit_ldm(self, dataset):
        self.autoencoder.net.eval()

        latent_dataset = LatentDataset(dataset, self.autoencoder.net)
        dataloader = DataLoader(latent_dataset, batch_size=2)
        trainer = L.Trainer()
        trainer.fit(self.ldm_lightning, dataloader)
    
    def fit_autoencoder(self, dataset):
        pass

    def load(self):
        pass
    def predict(self, inputs):
        '''inputs.shape = (batch_size, 1, 4, 256, 256)'''
        latent_inputs = self.autoencoder.net.encode(inputs)
        latent_pred = self.ldm_lightning(latent_inputs)
        return self.autoencoder.net.decode(latent_pred)
        
    def save(self):
        pass
    