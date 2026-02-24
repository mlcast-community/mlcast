from torch.utils.data import Dataset
import torch

class LatentDataset(Dataset):
    def __init__(self, dataset, autoencoder):
        super().__init__()

        self.autoencoder = autoencoder
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        with torch.no_grad():
            inputs, true = self.dataset[idx]
            latent_inputs = self.autoencoder.encode(inputs)
            latent_true = self.autoencoder.encode(true)

        # until here, the first dimension of latent_inputs and latent_true is the 'batch dimension'
        # if idx is a list, keep this batch dimension along this list
        # if idx is not a list, this batch dimension is 1 and needs to be removed because the dataloader will repeatedly call __getitem__ and add an extra dimension for the batch dimension
        if not isinstance(idx, list):
            latent_inputs = latent_inputs[0]
            latent_true = latent_true[0]
        
        return (latent_inputs, latent_true)