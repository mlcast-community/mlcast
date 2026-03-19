from torch.utils.data import Dataset, random_split, DataLoader
import torch
import pytorch_lightning as pl
from tqdm import tqdm

class LatentDataset(Dataset):
    def __init__(self, sampled_radar_dataset, autoencoder, autoenc_time_ratio = 4):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.autoencoder = autoencoder.to(self.device)
        self.dataset = sampled_radar_dataset
        self.autoenc_time_ratio = autoenc_time_ratio

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        with torch.no_grad():
            sequence = self.dataset[idx]
            x = sequence[:, :self.autoenc_time_ratio]
            y = sequence[:, self.autoenc_time_ratio:]

            # for some reason, Gabriele put the time axis before the channel axis, change this
            #x = x.swapaxes(0, 1).to(self.device)
            #y = y.swapaxes(0, 1).to(self.device)
            
            #latent_x = self.autoencoder.encode(x)
            #latent_y = self.autoencoder.encode(y)

        return x, y #latent_x, latent_y

class AutoencoderDataset(Dataset):
    '''
    shape of one sample of sampled_radar_dataset = (1, 24, 1,) + spatial_shape
    But, for the LDCast autoencoder, we want to have samples of (1, 4, 1,) + spatial_shape
    So 1 sample of sampled_radar_dataset is partitioned in 6 samples for the autoencoder
    '''
    def __init__(self, sampled_radar_dataset, autoenc_time_ratio = 4):
        super().__init__()
        self.srd = sampled_radar_dataset
        self.autoenc_time_ratio = autoenc_time_ratio
        self.samples_ratio = int(self.srd.steps / self.autoenc_time_ratio) # is 6 in the usual case where steps = 24 and autoenc_time_ratio = 4
        
    def __len__(self):
        return self.samples_ratio * len(self.srd)

    def __getitem__(self, idx):
        '''
        when given idx between 0 and 6 * len(srd) - 1, one has first to find in which sample of srd we are (the index of this sample is index_srd)
        then, within this sample, one has to find in which partition of this sample we are (this is given by index_in_srd_sample)
        '''
        index_srd = idx // self.samples_ratio
        index_in_srd_sample = idx - index_srd * self.samples_ratio
        x = self.srd[index_srd].reshape(self.samples_ratio, 1, self.autoenc_time_ratio, self.srd.w, self.srd.h)[index_in_srd_sample]

        # for some reason, Gabriele put the time axis before the channel axis, change this
        # x = x.swapaxes(0, 1)

        # for the autoencoder, y is equal to x
        y = x
        
        return x, y

class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, train_ratio = 0.6, val_ratio = 0.2, **dataloader_kwargs):
        super().__init__()
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - self.train_ratio - self.val_ratio

        train_ds, val_ds, test_ds = random_split(dataset, [self.train_ratio, self.val_ratio, self.test_ratio])
        self.train_dataset = train_ds
        self.val_dataset = val_ds
        self.test_dataset = test_ds

        self.dataloader_kwargs = dataloader_kwargs
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle = True, **self.dataloader_kwargs)
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle = False, **self.dataloader_kwargs)
        
    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle = False, **self.dataloader_kwargs)

def load_in_memory(dataset):
    ds = []
    for i in tqdm(range(len(dataset)), desc = 'Loading data in memory'):
        # append the sample and create an extra dimension (to be the batch dimension)
        ds.append(dataset[i][None].to('cpu'))
    return torch.cat(ds, axis = 0)