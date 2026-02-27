# from https://github.com/MeteoSwiss/ldcast/blob/master/ldcast/models/diffusion/ema.py
'''
modifications following https://medium.com/@heyamit10/exponential-moving-average-ema-in-pytorch-eb8b6f1718eb
In the original code, EMA was a subclass of nn.Module, in order to register the parameters as buffers and (I guess) to have them saved automatically when saving the model. This made things a bit tricky and messy, because '.'-characters naturally appear in the names of model parameters, while they cannot appear in buffers names... Instead, the EMA weights will have to be saved with torch.save(ema.shadow) and loaded with ema.shadow = torch.load('ema_weights')'''

import torch
from torch import nn

class EMA():
    def __init__(self, model, decay=0.9999, use_num_updates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.model = model
        self.decay = decay
        self.shadow = {}                                  # to store EMA weights
        self.backup = {}                                  # to store the model weights when we replace them by ema weights
        self.num_updates = 0 if use_num_updates else -1   # for dynamical decay

        self.register()
        
    def register(self):
        '''initialize the ema weights with the model weights'''
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        '''update the shadow parameters'''

        # use dynamical decay if use_num_updates was true in __init__
        decay = self.decay
        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay,(1 + self.num_updates) / (10 + self.num_updates))
            
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        '''apply shadow (EMA) weights to the model'''
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        '''restore original model weights from backup'''
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

    def load(self, filename):
        '''load the ema (shadow) weights parameters'''
        self.shadow = torch.load(filename)
        self.decay = self.shadow.pop('decay')
        self.num_updates = self.shadow.pop('num_updates')

    def save(self, filename):
        '''save the ema (shadow) weights parameters'''
        self.shadow['decay'] = self.decay
        self.shadow['num_updates'] = self.num_updates
        torch.save(self.shadow, filename)

        self.shadow.pop('decay')
        self.shadow.pop('num_updates')