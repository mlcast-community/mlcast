import torch
import torch.nn as nn

class Antialiaser(nn.ModuleDict):
    def __init__(self):

        super().__init__()
        
        # construct the kernel (symmetric in both directions), shape = (5, 5)
        (x, y) = torch.meshgrid(torch.arange(-2, 3), torch.arange(-2, 3), indexing = 'ij')
        kernel = torch.exp(-0.5*(x**2+y**2)/(0.5**2))
        kernel /= kernel.sum()

        # the convolution will be done on x (shape = (1, autoenc_time_ratio) + spatial_shape)
        # so treat the autoenc_time_ratio as one axis of the convolution -> Conv3d
        # but we do not want to convolve on this axis, so use kernl_size = 1 and padding = 0 on this axis
        self.conv = nn.Conv3d(1, 1, bias = False, kernel_size = (1, 5, 5), padding = (0, 2, 2))

        # set the weights to be those of the kernel
        self.conv.weight = nn.Parameter(kernel[None, None, None], requires_grad = False)
        
    def forward(self, x):

        # factor is 1 in the bulk of x, but is greater than 1 near the border to accounr that less values were used in the convolution
        # recomputed each time because the image shape could change
        factor = self.conv(torch.ones(x.shape, device = x.device))

        return self.conv(x) / factor