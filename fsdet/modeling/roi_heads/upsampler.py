import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

#TO_DO Implementing Unet-like upsampler 

class Upsampler(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.cfg = cfg
        self.input_shape = input_shape
        self.num_deconv = cfg.MODEL.ROI_BOX_HEAD.NUM_DECONV
        h = self.num_deconv*input_shape.height
        w = self.num_deconv*input_shape.width
        self._output_size = (1, h , w)
        self.deconvs = []
        for k in range(self.num_deconv):
            if k != self.num_deconv-1:
                deconv = nn.ConvTranspose2d(
                        in_channels=input_shape.channels, 
                        out_channels=input_shape.channels, 
                        kernel_size=2, 
                        stride=2, 
                        padding=0
                    )
                
            else:
                deconv = nn.ConvTranspose2d(
                        in_channels=input_shape.channels, 
                        out_channels=1, 
                        kernel_size=2, 
                        stride=2, 
                        padding=0
                    )
            self.add_module("deconv{}".format(k + 1), deconv)
            self.deconvs.append(deconv)
        for layer in self.deconvs:
            weight_init.c2_msra_fill(layer)

    def forward(self, x):
        #input (512*batch_size, 28, 7, 7)
        #output (512*bs, 1, 112, 112)
        for k, layer in enumerate(self.deconvs):
            if k!= len(self.deconvs):
                x = F.relu(layer(x))
            else:
                x = F.Tanh(layer(x))
        return x

    def output_size(self):
        return self._output_size

#TO_DO Implementing pixel-wise los
class L2maps(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.cfg = cfg
        self.input_shape = input_shape

    