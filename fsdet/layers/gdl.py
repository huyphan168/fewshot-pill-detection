import torch
import torch.nn as nn
from torch.autograd import Function


class GradientDecoupleLayer(Function):

    @staticmethod
    def forward(ctx, x, _lambda):
        ctx._lambda = _lambda
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output * ctx._lambda
        return grad_output, None


class AffineLayer(nn.Module):
    def __init__(self, num_channels, bias=False):
        super(AffineLayer, self).__init__()
        weight = torch.FloatTensor(1, num_channels, 1, 1).fill_(1)
        self.weight = nn.Parameter(weight, requires_grad=True)

        self.bias = None
        if bias:
            bias = torch.FloatTensor(1, num_channels, 1, 1).fill_(0)
            self.bias = nn.Parameter(bias, requires_grad=True)

    def forward(self, X):
        out = X * self.weight.expand_as(X)
        if self.bias is not None:
            out = out + self.bias.expand_as(X)
        return out


def decouple_layer(x, _lambda):
    return GradientDecoupleLayer.apply(x, _lambda)
#Testing unit for GradientDecopuledLayer
# if __name__ == '__main__':
#     _cfg = {
#         "MODEL": {
#             "RPN": {
#                 "IN_FEATURES": ["r1", "r2"]
#             },
#             "ROI_HEADS": {
#                 "IN_FEATURES": ["r2", "r3"]
#             },
#             "GDL": {
#                 "rpn_scale": 0.0,
#                 "rcnn_scale": 0.1
#             }
#         }
#     }
#     cfg = munchify(_cfg)
#     gdl = GradientDecopuledLayer("rpn", cfg, {'r1': {"channels":512}, 'r2': {"channels":512}, 'r3': {"channels":128}})
#     outs = gdl.forward({"r1": torch.randn(4,512,16,16), "r2": torch.randn(4,512,16,16), "r3": torch.randn(4,128,16,16)})
#     new = outs["r1"] + outs["r2"] 
    
#     loss = new.sum()**2 - 5
#     loss.backward()
#     print(gdl.affine_layers['r2'][0].grad)



