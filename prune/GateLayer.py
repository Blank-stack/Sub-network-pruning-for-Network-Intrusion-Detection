import torch
import torch.nn.functional as F
import torch.nn as nn
from prune.GumbelSigmoid import GumbelSigmoidMask


class GateMLP(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, init_mask=0, device=None, dtype=None):
        super(GateMLP, self).__init__(in_features, out_features, bias=bias)
        self.mask = GumbelSigmoidMask(self.weight.shape, init_mask)

    # freeze 含义为将浮点数 mask 约束为 1 或 0，若二者均不启用，则剪枝后的网络仍可为密集全网
    def forward(self, input, pruning=False, freeze=False, flip=False):
        mask = None
        if pruning:
            mask = self.mask.sample(hard=True)
        elif freeze:
            mask = self.mask.fix_mask_after_pruning()

        if mask is not None:
            if flip: mask = 1 - mask
            if pruning:
                return F.linear(input, self.weight.detach().to(input.device) * mask.to(input.device),
                                self.bias.to(input.device))
            else:
                return F.linear(input, self.weight.to(input.device) * mask.detach().to(input.device),
                                self.bias.to(input.device))
        else:
            return F.linear(input, self.weight.to(input.device), self.bias.to(input.device))


# class GateConv2d(nn.Conv2d):
#     def __init__(self, in_features, out_features, kernel_size=3, stride=1, padding=0, bias=True,
#                  dilation=1, groups=1):
#         super(GateConv2d, self).__init__(in_features, out_features, kernel_size,
#                                          stride=stride, padding=padding, bias=bias,
#                                          dilation=dilation, groups=groups)
#         self.mask = GumbelSigmoidMask(self.weight.shape)
#
#     def forward(self, input, pruning=False, freeze=False):
#         mask = None
#         if pruning:
#             mask = self.mask.sample(hard=True)
#
#         if freeze:
#             mask = self.mask.fix_mask_after_pruning()
#
#         if mask is not None:
#             return F.conv2d(input, self.weight * mask.to(input.device), self.bias,
#                             self.stride, self.padding, self.dilation, self.groups)
#         else:
#             return F.conv2d(input, self.weight, self.bias,
#                             self.stride, self.padding, self.dilation, self.groups)
