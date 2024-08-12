import torch
import torch.nn.functional as F
import torch.nn as nn


class GumbelSigmoidMask(nn.Module):
    def __init__(self, mask_shape, init_mask=0):
        super(GumbelSigmoidMask, self).__init__()
        self.sigmoid = nn.Sigmoid()
        # self.gumbel_pi = nn.Parameter(1.5*torch.ones(mask_shape))
        # # self.gumbel_pi = nn.Parameter(1 + torch.rand(mask_shape))  # 生成1到2之间的随机数
        self.gumbel_pi = nn.Parameter(init_mask * torch.ones(mask_shape))


    def sample(self, hard=False, noise_wts=0.1000):
        res = self.sigmoid(self.gumbel_pi)
        noise = res.new_empty(list(res.shape)).uniform_(-1, 1)
        noise = noise_wts * noise
        res = res + noise

        if hard:
            res = ((res > 0.5).type_as(res) - res).detach() + res
        return res

    # # 源代码，重复使用sigmoid导致有问题，noise影响过大
    # def sample(self, tau=1., eps=1e-10, hard=False, flip=False):
    #     logits = self.sigmoid(self.gumbel_pi)
    #     if flip:
    #         logits = 1. - logits
    #     uniform = logits.new_empty([2] + list(logits.shape)).uniform_(0, 1)
    #
    #     noise = -((uniform[1] + eps).log() / (uniform[0] + eps).log() + eps).log()
    #     res = torch.sigmoid((logits + noise) / tau)
    #
    #     if hard:
    #         res = ((res > 0.5).type_as(res) - res).detach() + res
    #     return res
    # # 源代码

    def fix_mask_after_pruning(self):
        fixed_mask = torch.zeros_like(self.gumbel_pi)
        fixed_mask[self.gumbel_pi >= 0] = 1.
        return fixed_mask
