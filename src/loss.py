import torch
import torch.nn as nn


# TODO: Implemement the loss module

class NASLoss(nn.Module):

    def __init__(self, temperature = 0.7):
        super(NASLoss).__init__(self)
        self.temperature = temperature

    def forward(self):

        pass