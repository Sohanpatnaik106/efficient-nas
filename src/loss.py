import torch
import torch.nn as nn

# TODO: Implemement training using contrastive loss as well

class ContrastiveLoss(nn.Module):

    def __init__(self, temperature):
        pass

    def forward(self, outputs, labels):
        pass

class NASLoss(nn.Module):

    def __init__(self, criterion_type = "cross-entropy", temperature = 0.7):
        super(NASLoss, self).__init__()
        
        self.temperature = temperature
        self.criterion_type = criterion_type

        if self.criterion_type == "cross-entropy":
            self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, outputs, labels):
        return self.criterion(outputs, labels)