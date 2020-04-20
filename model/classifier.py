import torch.nn as nn
import torch.nn.functional as F

class Predictor(nn.Module):
    def __init__(self, num_classes):
        super(Predictor, self).__init__()
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x
