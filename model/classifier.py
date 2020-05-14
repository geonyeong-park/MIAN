import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class Predictor(nn.Module):
    def __init__(self, prev_feature_size, num_classes):
        super(Predictor, self).__init__()
        self.fc = spectral_norm(nn.Linear(prev_feature_size, num_classes))

    def forward(self, x):
        x = self.fc(x)
        return x
