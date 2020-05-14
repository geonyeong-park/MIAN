import torch.nn as nn
import math
import torch

def SVD_entropy(feature, k):
    _, sigma, _ = torch.svd(feature)
    sigma_normalized = sigma / torch.sum(sigma)
    for ld in sigma:
        en_transfer = Entropy(sigma_normalized[ :k])
        en_discrim = Entropy(sigma_normalized[k: ])
    return en_transfer, en_discrim

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy)
    return entropy
