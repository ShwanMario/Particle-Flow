import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distrib
import torch.distributions.transforms as transform
from torch.distributions.normal import Normal
import matplotlib.animation as animation
from IPython.display import HTML
# Imports for plotting
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from torch.utils.data import Dataset, DataLoader
class MNIST_Dataset(Dataset):
    def __init__(self, image):
        super(MNIST_Dataset).__init__()
        self.image = image
    def __len__(self):
        return self.image.shape[0]
    def __getitem__(self, idx):
        return np.random.binomial(1, self.image[idx, :]).astype('float32')


def binary_loss(x_tilde, x):
    return F.binary_cross_entropy(x_tilde, x, reduction='none').sum(dim = 0)

def divide(q,n):
    i=[]
    n1=(1-q)/(1-q**n)
    i.append(n1)
    for k in range(n-1):
        n1*=q
        i.append(n1)
    return np.array(i)

def reconstruction_loss(x_tilde, x, num_classes=1, average=True):
    if (num_classes == 1):
        loss = binary_loss(x_tilde, x)
    else:
        loss = multinomial_loss(x_tilde, x)
    if (average):
        loss = loss.sum() / x.size(0)
    return loss

def multinomial_loss(x_logit, x,num_classes):
    batch_size = x.shape[0]
    # Reshape input
    x_logit = x_logit.view(batch_size, num_classes, x.shape[1], x.shape[2], x.shape[3])
    # Take softmax
    x_logit = F.log_softmax(x_logit, 1)
    # make integer class labels
    target = (x * (num_classes - 1)).long()
    # computes cross entropy over all dimensions separately:
    ce = F.nll_loss(x_logit, target, weight=None, reduction='none')
    return ce.sum(dim = 0)*100