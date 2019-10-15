import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distrib
import torch.distributions.transforms as transform
import matplotlib.animation as animation
from IPython.display import HTML
# Imports for plotting
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
from torchvision import datasets
from torchvision import transforms
from flow import *
from utils import *
import argparse
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description="VAE with Normalizing Flow")
parser.add_argument("--batch_size",type=int,
                    default=1,
                    help="""batch size for training""")
parser.add_argument("--learning_rate",type=float,
                    default=0.001,
                    help="""learning rate""")
parser.add_argument("--device",type=str,
                    default="cuda",
                    help="""cpu or cuda""")
args = parser.parse_args()

batch_size =args.batch_size
with open('./data/data.pkl', 'rb') as file_handle:
    data = pickle.load(file_handle)

train_image = data['train_image']
train_label = data['train_label']
test_image = data['test_image']
test_label = data['test_label']

batch_size = args.batch_size
train_data = MNIST_Dataset(train_image)

train_data_loader = DataLoader(train_data, batch_size = batch_size,
                               shuffle = True)

test_data = MNIST_Dataset(test_image)
test_data_loader = DataLoader(test_data, batch_size = batch_size)

tens_t = transforms.ToTensor()
#train_dset = datasets.MNIST('./data', train=True, download=True, transform=tens_t)
#train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True)
#test_dset = datasets.MNIST('./data', train=False, transform=tens_t)
#test_loader = torch.utils.data.DataLoader(test_dset, batch_size=batch_size, shuffle=True)

num_classes = 1
# Number of hidden and latent
n_hidden = 512
n_latent = 2
fixed_batch= next(iter(test_data_loader))
nin = fixed_batch.shape[1]

encoder=construct_encoder()
decoder=construct_decoder()
particle=particle_flow(encoder,decoder,1)
optimizer = optim.Adam(particle.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9999)
train(particle,optimizer=optimizer,scheduler=scheduler,train_loader=train_data_loader,model_name='VAE_particle_flow',epochs=1000)