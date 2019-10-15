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
from utils import divide,binary_loss,multinomial_loss,reconstruction_loss
# Define grids of points (for later plots)


class Flow(transform.Transform, nn.Module):

    def __init__(self):
        transform.Transform.__init__(self)
        nn.Module.__init__(self)


    # Hacky hash bypass
    def __hash__(self):
        return nn.Module.__hash__(self)


class construct_encoder(nn.Module):
    def __init__(self, latent_size=64,n_hidden=450,kernel_size=5,padding=2,channel=(16,32,32),stride=2,activation=F.softplus,view_size=512,input_channel=1):
        super(construct_encoder, self).__init__()
        self.latent_size=latent_size
        self.n_hidden=n_hidden
        self.activation=activation
        self.channel=channel
        self.stride=stride
        self.padding=padding
        self.kernel_size=kernel_size
        self.conv1=nn.Conv2d(input_channel, self.channel[0], self.kernel_size, stride=self.stride, padding=self.padding)
        self.conv2=nn.Conv2d(self.channel[0], self.channel[1], self.kernel_size, stride=self.stride, padding=self.padding)
        self.conv3=nn.Conv2d(self.channel[1], self.channel[2], self.kernel_size, stride=self.stride, padding=self.padding)
        self.fc1=nn.Linear(view_size, self.n_hidden)
        self.fc_mu=nn.Linear(self.n_hidden,self.latent_size)
        self.fc_sigma=nn.Linear(self.n_hidden,self.latent_size)
    def forward(self,inputs):
        conv1 = self.activation(self.conv1(inputs))
        conv2 = self.activation(self.conv2(conv1))
        conv3 = self.activation(self.conv3(conv2))
        view = conv3.view(-1,conv3.shape[1]*conv3.shape[2]*conv3.shape[3])
        fc1 = self.fc1(view)
        output = self.activation(fc1)
        mu=self.fc_mu(output)
        sigma=self.activation(self.fc_sigma(output))
        return mu,sigma

class construct_decoder(nn.Module):
    def __init__(self, stride=2,latent_size=64, n_hidden=450,n_fc_layer=512,kernel_size=5,padding=2,channel=(32,32,16,1),activation=F.softplus):
        super(construct_decoder, self).__init__()
        self.n_hidden = n_hidden
        self.activation = activation
        self.channel = channel
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.n_fc_layer=n_fc_layer
        self.fc1=nn.Linear(latent_size, self.n_hidden)
        self.fc2=nn.Linear(self.n_hidden, self.n_fc_layer)
        self.deconv1=nn.ConvTranspose2d(self.channel[0], self.channel[1], self.kernel_size, stride=self.stride, padding=self.padding)
        self.deconv2=nn.ConvTranspose2d(self.channel[1], self.channel[2], self.kernel_size, stride=self.stride, padding=self.padding, output_padding=1)
        self.deconv3=nn.ConvTranspose2d(self.channel[2], self.channel[3], 5, stride=self.stride, padding=self.padding, output_padding=1)
    def forward(self,inputs):
        fc1 = self.activation(self.fc1(inputs))
        fc2 = F.softplus(self.fc2(fc1))
        reshape = fc2.view((-1, self.channel[0], 4, 4))
        deconv1 = F.softplus(self.deconv1(reshape))
        deconv2 = F.softplus(
            self.deconv2(deconv1))
        deconv3 = F.softplus(
            self.deconv3(deconv2))
        return deconv3
class particle_flow(Flow):
    def __init__(self,encoder,decoder,n_samples=1,q=6/5,n_lambda=29):
        super(particle_flow, self).__init__()
        self.n_samples=n_samples
        self.intervals=divide(q,n_lambda)
        self.encoder=encoder
        self.decoder=decoder
    def get_jacobian(self,z,x_tilde):
        z_repeat=z.repeat(x_tilde.shape[-1],1,1)
        x_tilde_repeat=self.decoder(z_repeat)
        x_tilde_repeat=x_tilde_repeat.reshape((x_tilde.shape[-1],x_tilde.shape[-1]))
        mat_repeat=torch.eye(x_tilde.shape[0])
        z_repeat.retain_grad()
        x_tilde_repeat.backward(mat_repeat,retain_graph=True)
        return z_repeat.grad.data

    def _call(self, x):
        mu,sigma=self.encoder(x)
        p=torch.diag(sigma[0]**2)
        sampler=Normal(loc=mu,scale=sigma**2)
        epsilon=sampler.sample((self.n_samples,))
        z=sigma*epsilon+mu
        x_tilde=self.decoder(z)
        x_flatten_tilde = x_tilde.flatten()
        bar_eta=mu.reshape((mu.shape[0],-1,mu.shape[1]))
        r = torch.diag(x_flatten_tilde * (1 - x_flatten_tilde))
        gamma=0
        alpha=1
        eta_1=torch.zeros(self.n_samples,1,z.shape[-1])
        eta_1=z[0]
        lambda_1=0
        eta_1.squeeze(1)
        for j in range(self.intervals.shape[0]):
            lambda_1 = lambda_1 + self.intervals[j]
            print(j,'th interval',lambda_1,self.intervals[j])
            h_eta=self.decoder(bar_eta)
            h_eta=h_eta.flatten()
            H=self.get_jacobian(bar_eta,h_eta).squeeze()
            bar_eta=bar_eta.squeeze(1)
            square_mat=lambda_1*H@p@(H.transpose(1,0))+r
            A_j_lambda=-1/2*p@H.transpose(1,0)@square_mat.inverse()@H
            temp=(torch.eye(bar_eta.shape[-1])+lambda_1*A_j_lambda)@p@H.transpose(1,0)@r.inverse()@x.reshape((784,1))+A_j_lambda@mu.transpose(1,0)
            b_j_lambda=(torch.eye(bar_eta.shape[-1])+2*lambda_1*A_j_lambda)@temp
            bar_eta=(bar_eta.transpose(1,0)+self.intervals[j]*(A_j_lambda@bar_eta.transpose(1,0)+b_j_lambda)).transpose(1,0)
            eta_1=(eta_1.transpose(1,0)+self.intervals[j]*(A_j_lambda@eta_1.transpose(1,0)+b_j_lambda)).transpose(1,0)
            alpha_mat = torch.eye(A_j_lambda.shape[0]) + self.intervals[j] * A_j_lambda
            alpha=alpha* torch.abs(torch.det(alpha_mat))
        gamma =gamma+ torch.log(alpha)
        x_reconstruction=self.decoder(eta_1)
        x_reconstruction=x_reconstruction.flatten()
        x_reconstruction=torch.sigmoid(x_reconstruction)
        log_x_g_z=-reconstruction_loss(x_reconstruction.reshape((self.n_samples,1,x.shape[-2],x.shape[-1])),x)## negative
        log_z=torch.sum(-0.5*eta_1**2,-1)
        log_z_g_x=torch.sum(-torch.log(sigma)-0.5*(z-mu)*(z-mu)*(sigma.reciprocal()**2))
        elbo=log_x_g_z+log_z-log_z_g_x+gamma
        print(-log_x_g_z,'\n',log_z,'\n',log_z_g_x,'\n',gamma)
        return x_reconstruction,eta_1,gamma,elbo
def train(model,optimizer,scheduler,train_loader,model_name='VAE_particle_flow',epochs=1000):
    losses=torch.zeros(epochs)
    for it in range(epochs):
        n_batch=0
        for batch_idx,x, in enumerate(train_loader):
            x=x.reshape((1,1,28,28))
            x_reconstruction,eta_1,gamma,elbo=model(x)
            loss=-elbo
            loss.backward()
            optimizer.zero_grad()
            print(("Epoch: {:>4}, Step: {:>4}, loss: {:>4.2f}")
                    .format(it, batch_idx, loss.item()), flush=True)
            optimizer.step()
            scheduler.step()
            losses[it] = losses[it]+loss.item()
            n_batch = n_batch+1.0
        losses[it] = losses[it] / n_batch
        if (epochs + 1) % 100 == 0:
            torch.save(model.state_dict(),
                        ("./output/model/{}_epoch_{}.model")
                        .format(model_name, epochs))
    return losses

