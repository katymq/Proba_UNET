# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 12:31:09 2022

@author: kmorales
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from unet_3block_conv import *
from loss import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
        
class priori(nn.Module):
    def __init__(self, input_channels, filters, z_dim, featureDim):
        super().__init__()
        self.input_channels = input_channels
        self.filters = filters
        self.z_dim = z_dim
        self.featureDim = featureDim
        #filters*2 *image_shape[0]//4  #4 = kernel _size
        self.p_z = nn.Sequential(nn.Conv2d(input_channels , filters, kernel_size=4, stride=2, padding=1),
                                 nn.BatchNorm2d(filters),
                                 nn.ReLU(), 
                                 nn.Conv2d(filters , filters*2, kernel_size=4, stride=2, padding=1),
                                 nn.BatchNorm2d(filters*2),
                                 nn.ReLU(),
                                 Flatten())
        self.mean_pz = nn.Linear(self.featureDim, self.z_dim)
        self.lstd_pz = nn.Linear(self.featureDim, self.z_dim)
        
    def forward(self, x):
        p_z = self.p_z(x)
        mean = self.mean_pz(p_z)
        log_std = self.lstd_pz(p_z)
        return mean, log_std


class variational(nn.Module):
    def __init__(self, input_channels, filters, z_dim, image_shape):
        super(variational, self).__init__()
        self.input_channels = input_channels
        self.filters = filters
        self.z_dim = z_dim
        self.featureDim = filters*4 *image_shape[0]//8 *image_shape[0]//8  #4 = kernel _size
        
        self.q_z = nn.Sequential(nn.Conv2d(input_channels+1 , filters, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(filters),
                    nn.ReLU(), 
                    nn.Conv2d(filters , filters*2, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(filters*2),
                    nn.ReLU(),
                    nn.Conv2d(filters*2 , filters*4, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(filters*4),
                    nn.ReLU(),
                    Flatten())
        
        self.mean_qz = nn.Linear(self.featureDim, self.z_dim)
        self.lstd_qz = nn.Linear(self.featureDim, self.z_dim)
        #self.z_qz =  nn.Linear(z_dim, self.featureDim)
        
    def forward(self, input, mask):
        q_z = self.q_z(torch.cat([input, mask.unsqueeze(dim = 1)], 1))
        mean = self.mean_qz(q_z)
        log_std = self.lstd_qz(q_z)
        return mean, log_std
        

class f_combined(nn.Module):
    def __init__(self, input_channels, num_classes, filters, z_dim, image_shape):
        super(f_combined, self).__init__()
        self.input_channels = input_channels
        self.filters = filters
        self.z_dim = z_dim
        self.num_classes = num_classes
        
        self.channel_axis = 1
        self.spatial_axes = [2,3]
        
        
        self.layers = nn.Sequential(nn.Conv2d(self.z_dim + self.input_channels , self.input_channels, kernel_size = 1),
                                 nn.ReLU(), 
                                 nn.Conv2d(self.input_channels , self.input_channels, kernel_size = 1),
                                 nn.ReLU())
        
        self.last_layer = nn.Conv2d(self.input_channels, self.num_classes, kernel_size=1)
        
        
        
    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)
    
    
    def forward(self, feature_map, z):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_sizexno_channelsxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        z = torch.unsqueeze(z,2)
        z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
        z = torch.unsqueeze(z,3)
        z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])

        #Concatenate the feature map (output of the UNet) and the sample taken from the latent space
        feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
        output = self.layers(feature_map)
        return self.last_layer(output)




class Probabilistic_UNET(nn.Module):
    def __init__(self, input_channels, num_classes, filters, z_dim, image_shape, featureDim):
        super(Probabilistic_UNET, self).__init__()
        self.input_channels = input_channels
        self.filters = filters
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.image_shape = image_shape
        self.featureDim = featureDim
        
        self.priori = priori(self.input_channels, self.filters, self.z_dim, self.featureDim)
        self.UNET =  UNET(self.input_channels, self.num_classes)
        self.variational  =  variational(self.input_channels, self.filters, self.z_dim, self.image_shape)
        self.F = f_combined(self.input_channels, self.num_classes, self.filters, self.z_dim, self.image_shape)
        
    def forward(self, image, mask):
        
        unet_features = self.UNET.forward(image)
        mu_priori, logstd_priori = self.priori.forward(image) 
        mu_vart, logstd_vart =  self.variational.forward(image, mask)
        reconstruction_loss,  kl = self.elbo(mask, mu_vart, logstd_vart, unet_features,mu_priori, logstd_priori)
        
        return reconstruction_loss,  kl
        
        
    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) +
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        return    0.5 * torch.sum(kld_element)
    
    def reparameterize(self, z_mean, z_log_var):
        """ z ~ N(z| z_mu, z_logvar) """
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5*z_log_var)*epsilon

    def elbo(self, mask, mu_vart, logstd_vart, unet_features,mu_priori, logstd_priori ):
        criterion = DiceLoss()
        z_sample =  self.reparameterize(mu_vart, logstd_vart)
        
        reconstruction = self.F.forward(unet_features, z_sample )
        
        reconstruction_loss = criterion(reconstruction, mask)
        
        kl = self._kld_gauss(mu_priori, torch.exp(logstd_priori), mu_vart, torch.exp(logstd_vart))
        
        return reconstruction_loss,  kl 
    
