import torch
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
import os, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
#------------------------------------------------------------------
# TRAINING
#------------------------------------------------------------------
from unet_4block_conv import *
from unet_3block_conv import *
from outils_prepro import * 
from data_loader_seg import *
from model_prob_unet_init import *
from create_images_3_classes import *
from training import *
from loss import *
#---------------------
# Data
error = 20
Area = 100
image_shape = (128, 128) 
batch_size = 2
#----------------------
path_ori = r'C:\Users\kmorales\Desktop\2DO PhD\Strasbourg\Hugo_seg\originals_180919\originals'
path = r'C:\Users\kmorales\Desktop\2DO PhD\Strasbourg\Hugo_seg\ora_180919\Layers'
image_Paths, mask_Paths = create_dir_paths(path)
image_list , mask_list =  create_box_images(image_Paths, mask_Paths, error, Area)
print('Images and masks are created')
print(len(image_list), len(mask_list))     
data = SegmentationDataset(image_list, mask_list, image_shape)
dataloaders = torch.utils.data.DataLoader(data, batch_size)

#----------------------
# Model 
input_channels = 3
num_classes = 3
learning_rate = 5e-4  
epochs = 101
loss_fn  =  DiceLoss() 
folder_name = 'model_unet_2'

beta = 10
filters = 8
z_dim = 10
 # different learning rate and the same error as Porba Unet
#----------------------
if filters == 8:
    featureDim = 16384
if filters ==4:
    featureDim = 8192

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelPath = os.path.join(r'C:\Users\kmorales\Desktop\2DO PhD\Strasbourg\Hugo_seg', folder_name)
if not os.path.exists(modelPath):
    os.makedirs(modelPath)

#net = Probabilistic_UNET(input_channels, num_classes, filters, z_dim, image_shape, featureDim)
net = UNET(input_channels, num_classes) 
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0)
#summary(net)
#sum(p.numel() for p in net.parameters() if p.requires_grad) 
tloss, tloss_list = training_Unet(dataloaders, epochs, device, loss_fn, net, optimizer, modelPath)
plt.plot(tloss_list)




#------------------------------------------------------------------
# TEST
#------------------------------------------------------------------
import torch
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
import os, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
#---------------------
from test import test 
from unet_4block_conv import *
from unet_3block_conv import *
from outils_prepro import * 
from data_loader_seg import *
from model_prob_unet_init import *
from create_images_3_classes import *
from training import *
from loss import *
#---------------------
# Data
batch_size_test  = 1
error = 20
Area = 100
image_shape = (128, 128) 
#----------------------
path = r'C:\Users\kmorales\Desktop\2DO PhD\Strasbourg\Hugo_seg\ora_180919\Layers'

image_Paths, mask_Paths = create_dir_paths(path, False)
image_list , mask_list =  create_box_images(image_Paths, mask_Paths, error, Area)
print('Images and masks are created')
print(len(image_list), len(mask_list))     
data_test = SegmentationDataset(image_list, mask_list, image_shape)
dataloaders_test = torch.utils.data.DataLoader(data_test, batch_size_test)
#----------------------
# Model 
input_channels = 3
num_classes = 3
# learning_rate = 5e-4  
# epochs = 101
loss_fn  =  DiceLoss() 
folder_name = 'model_unet_2'
#----------------------
# TEST
epoch_init = '60.torch'
#----------------------

filters = 8
beta = 101
z_dim = 10
#----------------------
if filters == 8:
    featureDim = 16384
if filters ==4:
    featureDim = 8192


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelPath = os.path.join(r'C:\Users\kmorales\Desktop\2DO PhD\Strasbourg\Hugo_seg', folder_name)

#net = Probabilistic_UNET(input_channels, num_classes, filters, z_dim, image_shape, featureDim)
net = UNET(input_channels, num_classes) 
net.to(device)
net.load_state_dict(torch.load(os.path.join(modelPath , epoch_init))) # Load trained model
net.eval()
#optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0)

loss_fn  =  DiceLoss()  
dice_score_classes, loss = test(net, loss_fn, dataloaders_test, device, ProbaUnet = False, plot_ = True)
print(dice_score_classes, loss)
