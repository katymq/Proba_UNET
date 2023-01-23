import os 
import pandas as pd
import pydicom as dicom
import numpy as np
import cv2
import torch


def SR_CT(path, list_img, normalization = True):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(path)
    sr_img = []
    if 'LapSRN_x8' in path:
        for img in list_img: 
            sr.setModel("lapsrn",8)
            if normalization:
                image =  cv2.normalize(img.astype(np.float32),  None, 0, 255, cv2.NORM_MINMAX)
            else:
                image  = img.astype(np.float32)
            final_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = sr.upsample(final_img)
            sr_img.append(image)
    return sr_img


def read_center_line(datos, error, dirr):
    folder_ct = 'data_ct'
    ct_im = []

    files = os.listdir(os.path.join(dirr, folder_ct)) 
    files.sort()
    print(np.max(datos['Slice']), np.min(datos['Slice']))
    files = files[np.min(datos['Slice']) -1:np.max(datos['Slice'])]
    for i, file in enumerate(files):
        im = dicom.dcmread(os.path.join(dirr,folder_ct, file)).pixel_array
        x , y = datos['px'][i].astype('int'), datos['py'][i].astype('int')
        x1,w1, y1 , h1 = x - error, x + error,  y - error, y + error
        ct_im.append(im[y1:h1, x1:w1].astype(np.float32))
        i += 1
    return ct_im



def test_SR_UNET(net, dataloaders_test, device, multi=False):
    seg_sr_images = []
    for _, batch in enumerate(dataloaders_test): 
        if multi:
            X, X_af, X_be = batch
            X, X_af, X_be = X.to(device), X_af.to(device), X_be.to(device)
            pred = net(X, X_af, X_be)
        else:
            X = batch
            X  = X.to(device)
            pred = net(X)

        seg = torch.argmax(pred, dim=1).numpy()[0]
        seg_sr_images.append(seg)# torch.argmax(pred, 1).numpy())
    return seg_sr_images


def test_SR(net, dataloaders_test, device,ProbaUnet = True):
    seg_sr_images = []
    for _, batch in enumerate(dataloaders_test): 
        X, y = batch

        X, y = X.to(device), y.to(device)
        
        if ProbaUnet:
            _,_, pred =  net(X, y)
        else:
            pred =  net(X)
        seg_sr_images.append(torch.argmax(pred, dim=1).numpy()[0])# torch.argmax(pred, 1).numpy())
    return seg_sr_images