"""
Created on Mon Apr 11 15:48:23 2022
@author: kmorales
"""
import numpy as np
import cv2


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_mask_SR(img):
    #[nod -> 1, shee->2]
    list_images = [cv2.bitwise_xor(1*(img[:,:,2]==255), 1*(img[:,:,0]==255)),  1*(img[:,:,2]==255)]    
    n, m = list_images[0].shape
    mask_img = np.zeros((n,m),np.int64) 
    for i, image in enumerate(list_images):
        mask_img[ image == 1 ]  = (i+1)
    return mask_img  

# def box_image(mask, img, error = 5, Area = 50):
#     masks_imgs = []
#     imgs = []
#     box_info = []
#     label = (mask!=0)*1
#     contours , _ = cv2.findContours(label.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     idxs =  list(np.where(np.array([cv2.contourArea(x) for x in contours])>Area))[0]
#     #print(idxs)
#     #j = int(np.where([cv2.contourArea(x) for x in contours] == np.max([cv2.contourArea(x) for x in contours]))[0][0])
#     pix  = np.sum(label)
#     if pix > 12 and len(idxs)>0:   
#         for idx in idxs:
#             x,y,w,h = cv2.boundingRect(contours[int(idx)])
#             #print(x,y,w,h)
#             masks_imgs.append(mask[y-error:y+h+error,x-error:x+w+error])
#             imgs.append(img[y-error:y+h+error,x-error:x+w+error])
#             box_info.append([x,y,w,h])
    
#     return masks_imgs, imgs,  box_info
def box_image_seg(mask, img, error, Area = 50):
    '''
    From an image we can get several images for training
    It depends on the minimun area to create a nwe image
    It means we have several boxes information which contain localisation information of calcifications  
    '''
    masks_imgs = []
    imgs = []
    box_info = []
    label = (mask!=0)*1
    contours , _ = cv2.findContours(label.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    idxs =  list(np.where(np.array([cv2.contourArea(x) for x in contours])>Area))[0]
    #print(idxs)
    #j = int(np.where([cv2.contourArea(x) for x in contours] == np.max([cv2.contourArea(x) for x in contours]))[0][0])
    pix  = np.sum(label)
    if pix > 12 and len(idxs)>0:   
        for idx in idxs:
            x,y,w,h = cv2.boundingRect(contours[int(idx)])
            #print(x,y,w,h)
            masks_imgs.append(mask[y-error:y+h+error,x-error:x+w+error])
            imgs.append(img[y-error:y+h+error,x-error:x+w+error])
            box_info.append([x,y,w,h])
    
    return masks_imgs, imgs,  box_info




def read_image(imagePath, mask = True ):
    if mask:
        image = cv2.imread(imagePath, 0)
        image = cv2.normalize(image.astype(np.int16),  None, 0, 255, cv2.NORM_MINMAX)
        image = np.where(image == 255, 1, 0)
        #(thresh, image) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    else: 
        image = cv2.imread(imagePath)    
        image = cv2.normalize(image.astype(np.int16),  None, 0, 255, cv2.NORM_MINMAX)
        image = cv2.cvtColor( np.float32(image[:,:,1]),cv2.COLOR_GRAY2RGB)/255
        
    return image

def create_mask(mask_Path):
    mask_img_bi = create_binary_mask(mask_Path)
    n, m  =  mask_img_bi.shape
    mask_img = np.zeros(shape=(n, m, 3))
    for i in range(3):
        mask_img[0,0, :] = 1
        mask_img[:,:, i] = (i+1)*np.where(mask_img_bi == i+1, 1, 0)
        #mask_img[ :,:, i ]  = (i+1)*(Mask[:,:,i] == 255)
        #mask_img[ image == 255 ]  = (i+1)
    return mask_img 

def create_binary_mask(mask_Path):
    list_images = create_img_mask(mask_Path)
    list_images.reverse()
    
    n, m = list_images[0].shape
    mask_img = np.zeros((n,m),np.int64) 
    for i, image in enumerate(list_images):
        mask_img[ image == 1 ]  = (i+1)
    return mask_img    
    

def create_img_mask(mask_Path):
    list_images = []
    for imagePath in mask_Path:
        list_images.append(read_image(imagePath))
    return list_images



def transform_to_hu(medical_image, image):
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = image * slope + intercept

    return hu_image

# def box_image_seg(i,seg_im, error = 5):
#     label =(seg_im[i][:,:,0]==255)*1
#     contours , _ = cv2.findContours(label.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     j = int(np.where([cv2.contourArea(x) for x in contours] == np.max([cv2.contourArea(x) for x in contours]))[0])
#     x1, x2, y1, y2 = [], [], [], []
#     for cntr in contours:
#         x,y,w,h = cv2.boundingRect(cntr)
#         x1.append(x)
#         x2.append(x+w)    
#         y1.append(y)
#         y2.append(y+h)
#     x, w = x1[j]-error, x2[j] + error
#     y, h  = y1[j]-error, y2[j] + error
#     return x,w, y , h

def box_image_calc(i,calc_im, error = 5):
    label1 =(calc_im[i][:,:,2]==255)*1
    label2 =(calc_im[i][:,:,0]==255)*1
    label = cv2.bitwise_xor(label1,label2)
    contours , _ = cv2.findContours(label.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)       
    return x - error ,w+x + error, y -error , h+y+error



def resolution(i, path, ct_im,calc_im, error= 4):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(path)
    if 'EDSR_x4' in path:
        sr.setModel("edsr",4)
    if 'LapSRN_x8' in path:
        sr.setModel("lapsrn",8)
        
    x1,w1, y1 , h1 =  box_image_calc(i,calc_im, error)
    img = ct_im[i][y1:h1, x1:w1]
    
    if 'EDSR_x4' in path:
        image =  cv2.normalize(img.astype(np.int16),  None, 0, 255, cv2.NORM_MINMAX)
        final_img = cv2.merge([image,image,image])
    if 'LapSRN_x8' in path:    
        image =  cv2.normalize(img.astype(np.float32),  None, 0, 255, cv2.NORM_MINMAX)
        final_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    result = sr.upsample(final_img)
    resized = cv2.resize(final_img,dsize=None,fx=4,fy=4)
    return result, resized


def resolution2(i, path, ct_im, error=0):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(path)
    if 'EDSR_x4' in path:
        sr.setModel("edsr",4)
    if 'LapSRN_x8' in path:
        sr.setModel("lapsrn",8)
        
    img = ct_im[i]
    
    if 'EDSR_x4' in path:
        image =  cv2.normalize(img.astype(np.int16),  None, 0, 255, cv2.NORM_MINMAX)
        final_img = cv2.merge([image,image,image])
        result = sr.upsample(final_img)
        resized = cv2.resize(final_img,dsize=None,fx=4,fy=4)
    if 'LapSRN_x8' in path:    
        image =  cv2.normalize(img.astype(np.float32),  None, 0, 255, cv2.NORM_MINMAX)
        final_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        result = sr.upsample(final_img)
        resized = cv2.resize(final_img,dsize=None,fx=8,fy=8)   
    
    return result, resized


def add_pad(image, new_height=512, new_width=512):
    height, width = image.shape

    final_image = np.zeros((new_height, new_width))

    pad_left = int((new_width - width) // 2)
    pad_top = int((new_height - height) // 2)
    
    # Replace the pixels with the image's pixels
    final_image[pad_top:pad_top + height, pad_left:pad_left + width] = image
    
    return final_image

def crop_image(image, display=False):
    # Create a mask with the background pixels
    mask = image == 0

    # Find the cal area
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)
    
    # Remove the background
    croped_image = image[top_left[0]:bottom_right[0],
                top_left[1]:bottom_right[1]]
    
    return croped_image