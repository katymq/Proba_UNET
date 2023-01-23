import cv2
import numpy as np
import os 
import pydicom as dicom
import sys 
# setting scripts/utils path
sys.path.append(r'C:\Users\kmorales\Desktop\2DO PhD\Strasbourg\Proba_UNET')
from create_images_3_classes import *


def str_after_before(idx):
    '''
    A function to combine the indexation on t+1 and t-1 wrt the actual image (t)
    '''
    if idx >=0 and idx <100:
        idx_str = str(0)+str(0)+str(idx)
    if idx >=100 and idx <1000:
        idx_str = str(0)+str(idx)
    if idx >=1000:
        idx_str = str(idx)
    return idx_str

def create_path_after_before(path_mct , names, names_files):
    '''
    YA NO.... Pilas aqui cuando vaya a utilizarlo con las demas carpetas porque hago un zip entre los directorios y los names que tengo disponibles
    Creation of path for images at time t-1 and t+1 wrt to the actual image t (image with its true mask)
    '''
    path_mct_files = []
    for path_, name in zip(path_mct, names_files):
        path_mct_files += [os.path.join(path_, f+'.dcm')   for f in names if name in f and f+'.dcm' in os.listdir(path_)]
    return path_mct_files
    
def read_image_dicom(imagePath):
    ''' 
    Lecture of DICOM images, it includes  normalization and transformation to gray scale (2 dimensions) 
    '''
    image = dicom.dcmread(imagePath).pixel_array
    image =  cv2.normalize(image.astype(np.float32),  None, 0, 255, cv2.NORM_MINMAX)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)/255
    return image
  

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


def create_box_images_after_before(image_Paths, mask_Paths, error, Area):
    image_list = []
    mask_list = []
    box_list = []

    for idx in range(len(mask_Paths)):
        mask =  create_binary_mask(mask_Paths[idx])
        image = read_image(image_Paths[idx], False )
        masks_imgs, imgs, boxes = box_image_seg(mask, image, error, Area)
        image_list += imgs
        mask_list+= masks_imgs
        box_list.append(boxes)
    
    return image_list, mask_list, box_list

def read_image_after_before(path , path_after, path_before, mask_Path, error, Area):
    '''
    Create a list of images with a box list (from one image) for  each image at time t, t+1 and t-1
    '''
    image_list = []
    image = read_image_dicom(path)
    image_after = read_image_dicom(path_after)
    image_before = read_image_dicom(path_before)
    mask =  create_binary_mask(mask_Path)
    masks_imgs, imgs, boxes = box_image_seg(mask, image, error, Area)
    for box in boxes:
        x,y,w,h =  box
        # print(x,y,w,h)
        # print(image.shape)
        img = [image[y-error:y+h+error,x-error:x+w+error], image_after[y-error:y+h+error,x-error:x+w+error] , image_before[y-error:y+h+error,x-error:x+w+error]] 
        image_list.append(img)
    return image_list, masks_imgs

def create_image_after_before(info_image, info_image_after_before, error, Area):
    '''
    info_image  dictionary 
    keys: file name
    values: mask, boxes, images (time t)

    info_image_after_before dictionary
    keys: file names
    values: path image at time t, path (t+1), path (t-1)
    '''
    image_list = []
    mask_list = []
    for key in list(info_image_after_before):
        #print(key)
        mask_Path  = info_image[key]
        path , path_after, path_before = info_image_after_before[key]
        #print(mask_Path[0].split('\\')[-1][:-4] ,'\n', path.split('\\')[-1][:-4] , '\n',path_after.split('\\')[-1][:-4], '\n',path_before.split('\\')[-1][:-4],'\n', key)
        imgs, masks_imgs = read_image_after_before(path , path_after, path_before, mask_Path, error, Area)
        image_list += imgs
        mask_list += masks_imgs

    return image_list, mask_list


def create_multi_input_data(error, Area, train_ = True):
    '''
    Create a multi inpu data set t, t+1, t-1
    '''
    path_ori = r'C:\Users\kmorales\Desktop\2DO PhD\Strasbourg\Hugo_seg\originals_180919\originals'
    path = r'C:\Users\kmorales\Desktop\2DO PhD\Strasbourg\Hugo_seg\ora_180919\Layers'
    path_MCTS = r'C:\Users\kmorales\Desktop\2DO PhD\Strasbourg\MicroCT_data'
    path_mcts = [os.path.join(path_MCTS, f) for f in os.listdir(path_MCTS)]
    names_files = [f for f in os.listdir(path_MCTS)]#print(path_mcts)

    # 1. Names of availables images (mask +images) at time t
    image_Paths, mask_Paths = create_dir_paths(path, train=train_)
    #########
    #image_list , mask_list, boxes =  create_box_images_after_before(image_Paths, mask_Paths, error, Area)
    #########
    info_image_dic = [[i[0].split('\\')[-1].split('.')[0].replace('-A', '').replace('_A', ''), i] for i in mask_Paths]
    info_image = {info_image_dic[i][0]: info_image_dic[i][1] for i in range(len(info_image_dic))}
    names = list(info_image.keys())
    names_after = [file[:-4]+str_after_before(int(file[-4:]) + 1) for file in names]
    names_before = [file[:-4]+str_after_before(int(file[-4:]) - 1) for file in names]

    # 2. Creation of paths t, t-1, t+1
    image_Paths_actual = create_path_after_before(path_mcts , names, names_files)
    image_Paths_after = create_path_after_before(path_mcts , names_after, names_files)
    image_Paths_before = create_path_after_before(path_mcts , names_before, names_files)
    #print(len(image_Paths_actual), len(image_Paths_after), len(image_Paths_before))

    # 3. Saving the information sing a dictionary 
    names_after_before = [path.split('\\')[-1][:-4] for path in image_Paths_actual]
    info_image_after_before = {names_after_before[i]: [image_Paths_actual[i], image_Paths_after[i], image_Paths_before[i] ]for i in range(len(names_after_before))}
    # for i , j, k , n  in zip(image_Paths_actual, image_Paths_after , image_Paths_before, names_after_before):
    #     print('\n',i.split('\\')[-1][:-4], '\n', j.split('\\')[-1][:-4], '\n', k.split('\\')[-1][:-4], '\n', n)

    # 4. Lists which contain microCT at time t, t+1, t-1 and the mask asociated 
    image_list, mask_list = create_image_after_before(info_image, info_image_after_before, error, Area)

    # print('number of images:', len(image_Paths), len(mask_Paths))
    # print('nombres disponibles: ', len(names))
    return image_list, mask_list 

    

