import os 
from outils_prepro import create_binary_mask, box_image_seg, read_image

def create_box_images(image_Paths, mask_Paths, error, Area):
    image_list = []
    mask_list = []
    box_list = []
    for idx in range(len(mask_Paths)):
        mask =  create_binary_mask(mask_Paths[idx])
        image = read_image(image_Paths[idx], False )
        masks_imgs, imgs, boxes = box_image_seg(mask, image, error, Area)
        image_list += imgs
        mask_list += masks_imgs
        box_list += boxes
    
    return image_list, mask_list, box_list

def create_dir_paths(path, train = True):
    '''
    Function which allows to create the images with 3 classes (sheet, nodular, otherwhise)
    '''
    if train:
        path_train_masks = os.path.join(path, 'train', 'mask')
        path_train = os.path.join(path, 'train')
    else:
        path_train_masks = os.path.join(path, 'test', 'mask')
        path_train = os.path.join(path, 'test')


    names_ = [f[:-25] for f in os.listdir(path_train_masks) if f.endswith('png') and 'sheet' in f]
    print(len(names_))
    names_.sort()


    mask_nod = [ f  for f in os.listdir(path_train_masks) if f.endswith('png') and 'nodular' in f and  f[:-27] in names_ ]
    names  = [f[:-27] for f in mask_nod ]

    mask_nod = [ os.path.join(path_train_masks, f)  for f in os.listdir(path_train_masks) if f.endswith('png') and 'nodular' in f and  f[:-27] in names_ ]
    mask_sh = [os.path.join(path_train_masks, f) for f in os.listdir(path_train_masks) if f.endswith('png') and 'sheet' in f  and  f[:-25] in names ]

    image_Paths  = [os.path.join(path_train_masks, f) for f in os.listdir(path_train_masks) if (f.endswith('grayscale.png') or f.endswith('greyscale.png') )  and f[:-14] in names]
    print(len(mask_nod), len(mask_sh), len(image_Paths))

    mask_bck = [os.path.join(path_train_masks, f) for f in os.listdir(path_train_masks) if f.endswith('png') and 'ale._nonbck' in f  and f[:-22] in names]
    #mask_bck = [os.path.join(path_train_masks, f) for f in os.listdir(path_train_masks) if f.endswith('png') and 'nonbck' in f  and f[:-12] in names]
    #mask_bck = [os.path.join(path_train_masks, f) for f in os.listdir(path_train_masks) if f.endswith('png') and 'nonbck' in f]

    print(len(mask_nod), len(mask_sh), len(mask_bck), len(image_Paths))
    mask_bck.sort()

    mask_nod.sort()
    mask_sh.sort()
    image_Paths.sort()

    mask_Paths  = [[nod, sh]  for nod, sh in zip(mask_nod,mask_sh) ]
    return image_Paths, mask_Paths

