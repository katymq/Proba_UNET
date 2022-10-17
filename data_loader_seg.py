from torch.utils.data import Dataset
import cv2
import torch 
import numpy as np

class SegmentationDataset_DA(Dataset):
    def __init__(self, imagePaths, maskPaths, image_shape, transforms = None, error = 20):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms
        self.error  =  error
        self.image_shape =  image_shape
    def __len__(self):
        return len(self.imagePaths)
    
    def __getitem__(self, idx):
        # grab the image path from the current index    
        image  = self.imagePaths[idx]
        mask = self.maskPaths[idx]
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)
        # check to see if we are applying any transformations
        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long) 
        return image, mask


class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, image_shape, transforms = None, error = 20):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms
        self.error  =  error
        self.image_shape =  image_shape
        #self.training = training
    def __len__(self):
        return len(self.imagePaths)
    
    def __getitem__(self, idx):
        # grab the image path from the current index
        imagePath = self.imagePaths[idx]
#--------------------------------------        
#         mask = create_binary_mask(self.maskPaths, idx)
#         #mask = create_mask(self.maskPaths, idx)  
#         image = read_image(imagePath, False )
        
        
#         x,w, y , h = box_image_seg(mask)
#         mask = mask[y-error:h+error,x-error:w+error]
#         image = image[y-error:h+error,x-error:w+error]
#--------------------------------------        
        image  = self.imagePaths[idx]
        mask = self.maskPaths[idx]
        #if self.training:
        image = cv2.resize(image, self.image_shape, interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, self.image_shape, interpolation=cv2.INTER_NEAREST)
    
        #image = torch.tensor(image, dtype=torch.float32).unsqueeze(dim=-1).permute(2,0,1)
        image = torch.tensor(image, dtype=torch.float32).permute(2,0,1)
        mask = torch.tensor(mask, dtype=torch.long) 
        #mask = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1) 
        
        #mask = torch.tensor(mask, dtype=torch.uint8).permute(2, 0, 1) 
        #mask = mask[:, :, None].permute(2, 0, 1)        
        # check to see if we are applying any transformations
        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            # apply the transformations to both images
            # image = self.transforms[0](image)
            # mask = self.transforms[1](mask)
        # return a tuple of the image and its mask
        
        return image, mask


class Segmentation_CT_Dataset(Dataset):
    def __init__(self, imageList, image_shape, transforms = None):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.imageList = imageList
        self.transforms = transforms
        self.image_shape =  image_shape
    def __len__(self):
        return len(self.imageList)
    
    def __getitem__(self, idx):
        # grab the image path from the current index
        image  = self.imageList[idx]
        image = torch.tensor(image, dtype=torch.float32).permute(2,0,1)
        if self.transforms is not None:
            image = self.transforms[0](image)
        return image