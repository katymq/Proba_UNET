from torch.utils.data import Dataset
import cv2
import torch 
class Segmentation_after_before(Dataset):
    def __init__(self, imageList, maskList, image_shape, transforms = None, error = 20):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.imageList = imageList
        self.maskList = maskList
        self.transforms = transforms
        self.error  =  error
        self.image_shape =  image_shape
        #self.training = training
    def __len__(self):
        return len(self.imageList)

    def _get_tensor_image_(self, image):
        image = cv2.resize(image, self.image_shape, interpolation=cv2.INTER_NEAREST)
        image = torch.tensor(image, dtype=torch.float32).permute(2,0,1)
        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image = transformed["image"]
        return image
        
    def __getitem__(self, idx):
        # grab the image path from the current index
        # t, t+1, t-1 
        image_t  = self._get_tensor_image_(self.imageList[idx][0])
        image_after  = self._get_tensor_image_(self.imageList[idx][1])
        image_before  = self._get_tensor_image_(self.imageList[idx][2])
        
        mask = self.maskList[idx]
        mask = cv2.resize(mask, self.image_shape, interpolation=cv2.INTER_NEAREST)
        mask = torch.tensor(mask, dtype=torch.long) 
        
        if self.transforms is not None:
            transformed = self.transforms(mask=mask)
            mask = transformed["mask"]
        
        return image_t, image_after, image_before, mask