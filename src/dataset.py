import torch 
import torch.nn as nn
import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2

class HandDataset:
    def __init__(self, image_paths, transforms = None):
        self.image_paths = image_paths
        self.transforms = transforms
        self.post_transforms = A.Compose([
                                A.Normalize(),
                                ToTensorV2(),
                            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path + "/" + image_path.split("/")[-1] + ".jpg")
        mask = cv2.imread(image_path + "/" + image_path.split("/")[-1] + "_mask.jpg", cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transforms(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']

        transformed_image = self.post_transforms(image = transformed_image)["image"]

        mask = (transformed_mask > 0).astype(np.uint8)
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return {"image": transformed_image, "mask": mask}