from model import HandModel
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F

class Pipeline:
    def __init__(self, 
                ckpt_path: str = "../models/efficientnet_b2_fpn_v2/epoch=8-step=2574.ckpt",
                image_height = 512,
                image_width = 512):
        self.ckpt_path = ckpt_path
        self.model = HandModel.load_from_checkpoint(self.ckpt_path, map_location="cpu")
        self.transforms = A.Compose([
                                    A.Resize(width=image_width, height=image_height),
                                    A.Normalize(),
                                    ToTensorV2(),
                            ])
        self.model.eval()

    def __call__(self, image):
        """
        Forward function to run inference on the image
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image_rgb)["image"]
        with torch.no_grad():
            y_hat = self.model(image.unsqueeze(0)).sigmoid()
        y_hat = nn.functional.interpolate(
            y_hat,
            size=(image_rgb.shape[0], image_rgb.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        mask = (y_hat > 0.5).squeeze().cpu().numpy().astype("uint8")
        output = self.post_process(mask, image_rgb)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output

    def post_process(self, mask, image_rgb):
        """
        Post process to overlay the mask on the image 
        """
        color_seg = image_rgb.copy()
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] == 1:
                    color_seg[i, j, :] = (255, 0, 0)
        final_mask = cv2.addWeighted(image_rgb, 0.5, color_seg, 0.5, 0)
        return final_mask

pipe = Pipeline()
print("Done loading of model and pipeline...")
video_path = "../test_videos/One-Pot Chicken Fajita Pasta.mp4"
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(video_path)
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    
    frame = pipe(frame)
    # Display the resulting frame
    cv2.imshow('Frame',frame)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()