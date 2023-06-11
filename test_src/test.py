import mmseg
from mmseg.apis import inference_segmentor, init_segmentor
import os 
import warnings
import cv2
import timm 
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
warnings.filterwarnings("ignore")

class Model(nn.Module):
    def __init__(self, pretrained = False):
        super().__init__()
        self.model = timm.create_model("resnet34", pretrained=pretrained)
        self.classifier = nn.Linear(in_features=self.model.num_features, out_features=3, bias=True)
    
    def forward(self, x):
        x = self.model.forward_features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1, 3)
        logits = self.classifier(x)
        return logits

class Pipeline:
    def __init__(self, config_file, checkpoint_path, checkpoint_cls_path = "../models/resnet34_cls.pth", device = "cpu"):
        self.config_file = config_file
        self.checkpoint_path = checkpoint_path
        self.model = init_segmentor(self.config_file, self.checkpoint_path, device = "cpu")
        self.cls_model = Model()
        self.cls_model.load_state_dict(torch.load(checkpoint_cls_path, map_location = "cpu"))
        self.cls_model.eval()
        self.test_transform = A.Compose(
                [
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ],
            p=1,
        )
        self.inv_cls_mapping = {0: 'None', 1: 'Adding Ingredients', 2: 'Stirring'}
    
    def __call__(self, image):
        """
        Forward function to run inference on the image
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        activity = self.predict_cls(image)
        result = inference_segmentor(self.model, image)[0]
        if result.sum() == 0:
           activity = "None"
        output_image = self.post_process(result, image)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        cv2.putText(output_image, f'Activity : {activity}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
        return output_image
    
    def predict_cls(self, image):
       """
       Predict the class of the image
       """
       image = cv2.resize(image.copy(), (224, 224))
       image = self.test_transform(image = image)["image"]
       image = image.unsqueeze(0)
       with torch.no_grad():
          logits = self.cls_model(image)
          preds = torch.argmax(logits, dim = 1).item()

       return self.inv_cls_mapping[preds]
    
    def post_process(self, mask, image_rgb):
        """
        Post process to overlay the mask on the image 
        """
        color_seg = image_rgb.copy()
        color_seg[mask == 1] = (255, 0, 0)
        color_seg[mask == 2] = (255, 0, 0) 
        final_mask = cv2.addWeighted(image_rgb, 0.5, color_seg, 0.5, 0)
        return final_mask

# Change these paths accordingly in your system
config_file = "../work_dirs/seg_twohands_ccda/seg_twohands_ccda.py" # Corresponding config file
checkpoint_file = "../work_dirs/seg_twohands_ccda/best_mIoU_iter_56000.pth" # Corresponding checkpoint file for hand segmentation
checkpoint_cls_path = "../models/resnet34_cls.pth" # Corresponding checkpoint file for activity classification

pipe = Pipeline(config_file, checkpoint_file)

# enter the video path you want to achieve inference on
print("Done loading of model and pipeline...")  
video_path = "../test_videos/Kitchen Training 101 Proper Use of Gloves.mp4" # Path to the video file
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)
fps = cap.get(cv2.CAP_PROP_FPS)
   
# Below VideoWriter object will create
# a frame of above defined The output 
# is stored in 'filename.avi' file.
result = cv2.VideoWriter('../test_videos_output/Kitchen Training 101 Proper Use of Gloves.avi',  # Provide a path to the output video file
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         int(fps), size)
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
frames_done = 0
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    
    frame = pipe(frame)
    # Display the resulting frame
    result.write(frame)

    frames_done += 1
    if frames_done % 20 == 0:
       print("Frames done: ", frames_done)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
    
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
result.release()
 
# Closes all the frames
cv2.destroyAllWindows()