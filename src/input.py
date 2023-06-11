import json 
import os
import cv2 
import numpy as np
import glob
from tqdm import tqdm

hand_images = 0
total_images = 0
file_names = glob.glob("../archive/*.json")
for json_file in file_names:
    with open(json_file, "r") as f:
        annots = json.load(f)

    print(f"Processing {json_file}...")
    for annot in tqdm(annots["video_annotations"]):
        image_path = "../archive/" + annot["image"]["image_path"]
        image_name = image_path.split(os.path.sep)[-1].split(".")[0]
        classes = any(["hand" in x["name"] for x in annot["annotations"]])
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        mask = np.zeros((height, width, 3), dtype = np.uint8)
        if classes:
            for annotation in annot["annotations"]:
                if "hand" in annotation["name"]:
                    for segment in annotation["segments"]:
                        points = np.array(segment).reshape((1, -1, 2))
                        cv2.fillPoly(mask, np.int32([points]), (255, 255, 255))
            # In order to debug or visualize the mask and image, uncomment the following lines
            # new_image = np.hstack([image, mask])
            # cv2.imshow("mask", new_image)
            # cv2.waitKey(0)
        image_dir = f"../dataset/{image_name}"
        if classes:
            if not os.path.exists(image_dir):
                os.mkdir(image_dir)
            image = cv2.resize(image, (512, 512))
            mask = cv2.resize(mask, (512, 512))
            cv2.imwrite(f"{image_dir}/{image_name}.jpg", image)
            cv2.imwrite(f"{image_dir}/{image_name}_mask.jpg", mask)
            hand_images += 1
        total_images += 1

print(hand_images)
print(total_images)
