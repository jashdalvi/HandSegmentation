## Project Details

### Test_src folder

Contains the code for inference. First make sure you have the weights and the config file available. Then change the paths in the file test_src/test.py. Download the classification weights from [here](https://drive.google.com/drive/folders/17gX2I5O8zP-LCdwC0j6WoomOb07x4zaL?usp=drive_link) and the segmentation weights from [here](https://drive.google.com/uc?id=1LNMQ6TGf1QaCjMgTExPzl7lFFs-yZyqX)

```
# Change these paths accordingly in your system
config_file = "../work_dirs/seg_twohands_ccda/seg_twohands_ccda.py" # Corresponding config file
checkpoint_file = "../work_dirs/seg_twohands_ccda/best_mIoU_iter_56000.pth" # Corresponding checkpoint file for hand segmentation
checkpoint_cls_path = "../models/resnet34_cls.pth" # Corresponding checkpoint file for activity classification
```

Then run the following command to get the results. Also change the output video path if you want.

```
python test_src/test.py
```

### Src folder structure
```
.
├── config
│   └── config.yaml
├── dataset.py
├── inference.py
├── input.py
├── main.py
├── main_cls.py
└── model.py
```

### File Details
1. config/config.yaml: Configuration file for the project
2. dataset.py: Contains the dataset class
3. inference.py: Contains a general inference class for segmentation
4. input.py: Contains the data preparation code.
5. main.py: Contains the main training code for training a segmentation model
6. main_cls.py: Contains the main training code for training a classification model for activity recognition
7. model.py: Contains the model class for the segmentation model (Pytorch Lightning)

