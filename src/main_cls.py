import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import timm
import cv2
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import warnings
from transformers import get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from sklearn import model_selection, metrics
import time
import math
import gc
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)

class CFG:
    dataset_path = "../classification_dataset"
    batch_size = 8
    lr = 3e-4
    n_fold = 5
    model_name = "resnet34"
    image_dir = "dataset/"
    num_workers = 0
    gradient_accumulation_steps = 1
    max_grad_norm = 10
    save_dir = "../models/"
    save_model_name = "resnet34_cls"
    max_grad_norm = 10
    epochs = 5
    device = "mps"
    warmup_ratio = 0.1
    print_freq = 5
    num_classes = 3

#Utiliy functions 
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, paths, targets, transform):
        self.paths = paths
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.paths[idx]), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = self.transform(image = image)["image"]
        target = self.targets[idx]
        
        return {"x" : torch.tensor(image, dtype = torch.float32), "y" : torch.tensor(target, dtype = torch.long)}

class Model(nn.Module):
    def __init__(self, pretrained = True):
        super().__init__()
        self.model = timm.create_model(CFG.model_name, pretrained=pretrained)
        self.classifier = nn.Linear(in_features=self.model.num_features, out_features=CFG.num_classes, bias=True)
    
    def forward(self, x):
        x = self.model.forward_features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1, 3)
        logits = self.classifier(x)
        return logits
    
    def get_optimizer_scheduler(self, num_train_steps, params):
        opt = torch.optim.AdamW(params, lr=CFG.lr)
        sch = get_cosine_schedule_with_warmup(
          opt,
          num_warmup_steps=int(num_train_steps * CFG.warmup_ratio),
          num_training_steps=num_train_steps,
          last_epoch=-1,
          )
        return opt, sch

def train(epoch, model, train_loader, valid_loader, optimizer, scheduler, device, scaler, loss_fn):
    model.train()
    losses = AverageMeter()
    start = end = time.time()
    for step, d in enumerate(train_loader):
        
        inputs = d["x"] # obtain the inputs
        targets = d["y"] # obtain the targets

        inputs = inputs.to(device) 
        targets = targets.to(device)
        
        with autocast():    
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item() * CFG.gradient_accumulation_steps , CFG.batch_size)
        scaler.scale(loss).backward()
        
        
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
        end = time.time()
        
        if ((step + 1) % CFG.print_freq == 0) or (step == (len(train_loader)-1)):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step + 1, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm= grad_norm,
                          lr=scheduler.get_lr()[0]))
        
    return losses.avg

def valid(epoch, model, valid_loader, device, loss_fn):
    model.eval()
    all_targets = []
    all_outputs = []
    losses = AverageMeter()
    with torch.no_grad():
        for step, d in enumerate(valid_loader):

            inputs = d["x"] # obtain the inputs
            targets = d["y"] # obtain the targets

            inputs = inputs.to(device) 
            targets = targets.to(device)   
            logits = model(inputs)
            loss = loss_fn(logits, targets)

            
            losses.update(loss.item(), CFG.batch_size)
            targets = targets.cpu().numpy().tolist()
            outputs = torch.argmax(logits.view(-1, CFG.num_classes), dim = -1).cpu().numpy().tolist()

            all_targets.extend(targets)
            all_outputs.extend(outputs)

            if ((step + 1) % CFG.print_freq == 0) or (step == (len(valid_loader)-1)):
                print('Epoch: [{0}][{1}/{2}] '
                      'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                      .format(epoch+1, step + 1, len(valid_loader), loss=losses))
    all_targets = np.array(all_targets)
    all_outputs = np.array(all_outputs)
    
    return all_targets, all_outputs, losses.avg

train_transforms = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Blur(),
            A.GaussNoise(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
    p=1,
)

test_transforms = A.Compose(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
    p=1,
)

def main():
    #kfold - 0 is for testing, 1 is for validation, and rest for training
    dataset_path = CFG.dataset_path
    class_mapping = {"None": 0 , "Adding Ingredients":1, "Stirring":2}
    image_paths = []
    labels = []
    for folder in os.listdir(dataset_path):
        for image in os.listdir(os.path.join(dataset_path, folder)):
            image_paths.append(os.path.join(dataset_path, folder, image))
            labels.append(class_mapping[folder])
    df = pd.DataFrame({"imagepath":image_paths, "label":labels})
    df_rest = df[df.label != 0]
    df_none = df[df.label == 0]

    # Sample 400 sample from none
    df_none = df_none.sample(n=800, random_state=42)
    df = pd.concat([df_rest, df_none])

    df = df.sample(frac=1).reset_index(drop=True)
    df["kfold"] = -1
    y = df.label.values
    kf = model_selection.StratifiedKFold(n_splits=CFG.n_fold)
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
    
    train_df = df[df.kfold != 0].reset_index(drop=True)
    valid_df = df[df.kfold == 0].reset_index(drop=True)
    
    train_paths = train_df["imagepath"].to_list()
    train_targets = train_df["label"].astype(int).to_numpy()
    loss_fn = nn.CrossEntropyLoss()

    valid_paths = valid_df["imagepath"].to_list()
    valid_targets = valid_df["label"].astype(int).to_numpy()
    
    train_ds = CustomDataset(train_paths, train_targets, train_transforms)
    valid_ds = CustomDataset(valid_paths, valid_targets, test_transforms)
    
    batch_size = CFG.batch_size
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers = CFG.num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size = batch_size, shuffle = False, num_workers = CFG.num_workers)
    
    model = Model()
    num_train_steps = int(len(train_ds) / batch_size / CFG.gradient_accumulation_steps * CFG.epochs)
    optimizer, scheduler = model.get_optimizer_scheduler(num_train_steps, model.parameters())
    model = model.to(CFG.device)
    best_acc_score = 0
    scaler = GradScaler()
    for epoch in range(CFG.epochs):
        print("\nTRAIN LOOP\n")
        train_loss = train(epoch, model, train_loader, valid_loader, optimizer, scheduler, CFG.device, scaler, loss_fn)
        print("\nVALID LOOP\n")
        t, o, valid_loss = valid(epoch, model, valid_loader, CFG.device, loss_fn)
        acc = metrics.accuracy_score(t, o)
        if acc >= best_acc_score:
            best_acc_score = acc
            if CFG.save_dir is not None:
                if not os.path.exists(CFG.save_dir):
                    os.mkdir(CFG.save_dir)
                save_path = os.path.join(CFG.save_dir, f"{CFG.save_model_name}.pth")
            else:
                save_path = f"{CFG.save_model_name}.pth"
            torch.save(model.state_dict(), save_path)
        print(f"The accuracy score is {acc}")
        print(f"The best accuracy score is {best_acc_score}")
    print(f'\nThe best score for model is {best_acc_score}')
    del model, optimizer, scheduler, train_loader, valid_loader, train_df, valid_df;
    gc.collect()

if __name__ == "__main__":
    main()
