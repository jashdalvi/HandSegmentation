from model import HandModel
from dataset import HandDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import lightning.pytorch as pl
import os
import random
import albumentations as A
import hydra
from omegaconf import DictConfig, OmegaConf

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg : DictConfig) -> None:
    device = cfg.device
    model = HandModel(cfg.model_name, cfg.encoder_name)
    dataset_path = cfg.dataset_path
    image_paths = [f"{dataset_path}/{x}" for x in os.listdir(dataset_path)]
    random.shuffle(image_paths)

    train_image_paths = image_paths[:int(0.8 * len(image_paths))]
    val_image_paths = image_paths[int(0.8 * len(image_paths)):]
    print(f"Training on {len(train_image_paths)} images")
    print(f"Validating on {len(val_image_paths)} images")


    train_transform = A.Compose([
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2)
    ])

    val_transform = A.Compose([
        A.Resize(width=256, height=256),
    ])

    train_dataset = HandDataset(train_image_paths, transforms=train_transform)
    val_dataset = HandDataset(val_image_paths, transforms=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.valid_batch_size, shuffle=False, num_workers=0)

    trainer = pl.Trainer(accelerator=device, max_epochs=cfg.num_epochs, default_root_dir="../models/")
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()    