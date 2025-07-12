import torch
import torch.utils
import torch.utils.data
import torchvision
from torch.utils.data import distributed
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.data import YOLODataset
from ultralytics.data.utils import PIN_MEMORY
from ultralytics.utils import RANK, colorstr
import os
import random
import cv2
import numpy as np
from typing import Dict, Optional

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

class AdvDataset(torchvision.datasets.VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.images = os.listdir(root)
        self.images = [os.path.join(root, x) for x in self.images if x.endswith(IMG_EXTENSIONS)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
        return img
    
class ADVYOLODataset(YOLODataset):
    def __init__(self, *args, data: Optional[Dict] = None, task: str = "detect", **kwargs):
        self.advimgs = self.build_imgs(data['adv'])
        self.adv_transform = self.build_adv_transforms(kwargs['imgsz'])
        super().__init__(*args, data=data, task=task, **kwargs)

    def __getitem__(self, index):
        lable = super().__getitem__(index)
        img = cv2.imread(self.advimgs[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.adv_transform is not None:
            img = self.adv_transform(img)
        lable["advimg"] = img
        return lable
    
    def build_adv_transforms(self, imgsz):
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((imgsz, imgsz), antialias=True),
        ])
    
    def build_imgs(self, dir):
        return [os.path.join(dir, x) for x in os.listdir(dir) if x.endswith(IMG_EXTENSIONS)]

    def get_adv_dateset(self):
        return AdvDataset(self.data['adv'], transform=self.adv_transform)


def build_adv_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32):
    return ADVYOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )