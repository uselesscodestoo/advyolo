import torch.utils
import torch.utils.data
import torchvision
from ultralytics.data import YOLODataset
from ultralytics.utils import RANK, colorstr
import os
import torch
import random
import cv2
from PIL import Image
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
        imgs = [os.path.join(dir, x) for x in os.listdir(dir) if x.endswith(IMG_EXTENSIONS)]
        if len(imgs) == 0:
            raise ValueError(f"No image files found in {dir}")
        return imgs

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

class FolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path) 
                           if os.path.isfile(os.path.join(folder_path, f)) 
                           and f.lower().endswith(IMG_EXTENSIONS)]
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {folder_path}")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class MultiFolderDataLoader:
    def __init__(self, folders, batch_size, iterations, transform=None):
        self.folders = folders
        self.batch_size = batch_size
        self.iterations = iterations
        self.current_iter = 0
        
        # 为每个文件夹创建单独的数据集和索引列表
        self.datasets = []
        self.indices_list = []
        self.current_indices = []
        
        for folder in folders:
            dataset = FolderDataset(folder, transform=transform)
            self.datasets.append(dataset)
            self.indices_list.append(list(range(len(dataset))))
            self.current_indices.append(0)
    
    def __iter__(self):
        self.current_iter = 0
        return self
    
    def __next__(self):
        if self.current_iter >= self.iterations:
            raise StopIteration
        
        # 为每个文件夹收集batch_size张图片
        batch_tensors = []
        for i, dataset in enumerate(self.datasets):
            folder_batch = []
            indices = self.indices_list[i]
            
            for _ in range(self.batch_size):
                # 如果当前索引超出范围，则重置并打乱
                if self.current_indices[i] >= len(indices):
                    self.current_indices[i] = 0
                    random.shuffle(indices)
                
                # 获取图片索引并递增
                img_idx = indices[self.current_indices[i]]
                self.current_indices[i] += 1
                
                # 获取并添加图片到批次
                img_tensor = dataset[img_idx]
                folder_batch.append(img_tensor)
            
            # 将批次转换为张量并添加到结果列表
            batch_tensor = torch.stack(folder_batch)
            batch_tensors.append(batch_tensor)
        
        self.current_iter += 1
        return batch_tensors
    
    def __len__(self):
        return self.iterations