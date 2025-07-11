import torch
import torch.utils
import torch.utils.data
import torchvision
from torch.utils.data import distributed
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.data.utils import PIN_MEMORY
from ultralytics.utils import RANK
import os
import random
import cv2
import numpy as np

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
    

class ADVDataLoader(InfiniteDataLoader):
    def __init__(self, adv_dataset, *args, **kwargs):
        self.adv_dataset = adv_dataset
        super().__init__(*args, **kwargs)
        if self.adv_dataset is None:
            raise ValueError("adv_dataset must be set")
        if isinstance(self.adv_dataset, str):
            self.adv_dataset = AdvDataset(self.adv_dataset)
            self.adv_dataset.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((640, 640), antialias=True),
                torchvision.transforms.Lambda(lambda x: x.unsqueeze(0)),
            ])
    
    def __iter__(self):
        for batch in super().__iter__():
            B = batch["img"].shape[0]
            batch |= { "advimg": torch.cat([random.choice(self.adv_dataset) for _ in range(B)], dim=0) }
            yield batch



def build_dataloader_adv(dataset, batch: int, workers: int, shuffle: bool = True, rank: int = -1, drop_last: bool = False, adv = None):
    """
    Create and return an InfiniteDataLoader or DataLoader for training or validation.

    Args:
        dataset (Dataset): Dataset to load data from.
        batch (int): Batch size for the dataloader.
        workers (int): Number of worker threads for loading data.
        shuffle (bool, optional): Whether to shuffle the dataset.
        rank (int, optional): Process rank in distributed training. -1 for single-GPU training.
        drop_last (bool, optional): Whether to drop the last incomplete batch.

    Returns:
        (InfiniteDataLoader): A dataloader that can be used for training or validation.

    Examples:
        Create a dataloader for training
        >>> dataset = YOLODataset(...)
        >>> dataloader = build_dataloader(dataset, batch=16, workers=4, shuffle=True)
    """
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min(os.cpu_count() // max(nd, 1), workers)  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return ADVDataLoader(
        adv,
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=drop_last,
    )