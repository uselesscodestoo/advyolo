
from ultralytics.engine.trainer import BaseTrainer, DEFAULT_CFG
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.torch_utils import torch_distributed_zero_first
from ultralytics.utils import LOGGER, RANK

from typing import Dict, Optional
from utils.datasets import build_dataloader_adv
from models.adv_detect import ADVDetectModel


class ADVTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.add_callback("on_train_epoch_start", lambda this: this.model.on_new_epoch())
        self.add_callback("on_train_start", lambda this: this.model.pre_train(this.train_loader.adv_dataset, this.batch_size, device = this.device))
        
    def get_model(self, cfg: Optional[str] = None, weights: Optional[str] = None, verbose: bool = True):
        model = ADVDetectModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
    
    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """
        Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader_adv(dataset, batch_size, workers, shuffle, rank, adv=self.data['adv'])  # return dataloader



    def preprocess_batch(self, batch: Dict) -> Dict:
        super().preprocess_batch(batch)
        temp = batch["img"]
        batch["img"] = batch["advimg"]
        super().preprocess_batch(batch)
        batch["advimg"] = batch["img"]
        batch["img"] = temp
        return batch