
from ultralytics.engine.trainer import DEFAULT_CFG
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.data import build_yolo_dataset
from ultralytics.utils import LOGGER, RANK
import torch

from typing import Dict, Optional
from utils.datasets import build_adv_dataset
from models.adv_detect import ADVDetectModel


class ADVTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.add_callback("on_train_epoch_start", lambda this: this.model.on_new_epoch())
        self.add_callback("on_train_start", lambda this: this.model.pre_train(this.train_loader.dataset.get_adv_dateset(), this.batch_size, device = this.device))
        
    def get_model(self, cfg: Optional[str] = None, weights: Optional[str] = None, verbose: bool = True):
        model = ADVDetectModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
    
    def build_dataset(self, img_path: str, mode: str = "train", batch: Optional[int] = None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        if mode == 'train':
            return build_adv_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
        else:
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
    
    def preprocess_batch(self, batch: Dict) -> Dict:
        super().preprocess_batch(batch)
        temp = batch["img"]
        batch["img"] = batch["advimg"]
        batch["img"] = torch.stack(batch["img"], dim=0)
        super().preprocess_batch(batch)
        batch["advimg"] = batch["img"]
        batch["img"] = temp
        return batch