from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils import LOGGER
from ultralytics.nn.modules import CBAM
import torch
import torch.nn as nn
import torchvision
from utils.loss import ComputeAttentionLoss, ComputeDomainLoss, ComputeBatchConsistentLoss
from models.advcommon import DiscriminatorConv
import random
from tqdm import tqdm

from utils.datasets import MultiFolderDataLoader

DEFAULT_HYP = {
    'lr0': 0.01, 
    'lrf': 0.2, 
    'momentum': 0.937, 
    'weight_decay': 0.0005, 
    'warmup_epochs': 3.0, 
    'warmup_momentum': 0.8, 
    'warmup_bias_lr': 0.1, 
    'box': 0.05, 
    'cls': 0.5, 
    'cls_pw': 1.0, 
    'obj': 1.0, 
    'obj_pw': 1.0, 
    'iou_t': 0.2, 
    'anchor_t': 4.0, 
    'fl_gamma': 0.0, 
    'hsv_h': 0.015, 
    'hsv_s': 0.7, 
    'hsv_v': 0.4, 
    'degrees': 0.0, 
    'translate': 0.1, 
    'scale': 0.5, 
    'shear': 0.0, 
    'perspective': 0.0, 
    'flipud': 0.0, 
    'fliplr': 0.5, 
    'mosaic': 1.0, 
    'mixup': 0.0
}

class ADVDetectModel(DetectionModel):
    MAX_ADV_TRAIN_EPOCHS = 100
    PRE_TRAIN_EPOCHS = 100


    def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True, hyp=DEFAULT_HYP):
        self.skip_gan_train = 0
        self.is_adv_train = True
        self.epochs = 0
        self.adv_layers = []
        super(ADVDetectModel, self).__init__(cfg, ch, nc, verbose)
        self.hyp = hyp
        self.adv_layers = self.yaml['head'][-1][0]
        total_layer = len(self.yaml['head']) + len(self.yaml['backbone'])
        self.adv_layers = [layer if layer >= 0 else total_layer + layer for layer in self.adv_layers]
        self.nc = self.yaml["nc"]
        self.cache = [None for _ in self.adv_layers]
        self.domloss = ComputeDomainLoss(self)
        self.attloss = ComputeAttentionLoss(self)
        self.batchloss = ComputeBatchConsistentLoss()

        offset = len(self.yaml['backbone'])
        head_channel = lambda i: self.yaml['head'][i - offset][3][0]
        self.discriminators = nn.ModuleList([DiscriminatorConv(head_channel(i)//2) for i in self.adv_layers])
        self.cbams = nn.ModuleList([CBAM(head_channel(i)//2) for i in self.adv_layers])

    
    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            # if self.epochs < ADVDetectModel.MAX_ADV_TRAIN_EPOCHS and m.i == len(self.model) - 1:
            #     x = [xx.detach() for xx in x]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if m.i in self.adv_layers:
                self.cache[self.adv_layers.index(m.i)] = x
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _adv_prameters(self):
        params = []
        for i in range(len(self.adv_layers)):
            params += list(self.cbams[i].parameters())
            params += list(self.discriminators[i].parameters())
        return params
    
    def freeze_adv_prameters(self, freeze=True):
        for param in self._adv_prameters():
            param.requires_grad = not freeze

    
    def fake_predict(self, x):
        y = []
        mids = [None for _ in self.adv_layers]
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if m.i in self.adv_layers:
                mids[self.adv_layers.index(m.i)] = x
        return x, mids
    
    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"]) if preds is None else preds
        troditional_loss, item_loss = self.criterion(preds, batch)
        
        if not self.training or not self.model.training or not self.is_adv_train:
            return troditional_loss, item_loss

        fake_pred, fake_mids = self.fake_predict(batch["advimg"])
        
        # maximize cross domain loss and minimize consistent domain loss
        if self.skip_gan_train == 0:
            opt = torch.optim.Adam(self._adv_prameters(), lr=0.001)
            pre_dloss = float("inf")
            cache_detach = [t.detach() for t in self.cache]
            mid_detach = [t.detach() for t in fake_mids]
            self.freeze_adv_prameters(False)
            for _ in range(3):
                opt.zero_grad()
                dloss = torch.zeros(1, device=batch["img"].device)
                bloss = torch.zeros(1, device=batch["img"].device)
                for i in range(len(self.adv_layers)):
                    attn_s = self.cbams[i](cache_detach[i])
                    attn_t = self.cbams[i](mid_detach[i])
                    dis_s = self.discriminators[i](attn_s)
                    dis_t = self.discriminators[i](attn_t)
                    dloss += - self.domloss(dis_s, dis_t)[0]
                    bloss += self.batchloss(dis_t)
                
                if dloss.item() < -len(self.adv_layers):
                    self.skip_gan_train = 50
                    bloss.backward() # distroy the graph
                    break
                dloss.backward(retain_graph=True)
                bloss.backward(retain_graph=False)
                opt.step()
                if abs(1 - dloss.item() / pre_dloss) < 0.001:
                    break
                pre_dloss = dloss.item()
            self.freeze_adv_prameters(True)
        else:
            self.skip_gan_train -= 1


        domain_losses = torch.zeros(1, device=batch["img"].device)
        attn_losses = torch.zeros(1, device=batch["img"].device)
        for i in range(len(self.adv_layers)):
            # compute domain loss
            attn_s = self.cbams[i](self.cache[i])
            attn_t = self.cbams[i](fake_mids[i])
            dis_s = self.discriminators[i](attn_s)
            dis_t = self.discriminators[i](attn_t)

            domain_loss, domain_loss_items, domain_accuracy_items = self.domloss(dis_s, dis_t)
            domain_losses += domain_loss

            # compute attention loss
            attn_loss, attn_loss_items = self.attloss(attn_s, batch)
            attn_losses += attn_loss
        
        loss = troditional_loss.sum() * 4.0 + domain_losses + attn_losses
        return loss, item_loss

    def on_new_epoch(self):
        # self.skip_gan_train = 0
        self.skip_gan_train = 1000000 # probably same as inf
        self.epochs += 1
        if self.epochs > ADVDetectModel.MAX_ADV_TRAIN_EPOCHS:
            self.is_adv_train = False
            for param in self.parameters():
                param.requires_grad = False
            
            detect_layer = self.model[-1]
            for param in detect_layer.parameters():
                param.requires_grad = True
    
    def pre_train(self, target_data, source_data, batch_size, device="cuda", extra={}):
        print("Pre train the relation network")
        for param in self.parameters():
            param.requires_grad = False
        self.freeze_adv_prameters(False)

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        dataloader = MultiFolderDataLoader([target_data, source_data], batch_size, ADVDetectModel.PRE_TRAIN_EPOCHS, transform)
        opt = torch.optim.Adam(self._adv_prameters(), lr=0.001)
        bar = tqdm(dataloader)
        for target, source in bar:
            target = target.to(device)
            source = source.to(device)
            opt.zero_grad()
            _, mid_t = self.fake_predict(target)
            _, mid_s = self.fake_predict(source)
            bloss = torch.zeros(1, device=device)
            dloss = torch.zeros(1, device=device)
            for i in range(len(self.adv_layers)):
                attn_t = self.cbams[i](mid_t[i])
                attn_s = self.cbams[i](mid_s[i])
                dis_t = self.discriminators[i](attn_t)
                dis_s = self.discriminators[i](attn_s)
                bloss += self.batchloss(dis_t)
                dloss += - self.domloss(dis_s, dis_t)[0] * 1.5
            bloss.backward(retain_graph=True)
            dloss.backward(retain_graph=False)
            opt.step()
            bar.set_description(f"bloss: {bloss.item():.4f}, dloss: {dloss.item():.4f}")

        for param in self.parameters():
            param.requires_grad = True
        self.freeze_adv_prameters(True)



        