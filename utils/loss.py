import torch
import torch.nn as nn

class ComputeDomainLoss:
    # Compute domain losses
    def __init__(self, model):
        # Define criteria
        self.BCE = nn.BCEWithLogitsLoss()

    # def __call__(self, sp, tp):  # source predictions, target predictions
    #     device = sp[0].device
    #     losses = [torch.zeros(1, device=device) for _ in range(len(sp))]
    #     accuracies = [torch.zeros(1, device=device) for _ in range(len(sp))]
    #     targets = self.build_target(sp, tp)  # targets
    #     # Losses and accuracies
    #     for i in range(len(sp)):
    #         losses[i] += self.BCE(torch.cat((sp[i], tp[i])), targets[i].to(device))
    #         accuracies[i] = self.compute_accuracies(torch.cat((sp[i], tp[i])), targets[i].to(device))
    #     return sum(losses)/3., torch.cat(losses).detach(), torch.cat(accuracies).detach()

    def __call__(self, sp, tp):  # source predictions, target predictions
        device = sp.device
        target = self.build_target(sp, tp)

        # Losses and accuracies
        losse = self.BCE(torch.cat((sp, tp)), target.to(device))
        accuracie = self.compute_accuracies(torch.cat((sp, tp)), target.to(device))

        return losse/3., losse.detach(), accuracie.detach()

    def build_targets(self, sp, tp):
        t = []
        for i in range(len(sp)):
            t.append(torch.cat((torch.zeros(sp[i].shape), torch.ones(tp[i].shape))))
        return t

    def build_target(self, sp, tp):
        return torch.cat((torch.zeros(sp.shape), torch.ones(tp.shape)))

    def compute_accuracies(self, scores, ground_truth):
        # Compute accuracies for compute_domain_loss()
        predictions = (scores > 0.) # if > 0 it predicted source
        num_correct = (predictions == ground_truth).sum()
        num_samples = torch.prod(torch.tensor(predictions.shape))
        accuracy = float(num_correct)/float(num_samples)*100
        return torch.tensor([accuracy]).to(scores.device)


def process_masks(batch, img_size=(640, 640), batch_size=None):
    """
    为批处理中的每个图像生成掩码，其中有标签的边界框区域为 1，其余区域为 0
    
    参数:
        batch (dict): 包含 'batch_idx', 'bboxes', 'cls' 三个键的字典
        img_size (tuple): 图像尺寸，格式为 (height, width)
    
    返回:
        torch.Tensor: 掩码张量，形状为 [batch_size, 1, height, width]
    """
    # 确保输入有效
    assert 'batch_idx' in batch and 'bboxes' in batch and 'cls' in batch, \
        "Batch must contain 'batch_idx', 'bboxes', and 'cls'"
    
    batch_idx = batch['batch_idx']
    bboxes = batch['bboxes']
    cls = batch['cls']

    # 获取批次大小
    if batch_size is None:
        batch_size = int(batch_idx.max().item()) + 1
    
    # 创建全零掩码
    masks = torch.zeros((batch_size, 1, img_size[0], img_size[1]), dtype=torch.float32)
    
    # 处理每个边界框
    for i in range(len(bboxes)):
        # 获取图像索引
        img_idx = int(batch_idx[i].item())
        
        # 获取边界框坐标 (xywh -> xyxy) 并转换为像素坐标
        x, y, w, h = bboxes[i]
        x1 = int((x - w/2) * img_size[1])
        y1 = int((y - h/2) * img_size[0])
        x2 = int((x + w/2) * img_size[1])
        y2 = int((y + h/2) * img_size[0])
        
        # 确保坐标在有效范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_size[1] - 1, x2)
        y2 = min(img_size[0] - 1, y2)
        
        # 将边界框区域设置为 1
        if x2 > x1 and y2 > y1:
            masks[img_idx, 0, y1:y2, x1:x2] = 1.0
    
    return masks


class ComputeAttentionLoss:
    # Compute attention losses
    def __init__(self, model):
        self.device = next(model.parameters()).device  # get model device

        # Define criteria
        Dice = DiceLoss()

        self.Dice = Dice
        self.anchor_t = 4
        self.iou_t = 0.20
        if hasattr(model, 'hyp'):
            self.anchor_t = model.hyp['anchor_t']
            self.iou_t = model.hyp['iou_t']
        
        det = model.model[-1]  # Detect() module
        
        self.nl = det.nl
        self.stride = det.stride


    # def __call__(self, attn_maps, sep_targets):  # objectness maps, targets
    #     lattn = [torch.zeros(1, device=self.device) for _ in range(len(attn_maps))]
    #     tattn = self.build_targets(attn_maps, sep_targets)  # targets

    #     for i, attn_map in enumerate(attn_maps):
    #         lattn[i] += self.Dice(attn_map, tattn[i])

    #     return sum(lattn)*0.005, torch.cat(lattn).detach()

    def __call__(self, attn_map, batch):  # objectness maps, targets
        tattn = self.build_target(attn_map, batch)  # targets
        # print(attn_map.shape)
        lattn = self.Dice(attn_map, tattn)

        return lattn*0.005, lattn.detach()
    
    def build_target(self, attn_map, batch):
        
        return process_masks(batch, attn_map.shape[-2:], attn_map.shape[0]).to(attn_map.device)


    def build_targets(self, attn_maps, targets):
        """
        Build attention targets from YOLOv8 predictions and ground truth
        """
        tattns = [torch.zeros((len(targets), *m.shape[1:]), device=self.device) 
                 for m in attn_maps]
        
        for i, t in enumerate(targets):
                
            img_targets = t[t[:, 0] == i][:, 1:]  # Get targets for this image
            if len(img_targets) == 0:  # No targets
                continue
                
            # Process predictions (xyxy, conf, cls)
            boxes = p[:, :4]  # Predicted boxes
            conf = p[:, 4]   # Confidence scores
            
            # Generate masks for each detection layer
            for j, (attn_map, stride) in enumerate(zip(attn_maps, self.stride)):
                h, w = attn_map.shape[1:]
                
                # Create binary mask from high-confidence predictions
                high_conf_idx = conf > 0.5  # Threshold for mask generation
                if not high_conf_idx.any():
                    continue
                    
                pred_boxes = boxes[high_conf_idx]
                
                # Scale boxes to feature map size
                pred_boxes = pred_boxes / stride
                
                # Create binary mask
                attn_mask = torch.zeros((h, w), device=self.device)
                for box in pred_boxes:
                    left = box[0].round().int().clamp(0, w-1)
                    top = box[1].round().int().clamp(0, h-1)
                    right = box[2].round().int().clamp(0, w-1)
                    bottom = box[3].round().int().clamp(0, h-1)
                    attn_mask[top:bottom+1, left:right+1] = 1
                
                tattns[j][i] = attn_mask
                
        return tattns


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice


class ComputeBatchConsistentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        if len(x) <= 1:
            return torch.tensor(0.0, device=x[0].device) if x else 0

        all_features = torch.stack([torch.mean(feats, dim=0) for feats in x])
        batch_centers = all_features.unsqueeze(1)
        pairwise_diffs = batch_centers - all_features.unsqueeze(0)  # [num_batches, num_batches, feature_dim]
        pairwise_distances = torch.sum(pairwise_diffs ** 2, dim=-1)  # [num_batches, num_batches]
        
        # 只取上三角部分（不包括对角线）的距离
        mask = torch.triu(torch.ones_like(pairwise_distances, dtype=torch.bool), diagonal=1)
        valid_distances = pairwise_distances[mask]
        
        # 计算平均损失
        return torch.mean(valid_distances) if valid_distances.numel() > 0 else torch.tensor(0.0, device=all_features.device)
