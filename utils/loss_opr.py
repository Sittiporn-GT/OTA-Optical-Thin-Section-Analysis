import numpy as np
import scipy.ndimage as nd

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.logger import get_logger

logger = get_logger()

def compute_class_weights(class_counts: dict, num_classes: int) -> torch.Tensor:
    n_total = sum(class_counts.values())  # N_total
    weights = []
    for c in range(num_classes):
        n_c = class_counts.get(c, 1)
        ratio = n_total / (num_classes * n_c)
        w_c = max(1.0, np.log(1.0 + ratio))
        weights.append(w_c)
    return torch.tensor(weights, dtype=torch.float32)

class FocalLoss2d(nn.Module):
    def __init__(self, gamma=0, weight=None, reduction='mean', ignore_index=255):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        if weight:
            self.loss = nn.NLLLoss(weight=torch.from_numpy(np.array(weight)).float(),
                                 reduction=reduction, ignore_index=ignore_index)
        else:
            self.loss = nn.NLLLoss(reduction=reduction, ignore_index=ignore_index)

    def forward(self, input, target):
        return self.loss((1 - F.softmax(input, 1))**2 * F.log_softmax(input, 1), target)

class RCELoss(nn.Module):
    def __init__(self, ignore_index=255, reduction='mean', weight=None, class_num=37, beta=0.01):
        super(RCELoss, self).__init__()
        self.beta = beta
        self.class_num = class_num
        self.ignore_label = ignore_index
        self.reduction = reduction
        self.criterion = nn.NLLLoss(reduction=reduction, ignore_index=ignore_index, weight=weight)
        self.criterion2 = nn.NLLLoss(reduction='none', ignore_index=ignore_index, weight=weight)

    def forward(self, pred, target):
        b, c, h, w = pred.shape
        max_pred, max_id = torch.max(pred, dim=1)		# pred (b, h, w)
        target_flat = target.view(b, 1, h, w)
        mask = (target_flat.ne(self.ignore_label)).float()
        target_flat = (mask * target_flat.float()).long()
        # convert to onehot
        label_pred = torch.zeros(b, self.class_num, h, w).cuda().scatter_(1, target_flat, 1)
        # print(label_pred.shape, max_id.shape)

        prob = torch.exp(pred)
        prob = F.softmax(prob, dim=1)      # i add this

        weighted_pred = F.log_softmax(pred, dim=1)
        loss1 = self.criterion(weighted_pred, target)

        label_pred = torch.clamp(label_pred, min=1e-9, max=1.0-1e-9)

        label_pred = torch.log(label_pred)
        loss2 = self.criterion2(label_pred, max_id)
        loss2 = torch.mean(loss2*mask)
        # print(loss1, loss2)
        loss = loss1 + self.beta*loss2
        # print(loss1, loss2)
        # print(loss)
        return loss

class BalanceLoss(nn.Module):
    def __init__(self, ignore_index=255, reduction='mean', weight=None):
        super(BalanceLoss, self).__init__()
        self.ignore_label = ignore_index
        self.reduction = reduction
        self.criterion = nn.NLLLoss(reduction=reduction, ignore_index=ignore_index, weight=weight)

    def forward(self, pred, target):
        # prob = torch.exp(pred)
        # # prob = F.softmax(prob, dim=1)      # i add this
        # weighted_pred = pred * (1 - prob) ** 2
        # loss = self.criterion(weighted_pred, target)

        prob = torch.exp(pred)
        prob = F.softmax(prob, dim=1)      # i add this
        weighted_pred = F.log_softmax(pred, dim=1) * (1 - prob) ** 2
        loss = self.criterion(weighted_pred, target)
        return loss

class berHuLoss(nn.Module):
    def __init__(self, delta=0.2, ignore_index=0, reduction='mean'):
        super(berHuLoss,self).__init__()
        self.delta = delta
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, pred, target):
        valid_mask = (1 - target.eq(self.ignore_index)).float()
        valid_delta = torch.abs(pred - target) * valid_mask
        max_delta = torch.max(valid_delta)
        delta = self.delta * max_delta

        f_mask = (1 - torch.gt(target, delta)).float() * valid_mask
        s_mask = (1 - f_mask ) * valid_mask
        f_delta =  valid_delta * f_mask
        s_delta = ((valid_delta **2) + delta **2) / (2 * delta) * s_mask

        loss = torch.mean(f_delta + s_delta)
        return loss


class SigmoidFocalLoss(nn.Module):
    def __init__(self, ignore_label, gamma=2.0, alpha=0.25,
                 reduction='mean'):
        super(SigmoidFocalLoss, self).__init__()
        self.ignore_label = ignore_label
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        b, h, w = target.size()
        pred = pred.view(b, -1, 1)
        pred_sigmoid = pred.sigmoid()
        target = target.view(b, -1).float()
        mask = (target.ne(self.ignore_label)).float()
        target = mask * target
        onehot = target.view(b, -1, 1)

        max_val = (-pred_sigmoid).clamp(min=0)

        pos_part = (1 - pred_sigmoid) ** self.gamma * (
                pred_sigmoid - pred_sigmoid * onehot)
        neg_part = pred_sigmoid ** self.gamma * (max_val + (
                (-max_val).exp() + (-pred_sigmoid - max_val).exp()).log())

        loss = -(self.alpha * pos_part + (1 - self.alpha) * neg_part).sum(
            dim=-1) * mask
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


class ProbOhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label, reduction='mean', thresh=0.6, min_kept=256,
                 down_ratio=1, use_weight=False):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            logger.info('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(1 - valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)     # 概率小于阈值的挖出来
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask
                # logger.info('Valid Mask: {}'.format(valid_mask.sum()))

        target = target.masked_fill_(1 - valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)

# My implementation of dice loss, which is not used in the paper, but i want to try it. 
class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, target):
        """
        logits: [B, C, H, W]
        target: [B, H, W]
        """
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)

        target = target.clone()
        valid_mask = (target != self.ignore_index)

        target[~valid_mask] = 0

        target_onehot = F.one_hot(target, num_classes=num_classes)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()

        valid_mask = valid_mask.unsqueeze(1)
        probs = probs * valid_mask
        target_onehot = target_onehot * valid_mask

        dims = (0, 2, 3)
        intersection = (probs * target_onehot).sum(dims)
        cardinality = probs.sum(dims) + target_onehot.sum(dims)

        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        loss = 1 - dice.mean()
        return loss
        
class FocalDiceLoss(nn.Module):
    def __init__(self, ignore_index=255, gamma=2.0, dice_weight=0.3, smooth=1.0):
        super(FocalDiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.dice = DiceLoss(ignore_index=ignore_index, smooth=smooth)

    def focal_loss(self, logits, target):
        """
        logits: [B, C, H, W]
        target: [B, H, W]
        """
        ce_loss = F.cross_entropy(logits, target, ignore_index=self.ignore_index, reduction='none') # [B, H, W]
        valid_mask = (target != self.ignore_index).float()
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        focal = focal * valid_mask
        return focal.sum() / (valid_mask.sum() + 1e-8)
    
    def forward(self, logits, target):
        focal = self.focal_loss(logits, target)
        dice = self.dice(logits, target)
        loss = focal + self.dice_weight * dice
        return loss

class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, target):
        """
        logits: [B, C, H, W]
        target: [B, H, W]
        """
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)

        target = target.clone()
        valid_mask = (target != self.ignore_index)

        target[~valid_mask] = 0

        target_onehot = F.one_hot(target, num_classes=num_classes)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()

        valid_mask = valid_mask.unsqueeze(1).float()
        probs = probs * valid_mask
        target_onehot = target_onehot * valid_mask

        dims = (0, 2, 3)
        intersection = (probs * target_onehot).sum(dims)
        cardinality = probs.sum(dims) + target_onehot.sum(dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss = 1.0 - dice.mean()

        return loss

class CEDiceLoss(nn.Module):
    def __init__(self, ignore_index=255, ce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super(CEDiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index, smooth=smooth)

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        dice = self.dice(logits, target)
        return self.ce_weight * ce + self.dice_weight * dice

class ImbalanceAwareWeightedCELoss(nn.Module):
    "W_c = max(1, log(1 + N_total / (N_classes * N_c)))"
    def __init__(self, class_counts, num_classes: int, ignore_index: int = 255, reduction: str = 'mean'):
        super(ImbalanceAwareWeightedCELoss, self).__init__()
        if isinstance(class_counts, (list, np.ndarray)):
            class_counts = {i: v for i, v in enumerate(class_counts)}

        weights = compute_class_weights(class_counts, num_classes)
        self.register_buffer('weights', weights)

        self.criterion = nn.CrossEntropyLoss(
            weight=weights,
            reduction=reduction,
            ignore_index=ignore_index,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
      return self.criterion(pred, target)

# class CEDiceLoss_weight(nn.Module):          
#     def __init__(self, class_counts, num_classes=15, ignore_index=255,
#                  ce_weight=0.5, dice_weight=0.5, smooth=1.0):
#         super(CEDiceLoss_weight, self).__init__()
#         weights = compute_class_weights(class_counts, num_classes) 

#         self.ignore_index  = ignore_index
#         self.ce_weight      = ce_weight
#         self.dice_weight    = dice_weight
#         self.class_weights  = weights     

#         self.ce   = nn.CrossEntropyLoss(
#             weight=weights, ignore_index=ignore_index)
#         self.dice = DiceLoss(ignore_index=ignore_index, smooth=smooth)

#     def forward(self, logits, target):
#         ce   = self.ce(logits, target)
#         dice = self.dice(logits, target)
#         return self.ce_weight * ce + self.dice_weight * dice