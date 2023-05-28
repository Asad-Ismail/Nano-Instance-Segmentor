
import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryLoss(torch.nn.Module):
    def __init__(self, alpha=1.0):
        super(BoundaryLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        boundary_pred = F.conv2d(pred, torch.ones(1, 1, 3, 3).to(pred.device), padding=1) < 9
        boundary_target = F.conv2d(target, torch.ones(1, 1, 3, 3).to(target.device), padding=1) < 9
        boundary_loss = F.binary_cross_entropy_with_logits(boundary_pred.float(), boundary_target.float(), reduction='mean')
        return self.alpha * boundary_loss

class MaskLoss(nn.Module):
    def __init__(self, mode='bce', eps=1e-7):
        super().__init__()
        self.mode = mode
        self.eps = eps

    def forward(self, pred, target):
        if self.mode == 'bce':
            return self.bce_loss(pred, target)
        elif self.mode == 'dice':
            return self.dice_loss(pred, target)
        else:
            raise ValueError("Unsupported loss type")

    def bce_loss(self, pred, target):
        bce = nn.BCEWithLogitsLoss()
        return bce(pred, target)

    def dice_loss(self, pred, target):
        pred = torch.sigmoid(pred)

        num = 2. * (pred * target).sum() + self.eps
        den = pred.sum() + target.sum() + self.eps

        return 1. - num / den


class CombinedMaskLoss(nn.Module):
    def __init__(self, mask_loss_weight=1.0, boundary_loss_weight=1.0):
        super().__init__()
        self.mask_loss = MaskLoss()
        self.boundary_loss = BoundaryLoss()
        self.mask_loss_weight = mask_loss_weight
        self.boundary_loss_weight = boundary_loss_weight

    def forward(self, pred, target):
        mask_loss_value = self.mask_loss(pred, target)
        boundary_loss_value = self.boundary_loss(pred, target)
        combined_loss = self.mask_loss_weight * mask_loss_value + self.boundary_loss_weight * boundary_loss_value
        return combined_loss
