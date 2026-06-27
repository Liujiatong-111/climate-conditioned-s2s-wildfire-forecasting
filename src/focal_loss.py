"""Brief implementation note."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Brief implementation note."""
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        """Brief implementation note."""
        
        B, C, H, W = logits.shape

        
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)

        
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        
        
        targets_for_gather = targets.clone()
        targets_for_gather[targets == self.ignore_index] = 0  
        targets_for_gather = targets_for_gather.unsqueeze(1)  # (B, 1, H, W)

        pt = torch.gather(probs, 1, targets_for_gather).squeeze(1)  # (B, H, W)

        
        focal_weight = (1 - pt) ** self.gamma

        
        focal_loss = focal_weight * ce_loss

        
        if isinstance(self.alpha, (float, int)):
            
            alpha_weight = torch.where(targets == 1,
                                      torch.tensor(self.alpha, device=targets.device),
                                      torch.tensor(1 - self.alpha, device=targets.device))
            focal_loss = alpha_weight * focal_loss
        elif isinstance(self.alpha, (list, tuple)):
            
            alpha_tensor = torch.tensor(self.alpha, device=targets.device)
            alpha_weight = alpha_tensor[targets]
            focal_loss = alpha_weight * focal_loss

        
        mask = (targets != self.ignore_index).float()
        focal_loss = focal_loss * mask

        # Reduction
        if self.reduction == 'mean':
            return focal_loss.sum() / (mask.sum() + 1e-8)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """Brief implementation note."""
    def __init__(self, focal_weight=0.7, dice_weight=0.3,
                 focal_alpha=0.25, focal_gamma=2.0, ignore_index=2):
        super(CombinedLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, ignore_index=ignore_index)
        self.ignore_index = ignore_index

    def dice_loss(self, logits, targets):
        """
        Dice Loss for segmentation

        Args:
            logits: (B, C, H, W)
            targets: (B, H, W)
        """
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        
        pred_fire = probs[:, 1]  # (B, H, W)
        target_fire = (targets == 1).float()  # (B, H, W)

        
        mask = (targets != self.ignore_index).float()
        pred_fire = pred_fire * mask
        target_fire = target_fire * mask

        # Dice coefficient
        intersection = (pred_fire * target_fire).sum()
        union = pred_fire.sum() + target_fire.sum()

        dice = (2.0 * intersection + 1e-8) / (union + 1e-8)
        return 1 - dice

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W)
            targets: (B, H, W)
        """
        focal = self.focal_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        return self.focal_weight * focal + self.dice_weight * dice
