"""
Focal Loss 实现
用于处理类别不平衡问题，自动关注难分类样本
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance

    Reference: https://arxiv.org/abs/1708.02002

    Args:
        alpha: 类别权重，可以是标量或列表
               - 标量: 正类的权重，负类权重为 1-alpha
               - 列表: 每个类别的权重 [class0_weight, class1_weight, ...]
        gamma: 聚焦参数，gamma越大，对易分类样本的权重降低越多
               推荐值: 2.0
        ignore_index: 忽略的类别索引（如掩码区域）
        reduction: 'mean' 或 'sum'
    """
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) - 模型输出的 logits
            targets: (B, H, W) - 真值标签

        Returns:
            loss: 标量损失值
        """
        # 获取维度信息
        B, C, H, W = logits.shape

        # 计算 cross entropy loss (不使用 reduction)
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)

        # 计算 softmax 概率
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # 获取目标类别的预测概率（使用 gather 而不是 one_hot）
        # 这样可以避免维度不匹配的问题
        targets_for_gather = targets.clone()
        targets_for_gather[targets == self.ignore_index] = 0  # 临时替换 ignore_index
        targets_for_gather = targets_for_gather.unsqueeze(1)  # (B, 1, H, W)

        pt = torch.gather(probs, 1, targets_for_gather).squeeze(1)  # (B, H, W)

        # 计算 focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # 应用 focal weight
        focal_loss = focal_weight * ce_loss

        # 应用 alpha 权重
        if isinstance(self.alpha, (float, int)):
            # 标量 alpha: 正类权重为 alpha，负类权重为 1-alpha
            alpha_weight = torch.where(targets == 1,
                                      torch.tensor(self.alpha, device=targets.device),
                                      torch.tensor(1 - self.alpha, device=targets.device))
            focal_loss = alpha_weight * focal_loss
        elif isinstance(self.alpha, (list, tuple)):
            # 列表 alpha: 每个类别的权重
            alpha_tensor = torch.tensor(self.alpha, device=targets.device)
            alpha_weight = alpha_tensor[targets]
            focal_loss = alpha_weight * focal_loss

        # 忽略 ignore_index
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
    """
    组合损失: Focal Loss + Dice Loss

    Args:
        focal_weight: Focal Loss 的权重
        dice_weight: Dice Loss 的权重
        focal_alpha: Focal Loss 的 alpha 参数
        focal_gamma: Focal Loss 的 gamma 参数
        ignore_index: 忽略的类别索引
    """
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

        # 只计算 class=1 (fire) 的 Dice Loss
        pred_fire = probs[:, 1]  # (B, H, W)
        target_fire = (targets == 1).float()  # (B, H, W)

        # 忽略掩码区域
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
