import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., from_logits=True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, pred, target):
        if self.from_logits:
            pred = torch.sigmoid(pred)

        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)

        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        mean_dice = dice_score.mean()
        return 1 - mean_dice, mean_dice