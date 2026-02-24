import torch
import torch.nn as nn

EPS = 1e-6

def dice_loss_per_image(prob, labels):
    # prob/labels: (B,1,H,W) or (B,H,W)
    if prob.dim() == 4:
        prob = prob.squeeze(1)
    if labels.dim() == 4:
        labels = labels.squeeze(1)

    total = 0.0
    for p, y in zip(prob, labels):
        p = p.contiguous().view(-1)
        y = y.contiguous().view(-1)
        inter = (p * y).sum()
        denom = p.sum() + y.sum()
        dice = (2 * inter + EPS) / (denom + EPS)
        total += dice.pow(-1)  # your original style
    return total / prob.size(0)

def cross_entropy_with_weight(prob, labels):
    # prob/labels: (B,1,H,W) or (B,H,W), prob in (0,1)
    if prob.dim() == 4:
        prob = prob.squeeze(1)
    if labels.dim() == 4:
        labels = labels.squeeze(1)

    prob = prob.contiguous().view(-1).clamp(EPS, 1.0 - EPS)
    labels = labels.contiguous().view(-1)

    pred_pos = prob[labels > 0]
    pred_neg = prob[labels == 0]
    w_anno = labels[labels > 0]  # usually all ones for binary edge

    # same as your code: mean over pos + mean over neg
    loss_pos = (-pred_pos.log() * w_anno).mean() if pred_pos.numel() > 0 else prob.new_tensor(0.0)
    loss_neg = (-(1.0 - pred_neg).log()).mean() if pred_neg.numel() > 0 else prob.new_tensor(0.0)
    return loss_pos + loss_neg

def cross_entropy_per_image(prob, labels):
    # prob/labels: (B,1,H,W)
    total = 0.0
    for p, y in zip(prob, labels):
        total += cross_entropy_with_weight(p.unsqueeze(0), y.unsqueeze(0))
    return total / prob.size(0)

class EdgeCrossEntropyDice(nn.Module):
    """
    Your legacy edge loss: weighted CE on probabilities + optional Dice + optional side outputs.
    NOTE: pred/side_output must be PROBABILITIES (after sigmoid).
    """
    def __init__(self, dice_weight=0.0, side_weight=1.0):
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.side_weight = float(side_weight)

    def forward(self, pred_prob, labels, side_output=None):
        # pred_prob: (B,1,H,W) prob
        # labels:    (B,1,H,W) or (B,H,W) in {0,1}
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)

        total_loss = cross_entropy_per_image(pred_prob, labels) + self.dice_weight * 0.1 * dice_loss_per_image(pred_prob, labels)

        if side_output is not None:
            # side_output: list of (B,1,H,W) prob
            for s in side_output:
                total_loss = total_loss + self.side_weight * cross_entropy_per_image(s, labels) / max(1, len(side_output))

        # return signature similar to your old code, but extra tensors are optional
        return total_loss
