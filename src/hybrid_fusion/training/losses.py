"""
Custom loss functions for Hybrid Fusion multi-task learning.

Implements:
- Focal Loss for binary classification (handles class imbalance)
- Masked MSE for regression (only on positive triggers)
- Combined multi-task loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with class imbalance.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Where:
    - p_t = p if y=1, else 1-p
    - α_t = α if y=1, else 1-α
    - γ is the focusing parameter

    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weight: float = 1.0,
        reduction: str = "mean",
        input_type: str = "prob",
    ):
        """
        Args:
            alpha: Weighting factor for positive class (default 0.25)
            gamma: Focusing parameter (default 2.0)
            pos_weight: Additional weight for positive samples (default 1.0)
            reduction: 'none', 'mean', or 'sum'
            input_type: 'prob' for probabilities (0-1), 'logit' for raw logits
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.input_type = input_type

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predicted probabilities or logits of shape (batch, 1) or (batch,)
            targets: Ground truth labels of shape (batch, 1) or (batch,)

        Returns:
            Focal loss value
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()

        if self.input_type == "prob":
            # Input is probability - use numerically stable formulation
            # Cast to float32 to exit autocast context
            p = inputs.float()
            targets_f = targets.float()

            # Clamp probabilities for numerical stability
            eps = 1e-6
            p_clamped = p.clamp(eps, 1.0 - eps)

            # Compute BCE manually in float32 (safe from autocast)
            # BCE = -y*log(p) - (1-y)*log(1-p)
            log_p = torch.log(p_clamped)
            log_1_minus_p = torch.log(1.0 - p_clamped)
            ce_loss = -(targets_f * log_p + (1.0 - targets_f) * log_1_minus_p)

            # Clamp loss to prevent extreme values
            ce_loss = ce_loss.clamp(0.0, 100.0)

            p_t = p * targets_f + (1.0 - p) * (1.0 - targets_f)
            p_t = p_t.clamp(eps, 1.0 - eps)  # Prevent pow underflow
        else:
            # Input is logits - original behavior
            # Clamp inputs to prevent numerical instability
            inputs = inputs.clamp(-20, 20)

            # Apply sigmoid to get probabilities
            p = torch.sigmoid(inputs)

            # Compute ce loss (without pos_weight to avoid NaN)
            ce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduction='none'
            )

            # p_t is the probability of the true class
            p_t = p * targets + (1 - p) * (1 - targets)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Compute focal loss
        focal_loss = alpha_t * focal_weight * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class MaskedMSELoss(nn.Module):
    """
    MSE loss computed only on masked (positive trigger) samples.

    For max_pct regression, we only want to compute loss on samples
    where trigger=True, since max_pct is only meaningful for triggers.
    """

    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked MSE loss.

        Args:
            predictions: Predicted values of shape (batch, 1) or (batch,)
            targets: Ground truth values of shape (batch, 1) or (batch,)
            mask: Boolean mask of shape (batch, 1) or (batch,), True for valid samples

        Returns:
            Masked MSE loss value
        """
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        mask = mask.view(-1).bool()

        # Count positive samples
        n_positive = mask.sum().item()

        if n_positive == 0:
            # No positive samples in batch, return zero loss
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # Select only masked samples
        masked_preds = predictions[mask]
        masked_targets = targets[mask]

        # Compute MSE
        mse = (masked_preds - masked_targets) ** 2

        # Apply reduction
        if self.reduction == "mean":
            return mse.mean()
        elif self.reduction == "sum":
            return mse.sum()
        else:
            return mse


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning.

    Total Loss = w1 * FocalLoss(trigger) + w2 * MaskedMSE(max_pct)
    """

    def __init__(
        self,
        trigger_weight: float = 1.0,
        max_pct_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        """
        Args:
            trigger_weight: Weight for trigger classification loss
            max_pct_weight: Weight for max_pct regression loss
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()

        self.trigger_weight = trigger_weight
        self.max_pct_weight = max_pct_weight

        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, input_type="prob")
        self.masked_mse = MaskedMSELoss()

    def forward(
        self,
        trigger_probs: torch.Tensor,
        trigger_targets: torch.Tensor,
        max_pct_preds: torch.Tensor,
        max_pct_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined multi-task loss.

        Args:
            trigger_probs: Predicted trigger probabilities (batch, 1)
            trigger_targets: Ground truth triggers (batch, 1)
            max_pct_preds: Predicted max percentages (batch, 1)
            max_pct_targets: Ground truth max percentages (batch, 1)

        Returns:
            Tuple of (total_loss, trigger_loss, max_pct_loss)
        """
        # Classification loss (focal)
        trigger_loss = self.focal_loss(trigger_probs, trigger_targets)

        # Regression loss (masked MSE, only on positive triggers)
        mask = trigger_targets > 0.5
        max_pct_loss = self.masked_mse(max_pct_preds, max_pct_targets, mask)

        # Combined loss
        total_loss = (
            self.trigger_weight * trigger_loss +
            self.max_pct_weight * max_pct_loss
        )

        return total_loss, trigger_loss, max_pct_loss


def create_loss_function(
    trigger_weight: float = 1.0,
    max_pct_weight: float = 0.5,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> MultiTaskLoss:
    """
    Create the multi-task loss function.

    Args:
        trigger_weight: Weight for trigger classification loss
        max_pct_weight: Weight for max_pct regression loss
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss

    Returns:
        MultiTaskLoss instance
    """
    return MultiTaskLoss(
        trigger_weight=trigger_weight,
        max_pct_weight=max_pct_weight,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
    )
