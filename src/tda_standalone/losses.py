"""
TDA Standalone Loss Functions.

Multi-task loss combining:
- Binary cross-entropy for trigger prediction (primary task)
- Cross-entropy for regime classification (auxiliary task)
- Entropy regularization for confident regime predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .config import TrainingConfig


class TDAStandaloneLoss(nn.Module):
    """
    Multi-task loss for TDA Standalone Model.

    Combines:
    1. Trigger BCE loss (primary task)
    2. Regime CE loss (auxiliary, self-supervised)
    3. Entropy regularization (encourage confident regimes)

    Loss = trigger_weight * BCE(trigger) +
           regime_weight * CE(regime) -
           entropy_reg_weight * mean_entropy(regime_prob)
    """

    def __init__(
        self,
        trigger_weight: float = 1.0,
        regime_weight: float = 0.3,
        entropy_reg_weight: float = 0.1,
        pos_weight: Optional[float] = None,
    ):
        """
        Initialize loss function.

        Args:
            trigger_weight: Weight for trigger prediction loss
            regime_weight: Weight for regime classification loss
            entropy_reg_weight: Weight for entropy regularization
            pos_weight: Optional positive class weight for imbalanced triggers
        """
        super().__init__()

        self.trigger_weight = trigger_weight
        self.regime_weight = regime_weight
        self.entropy_reg_weight = entropy_reg_weight

        # Trigger loss (BCE with optional pos_weight for class imbalance)
        if pos_weight is not None:
            self.trigger_criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight])
            )
        else:
            self.trigger_criterion = nn.BCEWithLogitsLoss()

        # Regime loss (Cross-entropy)
        self.regime_criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.

        Args:
            outputs: Model outputs with keys:
                - trigger_logits: (batch,) trigger prediction logits
                - regime_logits: (batch, num_regimes) regime classification logits
                - confidence: (batch,) confidence scores (not used in loss)
            targets: Target values with keys:
                - trigger: (batch,) binary trigger labels
                - regime: (batch,) regime labels (0 to num_regimes-1)

        Returns:
            Dict with 'total', 'trigger', 'regime', 'entropy_reg' losses
        """
        # Move pos_weight to same device if needed
        if hasattr(self.trigger_criterion, 'pos_weight') and self.trigger_criterion.pos_weight is not None:
            self.trigger_criterion.pos_weight = self.trigger_criterion.pos_weight.to(
                outputs['trigger_logits'].device
            )

        # Trigger loss
        trigger_loss = self.trigger_criterion(
            outputs['trigger_logits'],
            targets['trigger'],
        )

        # Regime loss
        regime_loss = self.regime_criterion(
            outputs['regime_logits'],
            targets['regime'],
        )

        # Entropy regularization (encourage confident regime predictions)
        regime_prob = F.softmax(outputs['regime_logits'], dim=-1)
        entropy = self._compute_entropy(regime_prob)
        entropy_reg = entropy.mean()

        # Total loss (note: we subtract entropy because lower entropy = more confident)
        total_loss = (
            self.trigger_weight * trigger_loss +
            self.regime_weight * regime_loss -
            self.entropy_reg_weight * entropy_reg
        )

        return {
            'total': total_loss,
            'trigger': trigger_loss,
            'regime': regime_loss,
            'entropy_reg': entropy_reg,
        }

    def _compute_entropy(self, prob: torch.Tensor) -> torch.Tensor:
        """Compute entropy of probability distribution."""
        eps = 1e-8
        entropy = -torch.sum(prob * torch.log(prob + eps), dim=-1)
        return entropy


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reduces loss for well-classified examples, focusing on hard negatives.
    Useful when trigger events are rare.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for positive class (0-1)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: (batch,) prediction logits
            targets: (batch,) binary targets

        Returns:
            Focal loss value
        """
        probs = torch.sigmoid(logits)

        # Compute focal weights
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        # Compute alpha weights
        alpha_weight = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=targets.device),
            torch.tensor(1 - self.alpha, device=targets.device),
        )

        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Apply focal and alpha weights
        focal_loss = alpha_weight * focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TDAFocalLoss(nn.Module):
    """
    Multi-task loss using Focal Loss for trigger prediction.

    Better for highly imbalanced trigger labels.
    """

    def __init__(
        self,
        trigger_weight: float = 1.0,
        regime_weight: float = 0.3,
        entropy_reg_weight: float = 0.1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()

        self.trigger_weight = trigger_weight
        self.regime_weight = regime_weight
        self.entropy_reg_weight = entropy_reg_weight

        self.trigger_criterion = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
        )
        self.regime_criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss with focal loss for trigger."""
        # Trigger loss (focal)
        trigger_loss = self.trigger_criterion(
            outputs['trigger_logits'],
            targets['trigger'],
        )

        # Regime loss
        regime_loss = self.regime_criterion(
            outputs['regime_logits'],
            targets['regime'],
        )

        # Entropy regularization
        regime_prob = F.softmax(outputs['regime_logits'], dim=-1)
        eps = 1e-8
        entropy = -torch.sum(regime_prob * torch.log(regime_prob + eps), dim=-1)
        entropy_reg = entropy.mean()

        # Total loss
        total_loss = (
            self.trigger_weight * trigger_loss +
            self.regime_weight * regime_loss -
            self.entropy_reg_weight * entropy_reg
        )

        return {
            'total': total_loss,
            'trigger': trigger_loss,
            'regime': regime_loss,
            'entropy_reg': entropy_reg,
        }


def create_loss_function(
    config: TrainingConfig,
    pos_weight: Optional[float] = None,
    use_focal: bool = False,
) -> nn.Module:
    """
    Create loss function from config.

    Args:
        config: Training configuration
        pos_weight: Optional positive class weight for imbalanced data
        use_focal: Whether to use focal loss instead of BCE

    Returns:
        Loss module
    """
    if use_focal:
        return TDAFocalLoss(
            trigger_weight=config.trigger_weight,
            regime_weight=config.regime_weight,
            entropy_reg_weight=config.entropy_reg_weight,
        )
    else:
        return TDAStandaloneLoss(
            trigger_weight=config.trigger_weight,
            regime_weight=config.regime_weight,
            entropy_reg_weight=config.entropy_reg_weight,
            pos_weight=pos_weight,
        )


def compute_class_weights(labels: torch.Tensor) -> float:
    """
    Compute positive class weight for imbalanced binary classification.

    Args:
        labels: (N,) binary labels

    Returns:
        pos_weight = num_negative / num_positive
    """
    num_pos = labels.sum().item()
    num_neg = len(labels) - num_pos

    if num_pos == 0:
        return 1.0

    return num_neg / num_pos
