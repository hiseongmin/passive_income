"""
Loss Functions for Trigger Prediction Model

Combined loss for:
- Trigger probability (BCE)
- Imminence score (MSE)
- Direction classification (CE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class TriggerLoss(nn.Module):
    """
    Combined loss function for trigger prediction.

    Components:
    - Trigger Loss: Binary Cross Entropy
    - Imminence Loss: Mean Squared Error (only for TRIGGER=1 samples)
    - Direction Loss: Cross Entropy (only for TRIGGER=1 samples)
    """

    def __init__(
        self,
        trigger_weight: float = 1.0,
        imminence_weight: float = 0.5,
        direction_weight: float = 0.3,
        class_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0
    ):
        """
        Args:
            trigger_weight: Weight for trigger loss
            imminence_weight: Weight for imminence loss
            direction_weight: Weight for direction loss
            class_weights: Class weights for trigger BCE (for imbalance)
            use_focal_loss: Whether to use focal loss for trigger
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()

        self.trigger_weight = trigger_weight
        self.imminence_weight = imminence_weight
        self.direction_weight = direction_weight
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma

    def focal_loss(
        self,
        prob: torch.Tensor,
        target: torch.Tensor,
        gamma: float = 2.0
    ) -> torch.Tensor:
        """
        Compute focal loss for binary classification.

        Args:
            prob: Predicted probabilities (batch, 1)
            target: Ground truth (batch, 1)
            gamma: Focusing parameter

        Returns:
            Focal loss value
        """
        # BCE loss
        bce = F.binary_cross_entropy(prob, target, reduction='none')

        # Focal weight
        pt = prob * target + (1 - prob) * (1 - target)
        focal_weight = (1 - pt) ** gamma

        # Apply class weights if provided
        if self.class_weights is not None:
            weight = self.class_weights[1] * target + self.class_weights[0] * (1 - target)
            focal_weight = focal_weight * weight

        return (focal_weight * bce).mean()

    def forward(
        self,
        trigger_prob: torch.Tensor,
        imminence: torch.Tensor,
        direction_logits: torch.Tensor,
        trigger_target: torch.Tensor,
        imminence_target: torch.Tensor,
        direction_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            trigger_prob: Predicted trigger probability (batch, 1)
            imminence: Predicted imminence score (batch, 1)
            direction_logits: Direction logits (batch, 3)
            trigger_target: Ground truth trigger (batch, 1)
            imminence_target: Ground truth imminence (batch, 1)
            direction_target: Ground truth direction (batch,)

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        device = trigger_prob.device

        # ==================== Trigger Loss ====================
        if self.use_focal_loss:
            trigger_loss = self.focal_loss(
                trigger_prob,
                trigger_target.float(),
                self.focal_gamma
            )
        else:
            if self.class_weights is not None:
                weight = self.class_weights.to(device)
                pos_weight = weight[1] / weight[0]
                trigger_loss = F.binary_cross_entropy(
                    trigger_prob,
                    trigger_target.float(),
                    reduction='mean'
                )
            else:
                trigger_loss = F.binary_cross_entropy(
                    trigger_prob,
                    trigger_target.float(),
                    reduction='mean'
                )

        # ==================== Masked Losses ====================
        # Only compute imminence and direction losses for TRIGGER=1 samples
        mask = trigger_target.squeeze() == 1

        if mask.sum() > 0:
            # Imminence Loss (MSE)
            imminence_loss = F.mse_loss(
                imminence[mask],
                imminence_target[mask],
                reduction='mean'
            )

            # Direction Loss (Cross Entropy)
            direction_loss = F.cross_entropy(
                direction_logits[mask],
                direction_target[mask],
                reduction='mean'
            )
        else:
            imminence_loss = torch.tensor(0.0, device=device)
            direction_loss = torch.tensor(0.0, device=device)

        # ==================== Total Loss ====================
        total_loss = (
            self.trigger_weight * trigger_loss +
            self.imminence_weight * imminence_loss +
            self.direction_weight * direction_loss
        )

        loss_dict = {
            'total_loss': total_loss.item(),
            'trigger_loss': trigger_loss.item(),
            'imminence_loss': imminence_loss.item() if isinstance(imminence_loss, torch.Tensor) else imminence_loss,
            'direction_loss': direction_loss.item() if isinstance(direction_loss, torch.Tensor) else direction_loss
        }

        return total_loss, loss_dict


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss for handling class imbalance.
    """

    def __init__(self, pos_weight: float = 1.0):
        """
        Args:
            pos_weight: Weight for positive class (TRIGGER=1)
        """
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        prob: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted BCE loss.

        Args:
            prob: Predicted probabilities
            target: Ground truth

        Returns:
            Loss value
        """
        # Weight for each sample
        weight = torch.where(
            target == 1,
            torch.tensor(self.pos_weight, device=prob.device),
            torch.tensor(1.0, device=prob.device)
        )

        bce = F.binary_cross_entropy(prob, target.float(), reduction='none')
        return (weight * bce).mean()


def compute_class_weights_from_labels(
    labels: torch.Tensor,
    smooth: float = 0.1
) -> torch.Tensor:
    """
    Compute class weights from label distribution.

    Args:
        labels: Tensor of labels (0 or 1)
        smooth: Smoothing factor

    Returns:
        Class weights tensor [weight_0, weight_1]
    """
    n_samples = len(labels)
    n_positive = labels.sum().item()
    n_negative = n_samples - n_positive

    # Avoid division by zero
    n_positive = max(n_positive, 1)
    n_negative = max(n_negative, 1)

    # Compute weights (inverse frequency)
    weight_0 = n_samples / (2 * n_negative)
    weight_1 = n_samples / (2 * n_positive)

    # Apply smoothing
    weight_0 = (1 - smooth) * weight_0 + smooth
    weight_1 = (1 - smooth) * weight_1 + smooth

    return torch.tensor([weight_0, weight_1])


if __name__ == "__main__":
    # Test loss functions
    print("Testing TriggerLoss...")

    batch_size = 32

    # Create dummy data
    trigger_prob = torch.sigmoid(torch.randn(batch_size, 1))
    imminence = torch.sigmoid(torch.randn(batch_size, 1))
    direction_logits = torch.randn(batch_size, 3)

    trigger_target = torch.randint(0, 2, (batch_size, 1)).float()
    imminence_target = torch.rand(batch_size, 1)
    direction_target = torch.randint(0, 3, (batch_size,))

    # Test basic loss
    criterion = TriggerLoss()
    total_loss, loss_dict = criterion(
        trigger_prob, imminence, direction_logits,
        trigger_target, imminence_target, direction_target
    )

    print(f"Basic loss:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

    # Test with class weights
    class_weights = compute_class_weights_from_labels(trigger_target.squeeze())
    print(f"\nClass weights: {class_weights}")

    criterion_weighted = TriggerLoss(
        class_weights=class_weights,
        use_focal_loss=True
    )
    total_loss, loss_dict = criterion_weighted(
        trigger_prob, imminence, direction_logits,
        trigger_target, imminence_target, direction_target
    )

    print(f"\nWeighted focal loss:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
