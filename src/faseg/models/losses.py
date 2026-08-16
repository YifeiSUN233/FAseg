
"""Self-contained loss functions for TOF ultrasound segmentation.

These are the losses used by the primary training pipeline.  They have
**no** external dependencies beyond PyTorch.

Note
----
The third-party ``losses.py`` in ``legacy_reference/models/`` (DiceLoss,
FocalLoss, etc.) depends on the external ``utils.py`` from Hoel Kervadec's
MIT-licensed code and is **not** used by the main pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def pixelwise_cross_entropy_loss(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Standard pixel-wise cross-entropy loss.

    Parameters
    ----------
    logits : Tensor, shape ``(B, K, H, W)``
        Raw (pre-softmax) predictions.
    target : Tensor, shape ``(B, H, W)``
        Integer class labels in ``[0, K-1]``.

    Returns
    -------
    Tensor
        Scalar loss.
    """
    if target.dtype != torch.long:
        target = target.long()
    return nn.CrossEntropyLoss()(logits, target)


def pixelwise_cross_entropy_tv_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    lambda_tv: float = 1e-3,
    isotropic: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Cross-entropy loss with total-variation (TV) regularisation on the
    foreground probability channel.

    Parameters
    ----------
    logits : Tensor, shape ``(B, K, H, W)``
        Raw predictions.
    target : Tensor, shape ``(B, H, W)``
        Integer class labels.
    lambda_tv : float
        Weight of the TV term.
    isotropic : bool
        If ``True``, use isotropic TV ``sqrt(dx²+dy²)``; otherwise
        anisotropic ``|dx|+|dy|``.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    Tensor
        Scalar loss = CE + λ * TV.
    """
    # Cross-entropy
    loss_fn = nn.CrossEntropyLoss()
    if target.dtype != torch.long:
        target = target.long()
    loss_ce = loss_fn(logits, target)

    # Foreground probability (class index 1)
    prob = torch.softmax(logits, dim=1)          # (B, K, H, W)
    prob_fg = prob[:, 1, :, :]                    # (B, H, W)

    if isotropic:
        dx = prob_fg[:, 1:, :-1] - prob_fg[:, :-1, :-1]
        dy = prob_fg[:, :-1, 1:] - prob_fg[:, :-1, :-1]
        tv = torch.sqrt(dx * dx + dy * dy + eps).mean()
    else:
        dh = torch.abs(prob_fg[:, 1:, :] - prob_fg[:, :-1, :])
        dw = torch.abs(prob_fg[:, :, 1:] - prob_fg[:, :, :-1])
        tv = dh.mean() + dw.mean()

    return loss_ce + lambda_tv * tv
