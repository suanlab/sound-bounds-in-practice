"""Loss functions for Safe-CEG training."""

from __future__ import annotations

import torch
from torch.nn import functional as fn


def expected_risk_loss(
    logits: torch.Tensor,
    config_q_errors: torch.Tensor,
    config_estimates: torch.Tensor,
    pessimistic_bounds: torch.Tensor,
    safety_lambda: float,
    tau: float,
    oracle_labels: torch.Tensor | None = None,
    ce_lambda: float = 0.5,
) -> torch.Tensor:
    """Optimize expected q-error, safety penalty, and oracle cross-entropy.

    Args:
        logits: Model output logits of shape ``[batch, num_configs]``.
        config_q_errors: Per-config q-errors of the same shape.
        config_estimates: Per-config estimates of the same shape.
        pessimistic_bounds: Pessimistic bounds of shape ``[batch]``.
        safety_lambda: Weight for the safety penalty term.
        tau: Safety threshold multiplier.
        oracle_labels: Optional oracle-best config indices of shape ``[batch]``.
            When provided, a cross-entropy term is added to encourage the model
            to directly predict the oracle-best configuration.
        ce_lambda: Weight for the cross-entropy term (default 0.5).
    """
    if safety_lambda < 0:
        raise ValueError("Safety lambda must be non-negative")
    if tau <= 0:
        raise ValueError("Tau must be positive")
    if logits.ndim != 2:
        raise ValueError("Logits must be a 2D tensor of shape [batch, num_configs]")
    if config_q_errors.shape != logits.shape:
        raise ValueError("Per-config q-errors must match the logits shape")
    if config_estimates.shape != logits.shape:
        raise ValueError("Per-config estimates must match the logits shape")
    if pessimistic_bounds.shape != (logits.shape[0],):
        raise ValueError("Pessimistic bounds must have shape [batch]")

    probabilities = torch.softmax(logits, dim=1)
    expected_log_q_error = torch.sum(
        probabilities * torch.log(config_q_errors), dim=1
    ).mean()
    threshold = tau * pessimistic_bounds.unsqueeze(1)
    expected_safety_penalty = torch.sum(
        probabilities * torch.relu(config_estimates - threshold), dim=1
    ).mean()

    total = expected_log_q_error + safety_lambda * expected_safety_penalty

    if oracle_labels is not None:
        ce_loss = fn.cross_entropy(logits, oracle_labels)
        total = total + ce_lambda * ce_loss

    return total
