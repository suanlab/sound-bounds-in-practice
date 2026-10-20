"""Safety-aware estimator with upper-bound fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.ceg import pessimistic_bound, tighter_pessimistic_bound
from src.estimators.baseline_ceg import BaselineCEGEstimator
from src.utils.types import (
    CEGConfig,
    FallbackDirection,
    MLPrediction,
    SafeEstimationResult,
)

if TYPE_CHECKING:
    from src.core.query import Query
    from src.estimators.lower_certificate import LowerCertificate


@dataclass(frozen=True)
class SafeCEGEstimator:
    """Estimate query cardinalities with upper-bound safety fallback.

    Upper bound: if ``est_ml > tau_eff * pessimistic_bound``, fall back
    to the pessimistic bound (prevents catastrophic overestimation).

    When ``adaptive=True``, the effective tau is scaled by the selector's
    confidence: ``tau_eff = tau_min + (tau - tau_min) * confidence``.
    High-confidence queries use a looser threshold (aggressive), while
    low-confidence queries use a tighter threshold (conservative).
    """

    tau: float = 2.0
    adaptive: bool = False
    tau_min: float = 0.5
    use_tighter_bound: bool = False

    def __post_init__(self) -> None:
        if self.tau <= 0:
            raise ValueError("Tau must be positive")
        if self.tau_min <= 0:
            raise ValueError("Tau_min must be positive")
        if self.adaptive and self.tau_min > self.tau:
            raise ValueError("Tau_min must be <= tau when adaptive")

    def _effective_tau(self, confidence: float) -> float:
        """Compute per-query tau based on selector confidence."""
        if not self.adaptive:
            return self.tau
        c = max(0.0, min(1.0, confidence))
        return self.tau_min + (self.tau - self.tau_min) * c

    def estimate(
        self,
        query: Query,
        prediction: MLPrediction,
        lower_certificate: LowerCertificate | None = None,
    ) -> SafeEstimationResult:
        """Return a safe estimate using the predicted optimistic configuration.

        When ``lower_certificate`` is supplied, its unbiased point estimate is
        used as the (sound) lower floor and the bracket ``[L(Q), B(Q)]`` is
        recorded, giving a two-sided guarantee. Without it, the legacy
        min-across-configs floor is used (not a sound bound).
        """
        baseline_estimator = BaselineCEGEstimator()
        config = CEGConfig(
            path_length=prediction.path_length,
            aggregator=prediction.aggregator,
        )
        optimistic_estimate = baseline_estimator.estimate(query, config)
        if self.use_tighter_bound:
            pessimistic_estimate = tighter_pessimistic_bound(query)
        else:
            pessimistic_estimate = pessimistic_bound(query)

        # Lower floor: the sound certificate's unbiased estimate when available,
        # else the (unsound) min across all 9 optimistic configs.
        if lower_certificate is not None:
            optimistic_lower = lower_certificate.point_estimate
        else:
            all_estimates = baseline_estimator.estimate_all_configs(query)
            optimistic_lower = min(est.estimate for est in all_estimates.values())

        ml_est = optimistic_estimate.estimate
        tau_eff = self._effective_tau(prediction.confidence)
        upper_threshold = tau_eff * pessimistic_estimate
        lower_threshold = optimistic_lower / tau_eff

        # Safety check (upper-bound primary, lower-bound soft)
        if ml_est > upper_threshold:
            # Overestimation: fall back to pessimistic bound
            final_estimate = pessimistic_estimate
            used_fallback = True
            fallback_dir = FallbackDirection.UPPER
            config_used = None
        elif ml_est < lower_threshold:
            # Underestimation: fall back to optimistic lower bound
            final_estimate = optimistic_lower
            used_fallback = True
            fallback_dir = FallbackDirection.LOWER
            config_used = None
        else:
            final_estimate = ml_est
            used_fallback = False
            fallback_dir = FallbackDirection.NONE
            config_used = config

        return SafeEstimationResult(
            estimate=final_estimate,
            ml_estimate=ml_est,
            pessimistic_bound=pessimistic_estimate,
            optimistic_lower_bound=optimistic_lower,
            used_fallback=used_fallback,
            fallback_direction=fallback_dir,
            config_used=config_used,
            metadata={
                "tau": self.tau,
                "tau_effective": tau_eff,
                "upper_threshold": upper_threshold,
                "lower_threshold": lower_threshold,
                "confidence": prediction.confidence,
                "path_count": optimistic_estimate.path_count,
                "hop_count": optimistic_estimate.hop_count,
                "per_edge_alpha": dict(prediction.per_edge_alpha),
                "sound_lower_bound": (
                    None
                    if lower_certificate is None
                    else lower_certificate.sound_lower_bound
                ),
                "bracket": (
                    None
                    if lower_certificate is None
                    else (lower_certificate.sound_lower_bound, pessimistic_estimate)
                ),
            },
        )
