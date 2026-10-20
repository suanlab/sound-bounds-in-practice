"""Sample-based sound lower-bound certificate for the Safe-CEG interface.

The pessimistic (max-degree) certificate ``B(Q)`` upper-bounds the true
cardinality and guards *overestimation*. It says nothing about
*underestimation*, which is the dominant failure mode on skewed real data
(JOB): there the optimistic configurations stay far below the true count, so the
upper gate never fires. This module supplies the missing *lower* certificate
``L(Q)`` so Safe-CEG can guarantee a two-sided bracket.

**Construction.** Sample one relation (the largest) uniformly at rate ``r`` and
join the sample with the full other relations. Let ``S`` be the realized size of
that sub-sampled join.

**Soundness (``L(Q) = S`` is a true lower bound).** The sampled relation is a
subset of the original, so the sub-sampled join is a subset of the true join;
the size of a subset never exceeds the size of the whole. Hence ``S <= |J|`` for
the true join ``J`` *deterministically* (probability 1), for every sample and
every rate. Therefore ``L(Q) = S`` is a sound lower bound, and the bracket
``[L(Q), B(Q)]`` contains the true cardinality with probability 1.

**Unbiased point estimate.** Each true-join tuple survives sampling iff its
tuple from the sampled relation is kept, which happens with probability ``r``
(uniform sampling). So ``E[S] = r * |J|`` and ``L_hat = S / r`` is an unbiased
estimate of ``|J|``. ``L_hat`` is used as a soft floor (rescues underestimates);
``L(Q) = S`` is the hard, sound bound used for the guarantee.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.utils.join_counting import count_join

if TYPE_CHECKING:
    from src.core.query import JoinPredicate


@dataclass(frozen=True)
class LowerCertificate:
    """A sound lower bound plus an unbiased point estimate for one query.

    Attributes:
        sound_lower_bound: ``L(Q) = S`` -- ``true_cardinality >= L(Q)`` w.p. 1.
        point_estimate: ``L_hat = S / r`` -- unbiased estimate of the true size,
            used as a soft floor to rescue underestimates.
        sample_rate: The realized sampling rate ``r`` of the sampled relation.
    """

    sound_lower_bound: float
    point_estimate: float
    sample_rate: float


def sample_lower_certificate(
    rows_by_relation: dict[str, list[dict[str, str]]],
    relation_names: frozenset[str],
    joins: frozenset[JoinPredicate],
    rate: float,
    rng: random.Random,
) -> LowerCertificate:
    """Compute one sample-based lower certificate (see module docstring).

    Args:
        rows_by_relation: Rows for each relation/alias in the query.
        relation_names: The relation (or alias) names participating in the join.
        joins: The equi-join predicates.
        rate: Target sampling rate in ``(0, 1]`` for the largest relation.
        rng: Seeded RNG for reproducible sampling.

    Returns:
        The :class:`LowerCertificate` for this sample.

    Raises:
        ValueError: If ``rate`` is not in ``(0, 1]`` or a relation is empty.
    """
    if not 0.0 < rate <= 1.0:
        raise ValueError("Sample rate must be in (0, 1]")
    largest = max(relation_names, key=lambda r: len(rows_by_relation[r]))
    original = rows_by_relation[largest]
    if not original:
        raise ValueError(f"Cannot sample empty relation {largest!r}")
    k = max(1, round(len(original) * rate))
    sampled = dict(rows_by_relation)
    sampled[largest] = rng.sample(original, k)
    effective_rate = k / len(original)
    sound_lower_bound = count_join(sampled, relation_names, joins)
    return LowerCertificate(
        sound_lower_bound=sound_lower_bound,
        point_estimate=sound_lower_bound / effective_rate,
        sample_rate=effective_rate,
    )


def aggregate_lower_certificate(
    rows_by_relation: dict[str, list[dict[str, str]]],
    relation_names: frozenset[str],
    joins: frozenset[JoinPredicate],
    rate: float,
    seeds: list[int],
) -> LowerCertificate:
    """Average the lower certificate over ``seeds`` for a stabler estimate.

    The reported sound bound is the *maximum* per-seed bound: each per-seed
    sub-sample is a subset of the true join (so each bound is sound), and the
    maximum of sound lower bounds is the tightest still-sound lower bound. The
    point estimate is the mean of the per-seed unbiased estimates.
    """
    if not seeds:
        raise ValueError("At least one seed is required")
    certs = [
        sample_lower_certificate(
            rows_by_relation, relation_names, joins, rate, random.Random(seed)
        )
        for seed in seeds
    ]
    return LowerCertificate(
        sound_lower_bound=max(c.sound_lower_bound for c in certs),
        point_estimate=sum(c.point_estimate for c in certs) / len(certs),
        sample_rate=certs[0].sample_rate,
    )
