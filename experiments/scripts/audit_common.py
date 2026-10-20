"""Shared pieces of the paper-number audits: JSON path walking and the check record."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _dig(payload: object, path: str) -> Any:
    """Walk a dotted path through nested JSON mappings.

    A path containing ``::`` is split on ``::`` instead, for keys that themselves
    contain dots (``lpboundxsample0.25``).
    """
    node = payload
    if not path:  # the whole document
        return node
    separator = "::" if "::" in path else "."
    for part in path.split(separator):
        if not isinstance(node, dict) or part not in node:
            msg = f"path {path!r} missing at {part!r}"
            raise KeyError(msg)
        node = node[part]
    return node


@dataclass(frozen=True)
class Check:
    """One headline number: where it comes from and how the paper writes it."""

    label: str
    source: Path
    path: str
    # render receives transform's output: a float by default, or whatever a
    # check's own transform returns (e.g. a whole interval).
    render: Callable[[Any], str]
    transform: Callable[[Any], Any] = lambda v: float(v)
