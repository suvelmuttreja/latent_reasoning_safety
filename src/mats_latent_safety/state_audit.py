"""Material state-dict audit used before any public checkpoint is trusted."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class StateDictAudit:
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_mismatches: tuple[dict, ...]
    allowed_missing: tuple[str, ...]
    allowed_unexpected: tuple[str, ...]
    material_missing: tuple[str, ...]
    material_unexpected: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not (self.material_missing or self.material_unexpected or self.shape_mismatches)

    def to_dict(self) -> dict:
        return {**asdict(self), "clean": self.clean}


def _matches_any(key: str, prefixes: Iterable[str]) -> bool:
    return any(key == prefix or key.startswith(prefix) for prefix in prefixes)


def audit_state_dict(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
    *,
    allowed_missing_prefixes: tuple[str, ...] = (),
    allowed_unexpected_prefixes: tuple[str, ...] = (),
) -> StateDictAudit:
    missing = tuple(sorted(set(expected) - set(actual)))
    unexpected = tuple(sorted(set(actual) - set(expected)))
    mismatches = []
    for key in sorted(set(expected) & set(actual)):
        expected_shape = tuple(getattr(expected[key], "shape", ()))
        actual_shape = tuple(getattr(actual[key], "shape", ()))
        if expected_shape != actual_shape:
            mismatches.append(
                {"key": key, "expected_shape": expected_shape, "actual_shape": actual_shape}
            )
    allowed_missing = tuple(key for key in missing if _matches_any(key, allowed_missing_prefixes))
    allowed_unexpected = tuple(
        key for key in unexpected if _matches_any(key, allowed_unexpected_prefixes)
    )
    return StateDictAudit(
        missing=missing,
        unexpected=unexpected,
        shape_mismatches=tuple(mismatches),
        allowed_missing=allowed_missing,
        allowed_unexpected=allowed_unexpected,
        material_missing=tuple(key for key in missing if key not in allowed_missing),
        material_unexpected=tuple(key for key in unexpected if key not in allowed_unexpected),
    )

