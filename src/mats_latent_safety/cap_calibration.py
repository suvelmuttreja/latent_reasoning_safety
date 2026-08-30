"""Task-specific generation-cap calibration using lengths only."""

from __future__ import annotations


def cap_projection(rows: list[dict], candidates: list[int]) -> dict[str, dict[str, dict]]:
    if not rows:
        raise ValueError("cap calibration requires generation rows")
    conditions = sorted({int(row["k"]) for row in rows})
    projection: dict[str, dict[str, dict]] = {}
    for candidate in candidates:
        by_k = {}
        for k in conditions:
            group = [row for row in rows if int(row["k"]) == k]
            projected = sum(int(row["generated_tokens"]) >= candidate for row in group)
            by_k[str(k)] = {
                "outputs": len(group),
                "projected_truncations": projected,
                "projected_truncation_rate": projected / len(group),
            }
        projection[str(candidate)] = by_k
    return projection


def select_smallest_cap(
    rows: list[dict], candidates: list[int], threshold: float
) -> tuple[int, dict[str, dict[str, dict]]]:
    if candidates != sorted(set(candidates)):
        raise ValueError("candidate caps must be unique and increasing")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between zero and one")
    projection = cap_projection(rows, candidates)
    for candidate in candidates:
        rates = [
            condition["projected_truncation_rate"]
            for condition in projection[str(candidate)].values()
        ]
        if all(rate < threshold for rate in rates):
            return candidate, projection
    raise ValueError("no registered candidate satisfies the truncation threshold")
