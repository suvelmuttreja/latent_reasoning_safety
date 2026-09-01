"""Task-specific generation-cap calibration using lengths only."""

from __future__ import annotations


def validate_partial_calibration_prefix(
    rows: list[dict],
    expected_grid: list[dict],
    *,
    model_id: str,
    model_revision: str,
    checkpoint_sha256: str,
    partial_run_config_sha256: str,
) -> None:
    """Fail closed unless a partial cache is an exact, unjudged grid prefix."""

    expected_prefix = [
        (int(cell["k"]), cell["prompt_id"]) for cell in expected_grid[: len(rows)]
    ]
    observed_prefix = [(int(row.get("k", -1)), row.get("prompt_id")) for row in rows]
    if observed_prefix != expected_prefix:
        raise ValueError("partial calibration cache is not a frozen K/prompt-grid prefix")
    for row, cell in zip(rows, expected_grid):
        if (
            row.get("prompt_sha256") != cell["prompt_sha256"]
            or row.get("model_id") != model_id
            or row.get("model_revision") != model_revision
            or row.get("checkpoint_sha256") != checkpoint_sha256
            or row.get("partial_run_config_sha256") != partial_run_config_sha256
            or row.get("evaluator_payload") is not None
            or row.get("evaluator_score") is not None
        ):
            raise ValueError("partial calibration cache provenance differs from this run")


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


def select_smallest_cap_or_none(
    rows: list[dict], candidates: list[int], threshold: float
) -> tuple[int | None, dict[str, dict[str, dict]]]:
    """Return a mechanical non-pass instead of losing completed calibration data."""

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
    return None, projection


def select_smallest_cap_by_k_or_none(
    rows: list[dict], candidates: list[int], threshold: float
) -> tuple[dict[str, int | None], dict[str, dict[str, dict]]]:
    """Select caps independently for natural conditions sharing one calibration run."""

    _, projection = select_smallest_cap_or_none(rows, candidates, threshold)
    conditions = sorted({int(row["k"]) for row in rows})
    selected: dict[str, int | None] = {}
    for k in conditions:
        selected[str(k)] = next(
            (
                candidate
                for candidate in candidates
                if projection[str(candidate)][str(k)]["projected_truncation_rate"]
                < threshold
            ),
            None,
        )
    return selected, projection
