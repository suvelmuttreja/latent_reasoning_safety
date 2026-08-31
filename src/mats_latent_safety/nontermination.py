"""Content-blind structural metrics for generation nontermination diagnostics."""

from __future__ import annotations


def exact_suffix_cycle(
    tokens: list[int], max_period: int, minimum_repeats: int, minimum_covered: int
) -> dict | None:
    best = None
    for period in range(1, min(max_period, len(tokens) // minimum_repeats) + 1):
        block = tokens[-period:]
        repeats = 1
        cursor = len(tokens) - 2 * period
        while cursor >= 0 and tokens[cursor : cursor + period] == block:
            repeats += 1
            cursor -= period
        covered = repeats * period
        if repeats >= minimum_repeats and covered >= minimum_covered:
            candidate = {"period_tokens": period, "repeats": repeats, "covered_tokens": covered}
            if best is None or covered > best["covered_tokens"]:
                best = candidate
    return best


def unique_ngram_ratio(tokens: list[int], n: int = 4) -> float:
    if len(tokens) < n:
        return 1.0
    grams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)
