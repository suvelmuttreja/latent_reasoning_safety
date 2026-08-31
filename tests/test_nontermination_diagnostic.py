from mats_latent_safety.nontermination import exact_suffix_cycle
from mats_latent_safety.nontermination import unique_ngram_ratio


def test_exact_suffix_cycle_detects_registered_long_repeat():
    tokens = list(range(20)) + [7, 8] * 300
    cycle = exact_suffix_cycle(tokens, max_period=256, minimum_repeats=4, minimum_covered=512)
    assert cycle is not None
    assert cycle["covered_tokens"] >= 512


def test_exact_suffix_cycle_rejects_short_or_nonrepeating_suffix():
    assert exact_suffix_cycle(list(range(600)), 256, 4, 512) is None
    assert exact_suffix_cycle([1, 2] * 10, 256, 4, 512) is None


def test_unique_ngram_ratio_distinguishes_repeat_from_variety():
    assert unique_ngram_ratio([1, 2, 3, 4] * 100) < 0.1
    assert unique_ngram_ratio(list(range(400))) == 1.0
