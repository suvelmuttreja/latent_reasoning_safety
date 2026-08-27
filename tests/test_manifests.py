import json
from pathlib import Path

from mats_latent_safety.hashing import sha256_json, sha256_text


MANIFESTS = Path(__file__).parents[1] / "manifests"


def load(name):
    return json.loads((MANIFESTS / name).read_text())


def test_gsm8k_heldout_is_fixed_and_hashed():
    manifest = load("gsm8k_heldout_200.json")
    assert len(manifest["records"]) == 200
    assert manifest["records_sha256"] == sha256_json(manifest["records"])
    assert len({row["id"] for row in manifest["records"]}) == 200


def test_strongreject_audit_has_two_per_category():
    manifest = load("strongreject_audit_12.json")
    categories = {}
    for row in manifest["records"]:
        categories[row["category"]] = categories.get(row["category"], 0) + 1
    assert len(manifest["records"]) == 12
    assert set(categories.values()) == {2}
    assert manifest["records_sha256"] == sha256_json(manifest["records"])


def test_coherence_set_has_balanced_kinds_and_prompt_hashes():
    manifest = load("coherence_10.json")
    kinds = [row["kind"] for row in manifest["records"]]
    assert kinds.count("generic_benign") == 5
    assert kinds.count("benign_but_risky") == 5
    assert all(row["sha256"] == sha256_text(row["prompt"]) for row in manifest["records"])

