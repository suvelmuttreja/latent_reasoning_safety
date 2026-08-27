import torch

from mats_latent_safety.state_audit import audit_state_dict


def test_clean_state_dict_audit():
    audit = audit_state_dict({"a": torch.zeros(2)}, {"a": torch.ones(2)})
    assert audit.clean


def test_material_missing_unexpected_and_shape_mismatch():
    audit = audit_state_dict(
        {"a": torch.zeros(2), "b": torch.zeros(3)},
        {"a": torch.zeros(4), "c": torch.zeros(1)},
    )
    assert not audit.clean
    assert audit.material_missing == ("b",)
    assert audit.material_unexpected == ("c",)
    assert audit.shape_mismatches[0]["key"] == "a"


def test_explicit_allowlist_is_persisted_but_not_material():
    audit = audit_state_dict(
        {"base.weight": torch.zeros(1), "new.embedding": torch.zeros(1)},
        {"base.weight": torch.zeros(1), "metadata.version": torch.zeros(1)},
        allowed_missing_prefixes=("new.",),
        allowed_unexpected_prefixes=("metadata.",),
    )
    assert audit.clean
    assert audit.allowed_missing == ("new.embedding",)
    assert audit.allowed_unexpected == ("metadata.version",)

