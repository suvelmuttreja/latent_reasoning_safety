from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def test_matched_branch_configs_share_registered_training_fields():
    coconut = yaml.safe_load((ROOT / "configs/fallback_4b_skip0.yaml").read_text())
    cot = yaml.safe_load((ROOT / "configs/matched_4b_cot.yaml").read_text())
    fields = (
        "model_id",
        "model_revision",
        "seed",
        "dataset_examples",
        "c_thought",
        "max_latent_stage",
        "stages",
        "epochs_per_stage",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "max_sequence_length",
        "learning_rate",
        "weight_decay",
        "precision",
        "attention_implementation",
        "save_only_improve",
    )
    assert {field: coconut[field] for field in fields} == {
        field: cot[field] for field in fields
    }
    assert coconut["branch"] == "coconut_skip0"
    assert cot["branch"] == "explicit_cot"
    assert cot["submission_status"] == "locked_until_coco_u1_inline_gate_passes"

