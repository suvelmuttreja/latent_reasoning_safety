# Code and experiment structure

The core package is [src/mats_latent_safety](../src/mats_latent_safety).
Scripts handle model loading, command-line options, job control, and writing
records. The library implements the shared numerical and serialization logic.

| Component | Responsibility |
| --- | --- |
| [coconut.py](../src/mats_latent_safety/coconut.py) | Final-layer recurrent latent positions, Qwen cache compatibility, and autoregressive generation. |
| [data.py](../src/mats_latent_safety/data.py), [serialization.py](../src/mats_latent_safety/serialization.py), [batching.py](../src/mats_latent_safety/batching.py) | GSM8K conversion, latent curriculum sequences, loss masks, and matched batch construction. |
| [training.py](../src/mats_latent_safety/training.py), [fallback.py](../src/mats_latent_safety/fallback.py) | Token-weighted gradient accumulation, training accounting, and stage/checkpoint helpers. |
| [official_eval.py](../src/mats_latent_safety/official_eval.py), [parsing.py](../src/mats_latent_safety/parsing.py) | Official safety evaluation rules, answer extraction, and bounded outcome handling. |
| [cap_calibration.py](../src/mats_latent_safety/cap_calibration.py), [nontermination.py](../src/mats_latent_safety/nontermination.py) | Response-cap decisions and structural diagnostics for unfinished outputs. |
| [trajectory.py](../src/mats_latent_safety/trajectory.py), [coherence_audit.py](../src/mats_latent_safety/coherence_audit.py) | Stage comparisons, missingness bounds, and human-review packet helpers. |
| [records.py](../src/mats_latent_safety/records.py), [hashing.py](../src/mats_latent_safety/hashing.py), [runtime.py](../src/mats_latent_safety/runtime.py), [state_audit.py](../src/mats_latent_safety/state_audit.py) | Generation schemas, canonical hashes, code/job provenance, and state-dictionary compatibility audits. |

The data flow is:

```text
pinned source data -> frozen manifests -> matched training branches
                                           |
                          checkpoints + update metrics + durability receipts
                                           |
                              generations + stop reasons + provenance
                                           |
                        StrongREJECT scores / capability / human audit
                                           |
                        paired comparisons + bounds + diagnostic figures
```

The two training branches share data order, seed, optimizer, effective batch,
and stage schedule. The Coconut wrapper replaces selected written reasoning
steps with recurrent latent states. Numerical behavior follows the frozen
protocol; a cleanup of packaging, formatting, or documentation does not create
a new experimental condition.

The library and script-helper tests cover cache/latent behavior, serialization,
loss normalization, resume boundaries, frozen manifest identities, stopping
rules, censored accuracy, and paired trajectory analysis. The repository
verifier also checks the saved evidence against hashes and recomputes the
numerical audit. These checks do not rerun the full GPU experiment.

For entry points and historical job mappings, use the
[script index](../scripts/README.md). For setup and executable examples, use
[reproduction.md](reproduction.md).
