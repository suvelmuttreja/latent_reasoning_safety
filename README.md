# Safety drift under explicit and latent reasoning fine-tuning

Independent research project, September 2026.

Starting from `Qwen/Qwen3-4B-Thinking-2507`, I fine-tuned two branches on the
same 7,473 GSM8K training examples with matched data order, seed, optimizer,
batch size, and three stages of two epochs each. One branch was ordinary SFT on
the written rationale and answer. The other was the skip-stage-0
[Coconut](https://arxiv.org/abs/2412.06769) curriculum, which at each stage
replaces one more written reasoning step with two continuous latent positions.
The preregistered question was whether benign reasoning fine-tuning erodes
refusals ([self-jailbreaking](https://arxiv.org/abs/2510.20956)) differently
when the reasoning is latent rather than written.

The headline outcome is a negative result with diagnostics. My Coconut branch
lost most of its math ability (31% vs 92% GSM8K exact match for explicit CoT),
so the capability-matched safety comparison I designed never existed. The
endpoint StrongREJECT difference was inconclusive: +0.012 for Coconut over CoT
with a 95% prompt-bootstrap interval of [-0.023, +0.046].

## Explore the project

- **Research:** [full report](writeup/FULL_WRITEUP.built.md),
  [formatted Word report](writeup/FULL_WRITEUP.docx),
  [claim-to-artifact map](writeup/claims_and_numbers.md), and
  [independent analysis review](writeup/analysis_audit/REVIEW.md).
- **Implementation:** [code architecture](docs/code.md),
  [scripts and historical job index](scripts/README.md), and
  [configuration guide](configs/README.md).
- **Evidence:** [artifact guide](artifacts/README.md),
  [random raw examples](writeup/random_examples.md), and
  [chronological research log](research_log.md).
- **Method:** [frozen execution protocol](project_plan/MATS_execution_protocol_v6_3.md)
  and [upstream revisions](configs/pins.json).

## Verify the saved results

From the repository root, with Python 3.11 and Bash:

```bash
python3 scripts/verify_repository.py
```

This checks the frozen evidence inventory, JSON files, documentation links,
and shell syntax, then rebuilds the Markdown and independently recomputes the
numerical audit. It uses no GPU, model weights, or external services.

For the library tests, lint, and figure dependencies:

```bash
uv sync --locked --group report
uv run --no-sync pytest
uv run --no-sync ruff check src tests scripts writeup
uv run --no-sync ruff format --check src tests scripts writeup
```

See [reproduction.md](docs/reproduction.md) for setup, figure generation,
source-data reconstruction, checkpoint requirements, and adapting Discovery
jobs. The Python 3.11 environment and experiment dependencies are locked;
`report` adds plotting and Word-formatting packages. Pandoc is needed only to
rebuild the Word document.

## Repository layout

| Path | Contents |
| --- | --- |
| [src/mats_latent_safety](src/mats_latent_safety) | Coconut training/inference, batching, evaluation, and provenance helpers. |
| [scripts](scripts/README.md) | Training, generation, scoring, audit, and setup entry points; historical SLURM submissions. |
| [configs](configs/README.md) | Frozen run configs, upstream pins, and storage policy. |
| [manifests](manifests) | Fixed GSM8K, StrongREJECT, and coherence examples with content hashes. |
| [artifacts](artifacts/README.md) | Compact generations, scores, metrics, audit records, logs, and integrity checksums. |
| [project_plan](project_plan) | Preregistered execution protocol. |
| [writeup](writeup) | Source, built Markdown, Word document, figures, and analysis review. |
| [tests](tests) | CPU unit and regression tests; automated checks also run in GitHub Actions. |
| [docs](docs) | Code architecture and reproduction instructions. |

Training ran on USC's Discovery SLURM cluster. Historical configs and job
scripts retain their original cluster paths and run decisions for provenance.
Trained checkpoints live in a private Hugging Face repository named in
[configs/storage.json](configs/storage.json); they are not included or publicly
downloadable. Saved-result reanalysis is available without them.

Safety artifacts contain verbatim harmful or offensive benchmark prompts and
model responses. These are retained to make the evaluation inspectable.

## License and citation

Original code and writing: [MIT](LICENSE). Benchmark data and upstream-derived
code retain the notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Use [CITATION.cff](CITATION.cff) to cite this repository.
