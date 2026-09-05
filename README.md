# Safety drift under explicit and latent reasoning fine-tuning

MATS 12.0 application project (Neel Nanda stream), September 2026.

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

## Read the write-up

- `writeup/FULL_WRITEUP.docx`: the submitted document.
- `writeup/FULL_WRITEUP.built.md`: the same content as Markdown, with raw
  model outputs filled in from the committed result files.
- `writeup/FULL_WRITEUP.md`: the editable source template.
- `writeup/claims_and_numbers.md`: every number in the write-up, traced to the
  committed artifact it came from, with the claim wording each one supports.
- `writeup/analysis_audit/REVIEW.md`: an independent recomputation of the
  analyses, performed before submission, with the corrections it produced.
- `writeup/random_examples.md`: five randomly drawn safety prompts with the raw
  outputs of all three conditions.

## Repository layout

| Path | Contents |
| --- | --- |
| `project_plan/` | The frozen execution protocol (v6.3) and kickoff plan. The protocol is the preregistration; code must not silently change the registered method. |
| `src/mats_latent_safety/` | Project library: Coconut training and inference on Qwen3, GSM8K evaluation, StrongREJECT scoring, provenance helpers. |
| `scripts/` | Entry points for every training, generation, scoring, and audit step. `scripts/slurm/` holds the exact SLURM submissions that produced the results. |
| `configs/` | Frozen YAML/JSON configs for each run, plus `pins.json` (upstream commit pins) and the storage policy. |
| `manifests/` | Fixed prompt and example manifests: GSM8K train/held-out splits, the 60-prompt StrongREJECT set, audit and coherence subsets. |
| `artifacts/discovery/results/` | Compact result records: generations, scores, summaries, gate decisions, and audits for every run referenced in the write-up. |
| `artifacts/discovery/logs/` | SLURM stdout/stderr for those runs. |
| `writeup/` | Write-up source, built Markdown, DOCX, figures, and the scripts that draw them. |
| `research_log.md` | Chronological record of every job, failure, repair, and decision. |
| `tests/` | Unit tests for the library and script helpers (`uv run pytest`). |

Trained checkpoints are not in this repository. They live in a private Hugging
Face repository named in `configs/storage.json`.

## Reproducing

The runs were executed on USC's Discovery SLURM cluster. Paths in
`scripts/slurm/*.sbatch` and `scripts/sync.sh` still point at that cluster's
`home1` and `scratch1` tiers and would need editing for another site.

```bash
uv sync                      # Python 3.11, pinned in uv.lock
./scripts/clone_vendor.sh    # upstream Coconut / StrongREJECT checkouts at the pinned commits
uv run pytest
```

Upstream code lives in the ignored `vendor/` directory and is pinned by commit
in `configs/pins.json`. GPU jobs are submitted through the scripts in
`scripts/slurm/`; each one names the config it consumes.

To rebuild the write-up from the committed results:

```bash
uv run python writeup/build_full_writeup.py
pandoc writeup/FULL_WRITEUP.built.md --resource-path=writeup -o /tmp/writeup_raw.docx
python writeup/format_full_writeup.py /tmp/writeup_raw.docx writeup/FULL_WRITEUP.docx
```

The formatting step needs `python-docx`, which is not part of the locked
environment.

## License

MIT. See `LICENSE`.
