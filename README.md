# MATS latent safety

This repository implements Session S1 and the later registered experiment in
`project_plan/MATS_execution_protocol_v6_3.md`. The protocol and
`project_plan/kickoff_plan.md` are authoritative; code must not silently change
the registered method.

## Discovery layout

- `/home1/$USER/mats_latent_safety`: canonical checkout and `.venv`
- `/scratch1/$USER/mats_latent_safety`: caches, datasets, generations, logs,
  and working checkpoints
- private HF repo: durable copy of every unique trained checkpoint
- `project2`: prohibited for this project

Bootstrap only after `myquota` confirms enough home1 space:

```bash
./scripts/sync.sh --push
ssh discovery 'bash /home1/$USER/mats_latent_safety/scripts/bootstrap_env.sh'
```

Run local unit tests with `uv run pytest`. GPU jobs are submitted through the
specific scripts in `scripts/slurm/`; do not weaken the A100-80GB constraint on
4B preflights.

Upstream code lives in ignored `vendor/` checkouts and is pinned in
`configs/pins.json`. Project-specific code is tracked under `src/`.

