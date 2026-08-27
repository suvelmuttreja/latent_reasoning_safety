You have access to discovery, my university's hpc slurm jobs server. It's directories:
home1: small and quota-capped, lightly snapshotted: code, configs, and environments.
scratch1: huge, fast, unbacked, and purged on capacity: working files only, never the only copy.
project2: can't be used.

The private GitHub remote `origin` is the off-site progressive history for this
project. After each verified project commit, push `main` to `origin/main`.
Preserve the complete commit graph: never squash, rewrite, or force-push the
project history unless the user explicitly requests it.
