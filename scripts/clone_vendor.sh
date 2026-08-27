#!/usr/bin/env bash
# Clone/fetch exact ignored upstream references into the home1 checkout.
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/home1/$USER/mats_latent_safety}"
mkdir -p "$CODE_ROOT/vendor"

pin() {
  local name="$1"
  local remote="$2"
  local revision="$3"
  local target="$CODE_ROOT/vendor/$name"
  if [[ ! -d "$target/.git" ]]; then
    git clone --filter=blob:none "$remote" "$target"
  fi
  if [[ "$(git -C "$target" remote get-url origin)" != "$remote" ]]; then
    echo "unexpected origin for $target" >&2
    exit 2
  fi
  git -C "$target" fetch origin "$revision"
  git -C "$target" checkout --detach "$revision"
  test "$(git -C "$target" rev-parse HEAD)" = "$revision"
  printf '%s %s\n' "$name" "$revision"
}

pin facebookresearch-coconut https://github.com/facebookresearch/coconut.git 27273cb8cca4bb763c041a63b036d0c3b7cbbb48
pin wassname-coconut https://github.com/wassname/coconut.git 60ade4092a0e9f5ee635be435b56ab6a3ce8c964
pin strong_reject https://github.com/dsbowen/strong_reject.git 7a551d5b440ec7b75d4f6f5bb7c1719965b76b47
pin self-jailbreaking https://github.com/BatsResearch/self-jailbreaking.git be6033c6369399d626a00479ee7263aea286ec63

