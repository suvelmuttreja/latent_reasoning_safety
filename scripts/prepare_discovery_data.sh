#!/usr/bin/env bash
# Download only public, regenerable source data to scratch1 and verify hashes.
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/scratch1/$USER/mats_latent_safety}"
SOURCE_DIR="$WORK_ROOT/data/sources"
mkdir -p "$SOURCE_DIR" "$WORK_ROOT/tmp"

curl --fail -L -sS \
  https://huggingface.co/datasets/openai/gsm8k/resolve/740312add88f781978c0658806c59bc2815b9866/main/train-00000-of-00001.parquet \
  -o "$SOURCE_DIR/gsm8k-train.parquet"
curl --fail -L -sS \
  https://huggingface.co/datasets/openai/gsm8k/resolve/740312add88f781978c0658806c59bc2815b9866/main/test-00000-of-00001.parquet \
  -o "$SOURCE_DIR/gsm8k-test.parquet"
curl --fail -L -sS \
  https://raw.githubusercontent.com/alexandrasouly/strongreject/f7cad6c17e624e21d8df2278e918ae1dddb4cb56/strongreject_dataset/strongreject_small_dataset.csv \
  -o "$SOURCE_DIR/strongreject-small.csv"

EXPECTED="$WORK_ROOT/tmp/source-sha256.expected"
cat >"$EXPECTED" <<'EOF'
ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d  gsm8k-train.parquet
ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59  gsm8k-test.parquet
3051340e3e89a3598d764dde497d5fcda80e258ac05cc35e6bd87228ac3d467c  strongreject-small.csv
EOF
(cd "$SOURCE_DIR" && shasum -a 256 -c "$EXPECTED")
