#!/usr/bin/env bash
# Build m5_coherent.cu into $1 with the newest nvcc that can target this GPU.
# Tries /usr/local/cuda*/bin/nvcc (newest first), then nvcc on PATH; -arch from nvidia-smi.
OUT=${1:-/tmp/m5_coherent}
HERE="$(cd "$(dirname "$0")" && pwd)"
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ARCH=${CC:+sm_$CC}; ARCH=${ARCH:-native}
for nvcc in $(ls -d /usr/local/cuda*/bin/nvcc 2>/dev/null | sort -V -r) $(command -v nvcc); do
  if "$nvcc" -O2 -arch="$ARCH" -o "$OUT" "$HERE/m5_coherent.cu" -lcuda 2>/dev/null; then
    echo "built with $nvcc -arch=$ARCH" >&2; exit 0
  fi
done
echo "could not build m5_coherent.cu (need a CUDA toolkit that supports $ARCH)" >&2; exit 1
