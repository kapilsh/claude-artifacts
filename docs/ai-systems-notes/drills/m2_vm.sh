#!/usr/bin/env bash
# Module 2 drill: page size, THP, mmap'd checkpoint faults (cold vs warm page cache).
# Usage: bash drills/m2_vm.sh [CKPT_PATH] [GB]
#   CKPT_PATH defaults to $HOME/osf_ckpt.safetensors (put it on a real disk, not tmpfs)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
CKPT="${1:-$HOME/osf_ckpt.safetensors}"
GB="${2:-2}"
PY="python3 $HERE/m2_vm.py"
hr() { printf '\n==== %s ====\n' "$*"; }
tv() {  # run under /usr/bin/time -v and keep only the interesting lines
  /usr/bin/time -v "$@" 2> >(grep -E 'Elapsed|Maximum resident|Major|Minor' | sed 's/^\s*/    /' >&2)
}

hr "System: page size, THP, CUDA unified-memory attributes"
$PY info

[ -f "$CKPT" ] || { hr "Creating ${GB} GiB checkpoint at $CKPT"; $PY make "$CKPT" "$GB"; }
df -T "$(dirname "$CKPT")" | tail -1 | awk '{print "filesystem: "$2"  (tmpfs would make cold == warm)"}'

for mode in read mmap mmap-rand st; do
  hr "load mode=$mode  COLD page cache"
  $PY evict "$CKPT" >/dev/null
  tv $PY load "$CKPT" "$mode"
  hr "load mode=$mode  WARM page cache"
  tv $PY load "$CKPT" "$mode"
done

hr "THP: anonymous 1 GiB, MADV_NOHUGEPAGE vs MADV_HUGEPAGE (THP mode must be madvise or always)"
$PY thp 1

hr "Page cache now holds the checkpoint:"
grep -E '^(Cached|Mapped|AnonHugePages)' /proc/meminfo
echo
echo "Clean up with: rm $CKPT"
