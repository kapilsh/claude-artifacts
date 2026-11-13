#!/usr/bin/env bash
# Module 3 drill: syscalls of a training step, the DataLoader COW "leak", launch jitter.
# Usage: bash drills/m3_cpu.sh [syscalls|cow|jitter|all]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PART="${1:-all}"
hr() { printf '\n==== %s ====\n' "$*"; }
NCPU=$(nproc)

syscalls() {
  hr "strace -f -c: which syscalls does a training step make? (% time = time blocked in the call, summed over threads)"
  out=$(mktemp)
  strace -f -c -o "$out" python3 "$HERE/m3_train_step.py" && head -22 "$out"
  rm -f "$out"
  echo
  echo "Alternative with lower overhead (needs root or perf_event_paranoid<=1):"
  echo "  sudo perf trace -s python3 $HERE/m3_train_step.py 2>&1 | less"
}

cow() {
  hr "DataLoader workers, data as a list of Python objects (refcount writes -> COW copies)"
  python3 "$HERE/m3_cow_leak.py" --storage list
  hr "Same data as one numpy array (no per-item refcounts -> pages stay shared)"
  python3 "$HERE/m3_cow_leak.py" --storage numpy
}

jitter() {
  # pick a launch core and its SMT sibling(s); keep noise off the whole physical core.
  # Launch core = the biggest core by cpu_capacity (Arm hybrids: on DGX Spark CPU 0 is a
  # little A725), else CPU 0.
  pin=$(for f in /sys/devices/system/cpu/cpu[0-9]*/cpu_capacity; do
          [ -r "$f" ] && c=${f%/cpu_capacity} && echo "$(cat "$f") ${c##*cpu}"; done |
        sort -k1,1nr -k2,2n | awk 'NR==1{print $2}')
  pin=${pin:-0}
  sibs=$(cat /sys/devices/system/cpu/cpu$pin/topology/thread_siblings_list 2>/dev/null || echo "$pin")
  noise=$(seq 0 $((NCPU - 1)) | grep -vxF -f <(echo "$sibs" | tr ',' '\n' |
          awk -F- '{for(i=$1;i<=($2==""?$1:$2);i++)print i}') | paste -sd, -)
  nnoise=$(echo "$noise" | tr ',' '\n' | grep -c .)
  hr "Launch jitter: quiet machine"
  python3 "$HERE/m3_launch_jitter.py"
  hr "Launch jitter: $NCPU CPU hogs on all CPUs (launch thread competes for a core)"
  python3 "$HERE/m3_launch_jitter.py" --noise "$NCPU"
  hr "Launch jitter: hogs on the other $nnoise CPUs, launch thread pinned to CPU $pin (siblings $sibs kept quiet)"
  python3 "$HERE/m3_launch_jitter.py" --noise "$nnoise" --noise-cpus "$noise" --pin "$pin"
}

case "$PART" in
  syscalls) syscalls ;; cow) cow ;; jitter) jitter ;;
  all) syscalls; cow; jitter ;;
  *) echo "usage: $0 [syscalls|cow|jitter|all]"; exit 1 ;;
esac
