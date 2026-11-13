#!/usr/bin/env bash
# One-page machine profile: runs the read-only drills and writes results/<hostname>.md
#
#   bash drills/report.sh            # full (≈3–5 min: includes bandwidth + launch-jitter runs)
#   bash drills/report.sh --quick    # topology/attribute checks only (≈30 s)
#
# Safe: changes nothing on the system. Needs nvidia-smi; python3+torch and nvcc for the
# measurement sections (skipped with a note if missing). Paste or commit the result file;
# CLAUDE.md lists which machines still need one.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
QUICK=0; [ "${1:-}" = "--quick" ] && QUICK=1
HOST=$(hostname -s)
OUT="$ROOT/results/$HOST.md"
mkdir -p "$ROOT/results"
have() { command -v "$1" >/dev/null 2>&1; }
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

section() {   # section "title" timeout cmd...
  local title=$1 to=$2; shift 2
  printf '\n## %s\n\n```text\n' "$title" >>"$OUT"
  timeout "$to" "$@" >>"$OUT" 2>&1 || printf '(exit %s)\n' "$?" >>"$OUT"
  printf '```\n' >>"$OUT"
  printf '  done: %s\n' "$title"
}

# --- measurements first, so the summary can quote them ---
COH="$TMP/coherent.txt"
if bash "$HERE/build_m5.sh" "$TMP/m5c" 2>"$TMP/nvcc.err"; then
  timeout 300 "$TMP/m5c" "$([ $QUICK = 1 ] && echo 256 || echo 1024)" >"$COH" 2>&1
else
  echo "m5_coherent.cu not built: $(cat "$TMP/nvcc.err")" >"$COH"
fi
attr() { awk -v k="$1" '$1==k{print $2; exit}' "$COH"; }
num()  { grep -m1 "$1" "$COH" | grep -oE '[0-9]+\.[0-9]+ GB/s' | sed -n "${2:-1}p"; }

ATS=$(attr ..._USES_HOST_PAGE_TABLES); PAGEABLE=$(attr PAGEABLE_MEMORY_ACCESS); INTEG=$(attr INTEGRATED)
NUMACFG=$(attr NUMA_CONFIG); GDR=$(attr GPU_DIRECT_RDMA_SUPPORTED)
MODE=$(awk -F': *' '/^CoherentGPUMemoryMode/{gsub(/"/,"",$2);print $2}' /proc/driver/nvidia/params 2>/dev/null)
FABRIC=$(nvidia-smi -q 2>/dev/null | grep -A3 -E '^ +Fabric$' | awk -F': *' '/State/{print $2; exit}')
CLIQUE=$(nvidia-smi -q 2>/dev/null | awk -F': *' '/CliqueId/{print $2; exit}')
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
NVL=$(nvidia-smi topo -m 2>/dev/null | grep -oE 'NV[0-9]+' | sort -u | paste -sd, -)
IB=$(ls /sys/class/infiniband 2>/dev/null | paste -sd, -)

case "$ATS/$PAGEABLE/$INTEG" in
  1/*/1) Q1="integrated, coherent (ATS 1, INTEGRATED 1): CPU and GPU share one DRAM";;
  1/*)   Q1="coherent link with ATS (GPU walks CPU page tables)";;
  0/1/*) Q1="PCIe + HMM (GPU faults migrate pages)";;
  0/0/*) Q1="PCIe, not coherent: pinned/managed memory only";;
  *)     Q1="unknown (coherence probe didn't run)";;
esac
case "$MODE" in
  numa)   Q2="kernel + driver: GPU memory onlined as NUMA nodes (NUMA mode)";;
  driver) Q2="driver only (CDMM): GPU NUMA nodes present but empty";;
  *)      [ "$INTEG" = 1 ] && Q2="kernel (integrated: it's all system RAM)" || Q2="driver only (discrete GPU)";;
esac
if [ "$FABRIC" = Completed ]; then Q3="multi-node NVLink fabric: registered, clique $CLIQUE ($NGPU local GPUs; links ${NVL:-?})"
elif [ -n "$NVL" ]; then Q3="$NGPU GPUs on NVLink in this box (${NVL}); no multi-node fabric"
else Q3="none ($NGPU GPU(s), no NVLink)"; fi
Q4="RDMA devices: ${IB:-none}; GPUDirect RDMA attribute: ${GDR:-?}"

{
  echo "# Machine profile: $HOST"
  echo
  echo "_Generated $(date -u +%Y-%m-%dT%H:%MZ) by drills/report.sh$([ $QUICK = 1 ] && echo ' --quick')._"
  echo
  echo "- **CPU:** $(lscpu | awk -F: '/Model name/{gsub(/^ +/,"",$2);print $2;exit}') · $(uname -m) · $(nproc) CPUs · $(ls -d /sys/devices/system/node/node[0-9]* | wc -l) NUMA nodes"
  echo "- **GPU:** $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sort | uniq -c | sed 's/^ *//' | paste -sd';' -)"
  echo "- **Driver / kernel / page size:** $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1) / $(uname -r) / $(getconf PAGESIZE) B"
  have python3 && echo "- **torch:** $(python3 -c 'import torch;print(torch.__version__, "cuda", torch.version.cuda)' 2>/dev/null || echo 'not importable')"
  echo
  echo "## Four questions"
  echo
  echo "1. **How does the GPU reach CPU memory?** $Q1"
  echo "2. **Who manages GPU memory?** $Q2"
  echo "3. **NVLink domain?** $Q3"
  echo "4. **Way out?** $Q4"
  echo
  echo "## Headline numbers"
  echo
  echo "| measurement | value |"
  echo "|---|---|"
  echo "| H2D pageable / pinned (cudaMemcpy) | $(num 'H2D pageable' 1) / $(num 'H2D pageable' 2) |"
  echo "| D2H pageable / pinned | $(num 'D2H pageable' 1) / $(num 'D2H pageable' 2) |"
  echo "| SM read: GPU memory | $(num 'GPU memory (reference)') |"
  echo "| SM read: cudaHostAlloc zero-copy | $(num 'cudaHostAlloc') |"
  echo "| SM read: plain malloc (HMM/ATS) | $(num 'plain malloc' || true)$(grep -q 'plain malloc.*skipped' "$COH" && echo 'n/a (no pageable access)') |"
  echo "| fabric handle create | $(grep -m1 'cuMemCreate(FABRIC)' "$COH" | sed 's/^ *//' || echo 'not supported') |"
} >"$OUT"

printf '\n## m5_coherent.cu (full)\n\n```text\n' >>"$OUT"; cat "$COH" >>"$OUT"; printf '```\n' >>"$OUT"
echo "  done: coherence probe"
section "m5_rack.sh platform numa tune fabric" 120 bash "$HERE/m5_rack.sh" platform numa tune fabric
section "m1_machine.sh topo" 120 bash "$HERE/m1_machine.sh" topo
section "m2_vm.py info (page size, THP, CUDA VM attributes)" 60 python3 "$HERE/m2_vm.py" info
if [ $QUICK = 0 ]; then
  section "m1_h2d_bandwidth.py (pinned vs pageable, CPU time blocked)" 300 python3 "$HERE/m1_h2d_bandwidth.py"
  section "m5_rack.sh place (NUMA placement of malloc'd pages)" 120 bash "$HERE/m5_rack.sh" place
  section "m3_cpu.sh jitter (launch jitter vs CPU noise and SMT siblings)" 600 bash "$HERE/m3_cpu.sh" jitter
fi
echo
echo "wrote $OUT"
