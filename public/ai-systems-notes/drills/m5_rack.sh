#!/usr/bin/env bash
# Coherence drill: what kind of CPU–GPU memory system is this, and is it tuned the way NVIDIA's guides say?
#
#   platform  what is this box: CPU, GPUs, page size, coherent-memory mode
#   numa      NUMA nodes and how much memory each one offers (GPU HBM shows up here in NUMA mode)
#   tune      the Grace performance-tuning checklist, PASS/WARN per item (only enforced on Arm/Grace)
#   fabric    multi-node NVLink: fabric registration, IMEX service, IMEX channels, NVLink status
#   bw        build + run m5_coherent.cu (copy engine vs direct SM access to CPU memory, fabric probe)
#   place     where do malloc'd pages land? (m5_numa_place.py, with numactl variants if available)
#
# Usage:  bash drills/m5_rack.sh            (all parts)
#         bash drills/m5_rack.sh tune fabric
# Safe:   read-only except for building /tmp/m5_coherent. Nothing is changed on the system.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PARTS="${*:-platform numa tune fabric bw place}"
hr()   { printf '\n==== %s ====\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }
ok()   { printf '  PASS  %s\n' "$*"; }
warn() { printf '  WARN  %s\n' "$*"; }
info() { printf '  info  %s\n' "$*"; }

ARCH=$(uname -m)
GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c | sed 's/^ *//' | paste -sd';' -)
CPU_MODEL=$(lscpu 2>/dev/null | awk -F: '/Model name/{gsub(/^ +/,"",$2);print $2;exit}')
# CoherentGPUMemoryMode is "numa" or "driver" (CDMM) on Grace-attached GPUs, "" on PCIe GPUs
coh_mode() { awk -F': *' '/^CoherentGPUMemoryMode/{gsub(/"/,"",$2);print $2}' /proc/driver/nvidia/params 2>/dev/null; }
is_coherent() { [ -n "$(coh_mode)" ]; }
platform_name() {
  case "$GPUS" in
    *GB10*)  echo "DGX Spark (GB10: integrated, coherent, unified LPDDR5x)";;
    *GB300*|*"B300"*) [ "$ARCH" = aarch64 ] && echo "GB300 (Grace + Blackwell Ultra)" || echo "HGX B300 (x86 + PCIe)";;
    *GB200*|*"B200"*) [ "$ARCH" = aarch64 ] && echo "GB200 (Grace + Blackwell)" || echo "HGX B200 (x86 + PCIe)";;
    *GH200*) echo "GH200 (Grace + Hopper)";;
    *H100*|*H200*|*H800*) echo "HGX Hopper (x86 + PCIe + NVSwitch)";;
    *Rubin*|*R100*|*R200*|*VR*) echo "Rubin generation (check Vera CPU below)";;
    *) echo "discrete GPU(s) over PCIe";;
  esac
}

platform() {
  hr "Platform"
  echo "  CPU     : ${CPU_MODEL:-?} ($ARCH, $(nproc) CPUs online)"
  echo "  GPUs    : ${GPUS:-none visible}"
  echo "  guess   : $(platform_name)"
  echo "  kernel  : $(uname -r)   page size: $(getconf PAGESIZE) B"
  have nvidia-smi && echo "  driver  : $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  if is_coherent; then
    local m; m=$(coh_mode)
    echo "  coherent memory mode: $m   ($( [ "$m" = driver ] && echo 'CDMM: GPU memory hidden from the kernel' || echo 'NUMA: GPU memory onlined as NUMA nodes'))"
  else
    echo "  coherent memory mode: n/a (CoherentGPUMemoryMode empty/absent in /proc/driver/nvidia/params: PCIe GPU)"
  fi
  [ -r /sys/module/nvidia_uvm/parameters/uvm_disable_sam_migration ] && \
    echo "  uvm_disable_sam_migration = $(cat /sys/module/nvidia_uvm/parameters/uvm_disable_sam_migration)"
}

numa() {
  hr "NUMA nodes (CPUs + memory per node)"
  for n in /sys/devices/system/node/node[0-9]*; do
    local id=${n##*node} cpus mem
    cpus=$(cat "$n/cpulist"); mem=$(awk '/MemTotal/{printf "%.1f GiB", $4/1048576}' "$n/meminfo")
    [ "$mem" = "0.0 GiB" ] && [ -z "$cpus" ] && continue   # skip empty placeholder nodes
    printf '  node %-3s cpus=%-14s mem=%s%s\n' "$id" "${cpus:-none}" "$mem" "$( [ -z "$cpus" ] && echo '   <- memory-only node (GPU HBM in NUMA mode?)')"
  done
  local total empty
  total=$(ls -d /sys/devices/system/node/node[0-9]* | wc -l)
  empty=$(for n in /sys/devices/system/node/node[0-9]*; do [ -z "$(cat $n/cpulist)" ] && awk '/MemTotal/{if($4==0)print}' $n/meminfo; done | wc -l)
  info "$total NUMA nodes exist, $empty are empty placeholders (GH200/GB200 reserve nodes per GPU, e.g. for MIG)"
  if have nvidia-smi; then
    echo "  GPU -> CPU affinity / NUMA (nvidia-smi topo -m, last columns):"
    nvidia-smi topo -m 2>/dev/null | sed 's/^/    /' | head -20
  fi
  have numastat && { echo "  numastat -m (MemTotal/MemFree per node):"; numastat -m 2>/dev/null | grep -E 'Node|MemTotal|MemFree' | sed 's/^/    /'; }
}

tune() {
  hr "Grace / coherent-platform tuning checklist"
  local grace=0; [ "$ARCH" = aarch64 ] && grace=1
  [ $grace = 1 ] || info "not an Arm box: items below are printed for reference, WARNs don't apply here"
  local ps; ps=$(getconf PAGESIZE)
  if [ "$ps" = 65536 ]; then ok "64 KiB kernel pages"; else
    [ $grace = 1 ] && warn "page size $ps B; NVIDIA recommends a 64K kernel on Grace (GPU uses CPU PTEs via ATS)" || info "page size $ps B (x86: 4 KiB is the only base size)"; fi
  if grep -qw 'init_on_alloc=0' /proc/cmdline; then ok "init_on_alloc=0 on the kernel cmdline"; else
    if grep -q 'CONFIG_INIT_ON_ALLOC_DEFAULT_ON=y' "/boot/config-$(uname -r)" 2>/dev/null; then
      [ $grace = 1 ] && warn "kernel zeroes every page on allocation (CONFIG_INIT_ON_ALLOC_DEFAULT_ON=y); slows cudaMalloc when GPU memory is OS-managed. Add init_on_alloc=0" || info "init_on_alloc defaults ON in this kernel (harmless for PCIe GPUs)"
    else ok "init_on_alloc not enabled by default"; fi; fi
  local nb; nb=$(cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo n/a)
  if [ "$nb" = 0 ] || [ "$nb" = n/a ]; then ok "automatic NUMA balancing off ($nb)"; else
    [ $grace = 1 ] && warn "kernel.numa_balancing=$nb: extra hinting faults hurt GPU-heavy jobs; set 0" || info "kernel.numa_balancing=$nb"; fi
  local thp; thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)
  info "THP global: $thp"
  for f in /sys/kernel/mm/transparent_hugepage/hugepages-*kB/enabled; do
    [ -r "$f" ] || continue; local sz=${f%/enabled}; sz=${sz##*hugepages-}
    case "$sz" in 524288kB|2048kB) printf '        per-size %-10s %s\n' "$sz" "$(cat "$f")";; esac
  done
  [ "$ps" = 65536 ] && info "64K kernel: PMD THP = 512 MiB. Guide: hugepages-524288kB=never, hugepages-2048kB=madvise (mTHP, kernel 6.9+)"
  info "vm.min_free_kbytes = $(cat /proc/sys/vm/min_free_kbytes)   compaction_proactiveness = $(cat /proc/sys/vm/compaction_proactiveness 2>/dev/null || echo n/a)"
  if grep -qE 'iommu.passthrough=1|iommu=pt' /proc/cmdline; then ok "IOMMU passthrough on the cmdline"; else info "no iommu passthrough on cmdline (guide: optional iommu.passthrough=1 on Grace)"; fi
  local gov; gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)
  if [ "$gov" = performance ]; then ok "cpufreq governor: performance"; else info "cpufreq governor: $gov (guide: cpupower frequency-set -g performance)"; fi
  local swap; swap=$(awk '/SwapTotal/{printf "%.0f", $2/1048576}' /proc/meminfo)
  if [ "$(coh_mode)" = numa ]; then
    [ "$swap" -gt 0 ] && ok "swap ${swap} GiB (NUMA mode: kernel can migrate pages into GPU nodes; guide wants swap >= 1/4-1/2 of GPU memory)" \
                      || warn "no swap in NUMA mode: pages the kernel moves into GPU memory can't be reclaimed"
  else info "swap ${swap} GiB"; fi
  systemctl is-active --quiet libvirtd 2>/dev/null && info "libvirtd active: its virbr0 iptables rules add overhead on bare metal (guide)"
  return 0
}

fabric() {
  hr "Multi-node NVLink fabric"
  if ! have nvidia-smi; then info "no nvidia-smi"; return; fi
  local fq; fq=$(nvidia-smi -q 2>/dev/null | grep -A7 -E '^ +Fabric$' | head -8)
  if [ -n "$fq" ]; then echo "$fq" | sed 's/^ */  /'
    echo "$fq" | grep -q 'Completed' && ok "GPU 0 registered with the NVLink fabric" || info "fabric registration not completed (normal off NVL72)"
  else info "nvidia-smi -q has no Fabric section on this GPU (normal for PCIe / single-node)"; fi
  if systemctl list-unit-files 2>/dev/null | grep -q nvidia-imex; then
    info "nvidia-imex.service: $(systemctl is-active nvidia-imex 2>/dev/null)"
    [ -r /etc/nvidia-imex/nodes_config.cfg ] && info "nodes_config.cfg lists $(grep -cv '^\s*$' /etc/nvidia-imex/nodes_config.cfg) peers"
    have nvidia-imex-ctl && nvidia-imex-ctl -N 2>&1 | head -20 | sed 's/^/    /'
  else info "IMEX not installed (only needed for multi-node NVLink: GB200/GB300 NVL72)"; fi
  if grep -q nvidia-caps-imex-channels /proc/devices; then
    info "char device major: $(awk '/nvidia-caps-imex-channels/{print $1}' /proc/devices)"
    if [ -d /dev/nvidia-caps-imex-channels ]; then ls -l /dev/nvidia-caps-imex-channels/ | sed 's/^/    /'
    else info "driver supports IMEX channels but /dev/nvidia-caps-imex-channels/ doesn't exist (fine off NVL72; on NVL72: no NVreg_CreateImexChannel0, or not passed into this container)"; fi
  else info "no nvidia-caps-imex-channels in /proc/devices"; fi
  echo "  NVLink status (first GPU):"
  nvidia-smi nvlink -s -i 0 2>/dev/null | head -8 | sed 's/^/    /' || true
  nvidia-smi nvlink -s -i 0 2>/dev/null | grep -q GB/s || info "no active NVLink links on GPU 0 (expected on a 4090 or Spark)"
}

bw() {
  hr "Coherence + bandwidth probe (m5_coherent.cu)"
  bash "$HERE/build_m5.sh" /tmp/m5_coherent || { warn "build failed: install a CUDA toolkit that supports this GPU"; return; }
  /tmp/m5_coherent "${M5_MIB:-1024}"
}

place() {
  hr "Where do malloc'd pages land? (m5_numa_place.py)"
  python3 "$HERE/m5_numa_place.py"
  if have numactl && [ "$(ls -d /sys/devices/system/node/node[0-9]* | wc -l)" -gt 1 ]; then
    local last; last=$(ls -d /sys/devices/system/node/node[0-9]* | sed 's/.*node//' | sort -n | tail -1)
    echo "  -- numactl --cpunodebind=0 --membind=0 (force CPU node 0) --"
    numactl --cpunodebind=0 --membind=0 python3 "$HERE/m5_numa_place.py" --quiet
    echo "  -- numactl --preferred=$last (highest node: GPU HBM in NUMA mode; no effect in CDMM) --"
    numactl --preferred="$last" python3 "$HERE/m5_numa_place.py" --quiet || true
  else info "single NUMA node or no numactl: only the default placement is shown"; fi
}

for p in $PARTS; do "$p"; done
