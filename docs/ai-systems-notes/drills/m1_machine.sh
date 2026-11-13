#!/usr/bin/env bash
# Module 1 drill: how is this machine built?
#   Part A  topology: CPUs, caches, NUMA, PCIe devices, GPU <-> CPU/NIC affinity
#   Part B  pinned vs pageable H2D/D2H bandwidth (calls m1_h2d_bandwidth.py)
#   Part C  CPU placement vs copy speed:
#             multi-NUMA box  -> same-node vs cross-node with numactl
#             single-NUMA box -> performance core vs efficiency core with taskset
#
# Usage:  bash drills/m1_machine.sh            (all parts)
#         bash drills/m1_machine.sh topo|bw|place
# Needs:  nvidia-smi, python3 + torch with CUDA.  Optional: hwloc (lstopo), numactl.
#         sudo apt install hwloc numactl
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PART="${1:-all}"
# nvidia-smi prints 00000000:01:00.0 ; sysfs wants 0000:01:00.0
gpu_sysfs_addr() {
  local id dom rest
  id=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | tr 'A-F' 'a-f')
  dom=${id%%:*}; rest=${id#*:}
  printf '%04x:%s' "$((16#$dom))" "$rest"
}
is_spark() { [ "$(uname -m)" = aarch64 ] && nvidia-smi -L 2>/dev/null | grep -q GB10; }
hr() { printf '\n==== %s ====\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

topo() {
  hr "Kernel / arch / page size"
  echo "kernel: $(uname -r)   arch: $(uname -m)   page size: $(getconf PAGESIZE) bytes"

  hr "CPU and caches (lscpu)"
  lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core|Socket|NUMA node|L1d|L2|L3)' || true
  echo "cache line size: $(getconf LEVEL1_DCACHE_LINESIZE 2>/dev/null || cat /sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size) bytes"

  hr "Core types (hybrid CPUs have two kinds)"
  if [ -r /sys/devices/cpu_core/cpus ]; then
    echo "x86 hybrid: P-cores = CPUs $(cat /sys/devices/cpu_core/cpus)   E-cores = CPUs $(cat /sys/devices/cpu_atom/cpus)"
  elif [ -r /sys/devices/system/cpu/cpu0/cpu_capacity ]; then
    echo "Arm: per-CPU capacity (1024 = biggest core):"
    for c in /sys/devices/system/cpu/cpu[0-9]*; do printf '%s=%s ' "${c##*/}" "$(cat "$c/cpu_capacity")"; done; echo
  else
    lscpu -e=CPU,CORE,MAXMHZ 2>/dev/null | head -40
  fi

  hr "NUMA nodes"
  ls -d /sys/devices/system/node/node* 2>/dev/null | xargs -n1 basename
  have numactl && numactl --hardware || echo "(numactl not installed: sudo apt install numactl)"

  hr "Topology tree (lstopo)"
  if have lstopo-no-graphics; then
    lstopo-no-graphics --no-io --of console | head -30
    echo "--- I/O devices ---"
    lstopo-no-graphics --whole-io --of console | grep -E 'PCI|GPU|Net|Block|OpenFabrics|NVMe|CoProc' | head -30
    echo "(for a picture: lstopo topo.png)"
  else
    echo "(hwloc not installed: sudo apt install hwloc)"
  fi

  hr "GPU topology (nvidia-smi topo -m)"
  nvidia-smi topo -m

  hr "GPU PCIe link (current vs max) -- idle GPUs often downtrain to Gen1 to save power"
  nvidia-smi --query-gpu=name,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv

  hr "PCIe tree (lspci -tv), GPUs / NICs / NVMe highlighted"
  lspci -tv 2>/dev/null | grep -nE 'NVIDIA|Mellanox|ConnectX|NVMe|Non-Volatile|Ethernet' | head -20

  hr "Which NUMA node is each GPU / NIC / NVMe attached to? (-1 = no NUMA info / single node)"
  for d in /sys/bus/pci/devices/*; do
    cls=$(cat "$d/class")
    case "$cls" in
      0x0300*|0x0302*) kind=GPU ;; 0x0200*|0x0207*) kind=NIC ;; 0x0108*) kind=NVMe ;; *) continue ;;
    esac
    printf '%-5s %s  numa_node=%s  local_cpus=%s  link=%s x%s\n' "$kind" "${d##*/}" \
      "$(cat "$d/numa_node" 2>/dev/null)" "$(cat "$d/local_cpulist" 2>/dev/null)" \
      "$(cat "$d/current_link_speed" 2>/dev/null | tr -d ' ')" "$(cat "$d/current_link_width" 2>/dev/null)"
  done

  hr "Network interfaces -> PCI address -> driver"
  for n in /sys/class/net/*; do
    [ -e "$n/device" ] || continue
    printf '%-18s %s  %s  speed=%sMb/s\n' "${n##*/}" "$(basename "$(readlink -f "$n/device")")" \
      "$(basename "$(readlink -f "$n/device/driver")" 2>/dev/null)" "$(cat "$n/speed" 2>/dev/null || echo ?)"
  done
  have ibdev2netdev && ibdev2netdev

  if is_spark; then
    hr "DGX Spark specifics"
    echo "* GPU is integrated: no PCIe link to the GPU; it reaches LPDDR5x through NVLink-C2C."
    echo "* ConnectX-7: each QSFP port shows up TWICE (e.g. enp1s0f0np0 and enP2p1s0f0np0),"
    echo "  one per PCIe Gen5 x4 link. One x4 link tops out near ~100 Gb/s; 200 Gb/s needs both."
    echo "* nvidia-smi memory numbers may read N/A or differ from free(1): it is one shared pool."
    free -g | head -2
    nvidia-smi c2c -s 2>/dev/null | head -20 || echo "(nvidia-smi c2c not available on this driver)"
  fi

  if have ibv_devinfo; then
    hr "RDMA devices"
    ibv_devinfo -l
  fi
}

bw() {
  hr "Pinned vs pageable copy bandwidth"
  python3 "$HERE/m1_h2d_bandwidth.py"
}

place() {
  nodes=$(ls -d /sys/devices/system/node/node* 2>/dev/null | wc -l)
  gpu_node=$(cat "/sys/bus/pci/devices/$(gpu_sysfs_addr)/numa_node" 2>/dev/null || echo -1)
  if [ "$nodes" -gt 1 ] && have numactl; then
    [ "$gpu_node" -lt 0 ] && gpu_node=0
    far=$(( (gpu_node + 1) % nodes ))
    hr "Same NUMA node as GPU0 (node $gpu_node): CPU and buffers local"
    numactl --cpunodebind="$gpu_node" --membind="$gpu_node" python3 "$HERE/m1_h2d_bandwidth.py" --sizes 256M
    hr "Cross NUMA: CPU and buffers on node $far, GPU on node $gpu_node"
    numactl --cpunodebind="$far" --membind="$far" python3 "$HERE/m1_h2d_bandwidth.py" --sizes 256M
    hr "Split: CPU on GPU's node, memory on the far node"
    numactl --cpunodebind="$gpu_node" --membind="$far" python3 "$HERE/m1_h2d_bandwidth.py" --sizes 256M
  else
    echo
    echo "Only $nodes NUMA node(s) here, so there is no cross-NUMA path to measure."
    echo "Instead, compare a big core vs a small core. The pageable path is a CPU memcpy"
    echo "into a staging buffer, so it depends on the core; the pinned path is DMA."
    if [ -r /sys/devices/cpu_core/cpus ]; then
      big=$(cut -d, -f1 /sys/devices/cpu_core/cpus | cut -d- -f1)
      small=$(cut -d, -f1 /sys/devices/cpu_atom/cpus | cut -d- -f1)
    elif [ -r /sys/devices/system/cpu/cpu0/cpu_capacity ]; then
      big=$(for c in /sys/devices/system/cpu/cpu[0-9]*; do echo "$(cat $c/cpu_capacity) ${c##*cpu}"; done | sort -n | tail -1 | cut -d' ' -f2)
      small=$(for c in /sys/devices/system/cpu/cpu[0-9]*; do echo "$(cat $c/cpu_capacity) ${c##*cpu}"; done | sort -n | head -1 | cut -d' ' -f2)
    else
      big=0; small=$(( $(nproc) - 1 ))
    fi
    hr "Pinned to big core (CPU $big)"
    taskset -c "$big" python3 "$HERE/m1_h2d_bandwidth.py" --sizes 256M
    hr "Pinned to small core (CPU $small)"
    taskset -c "$small" python3 "$HERE/m1_h2d_bandwidth.py" --sizes 256M
  fi
}

case "$PART" in
  topo) topo ;; bw) bw ;; place) place ;;
  all) topo; bw; place ;;
  *) echo "usage: $0 [topo|bw|place|all]"; exit 1 ;;
esac
