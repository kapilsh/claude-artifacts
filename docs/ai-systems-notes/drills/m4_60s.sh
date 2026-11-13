#!/usr/bin/env bash
# Module 4 drill: Brendan Gregg's "Linux performance analysis in 60 seconds",
# plus GPU additions. Run it on the host WHILE a training job is running.
#   bash drills/m4_60s.sh            (needs sysstat: sudo apt install sysstat)
#   bash drills/m4_60s.sh <PID>      also per-thread view of the training process
set -u
PID="${1:-}"
hr() { printf '\n==== %s ====\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }
run() { hr "$*"; if have "$1"; then timeout 6 "$@" 2>&1 | head -40; else echo "($1 not installed: sudo apt install sysstat)"; fi; }

run uptime                              # load averages: rising/falling? > nCPU = queueing
hr "dmesg | tail"; (dmesg 2>/dev/null || sudo -n dmesg 2>/dev/null || echo "(need sudo for dmesg)") | tail -15   # OOM kills, Xid GPU errors, NIC link flaps
run vmstat 1 5                          # r > nCPU = CPU saturation; si/so = swapping; wa = I/O wait
run mpstat -P ALL 1 1                   # one hot CPU at 100% = single-threaded bottleneck (launch thread, GIL)
run pidstat 1 3                         # which processes use CPU; %wait = waiting for a CPU
run iostat -xz 1 3                      # r/s, rkB/s, await, %util per device
run free -m                             # 'available' is what matters; buff/cache is reclaimable
run sar -n DEV 1 3                      # NIC throughput vs line rate
run sar -n TCP,ETCP 1 3                 # retransmits = network trouble (TCP bootstrap/NCCL socket transport)
hr "top -b -n1 | head -20"; top -b -n1 | head -20

hr "GPU: nvidia-smi dmon (sm%, mem%, PCIe rx/tx MB/s)"
have nvidia-smi && timeout 6 nvidia-smi dmon -s ut -c 5
hr "GPU: throttle reasons and Xid errors"
have nvidia-smi && nvidia-smi -q -d PERFORMANCE | grep -E 'Active|Idle|Thermal|Power|Sync' | head -12

if [ -n "$PID" ]; then
  hr "Threads of PID $PID: voluntary vs involuntary context switches"
  for t in /proc/$PID/task/*; do
    printf '  %-8s %-18s vol=%-8s invol=%s\n' "${t##*/}" "$(cat $t/comm)" \
      "$(awk '/^voluntary_ctxt_switches/{print $2}' $t/status)" "$(awk '/^nonvoluntary_ctxt_switches/{print $2}' $t/status)"
  done | sort -t= -k3 -nr | head -15
  hr "PID $PID memory: RSS/PSS/swap"; grep -E '^(Rss|Pss|Swap|Private)' /proc/$PID/smaps_rollup
fi

cat <<'EOF'

Next steps when something stands out:
  perf top -p <PID>                          hot functions (user + kernel)
  sudo perf trace -s -p <PID>                syscall summary for a running process
  sudo bpftrace -e 'tracepoint:sched:sched_switch /args.prev_pid == <TID>/ { @[args.prev_state] = count(); }'
  nsys profile --trace=cuda,nvtx,osrt --sample=cpu -o prof python train.py
EOF
