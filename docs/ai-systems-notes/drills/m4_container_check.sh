#!/bin/sh
# Module 4 drill: "what will break NCCL / DataLoader in this container?"
# POSIX sh so it runs in minimal images too. Run it inside the container:
#
#   docker run --rm -v "$PWD/drills:/d:ro" IMAGE sh /d/m4_container_check.sh
#   docker run --rm -v "$PWD/drills:/d:ro" --shm-size=16g --ulimit memlock=-1 IMAGE sh /d/m4_container_check.sh
#   docker run --rm -v "$PWD/drills:/d:ro" --ipc=host --ulimit memlock=-1 --gpus all IMAGE sh /d/m4_container_check.sh
#
# Each line prints OK / WARN / FAIL with the reason.
ok()   { printf '  OK    %s\n' "$*"; }
warn() { printf '  WARN  %s\n' "$*"; }
fail() { printf '  FAIL  %s\n' "$*"; }
kb_h() { awk -v k="$1" 'BEGIN{ if (k>=1048576) printf "%.1f GiB", k/1048576; else if (k>=1024) printf "%.0f MiB", k/1024; else printf "%d KiB", k }'; }

echo "== identity"
echo "  hostname $(hostname)   pid of this shell: $$  (pid 1 is: $(cat /proc/1/comm 2>/dev/null))"
[ "$(cat /proc/1/comm 2>/dev/null)" = systemd ] && echo "  looks like the host (pid 1 = systemd)" || echo "  looks like a container (own PID namespace)"

echo "== /dev/shm (DataLoader worker -> main tensors; NCCL SHM transport)"
shm_kb=$(df -k /dev/shm 2>/dev/null | awk 'NR==2{print $2}')
if [ -z "$shm_kb" ]; then fail "/dev/shm missing"
elif [ "$shm_kb" -le 131072 ]; then fail "/dev/shm is $(kb_h "$shm_kb") (Docker default is 64 MiB). Use --shm-size=16g or --ipc=host"
elif [ "$shm_kb" -le 2097152 ]; then warn "/dev/shm is $(kb_h "$shm_kb"); large batches x workers may not fit"
else ok "/dev/shm is $(kb_h "$shm_kb")"; fi

echo "== locked-memory limit (RDMA ibv_reg_mr, cudaHostRegister on some setups)"
ml=$(ulimit -l)
if [ "$ml" = unlimited ]; then ok "ulimit -l unlimited"
elif [ "$ml" -lt 1048576 ]; then fail "ulimit -l = $ml KiB: RDMA memory registration will fail (NCCL WARN Call to ibv_reg_mr failed). Use --ulimit memlock=-1"
else warn "ulimit -l = $ml KiB"; fi

echo "== cgroup v2 limits (what this container may use)"
cg=/sys/fs/cgroup
if [ -f $cg/cgroup.controllers ]; then
  mm=$(cat $cg/memory.max 2>/dev/null)
  if [ -z "$mm" ]; then ok "no memory.max here (root cgroup: probably the host)"
  elif [ "$mm" = max ]; then ok "memory.max unlimited"
  else warn "memory.max = $(kb_h $(( mm / 1024 ))) (page cache counts too; OOM-kill above it)"; fi
  cm=$(cat $cg/cpu.max 2>/dev/null)
  case "$cm" in "") ok "no cpu.max here (root cgroup)" ;; max*) ok "cpu.max unlimited" ;; *) q=${cm% *}; p=${cm#* }; warn "cpu.max = $cm -> $(( q / p )) CPUs worth of quota; launch threads get THROTTLED, not just slowed";; esac
  [ -f $cg/cpu.stat ] && grep -E 'nr_throttled|throttled_usec' $cg/cpu.stat | sed 's/^/        /'
  pm=$(cat $cg/pids.max 2>/dev/null); { [ -z "$pm" ] || [ "$pm" = max ] || [ "${pm:-0}" -ge 8192 ]; } && ok "pids.max = ${pm:-none (root cgroup)}" || warn "pids.max = $pm (threads count: torch + DataLoader + NCCL can use hundreds)"
  [ -f $cg/memory.events ] && grep -E '^(oom|oom_kill) ' $cg/memory.events | sed 's/^/        memory.events: /'
else warn "cgroup v1 or not mounted"; fi
echo "  CPUs allowed: $(grep Cpus_allowed_list /proc/self/status | awk '{print $2}')   nproc: $(nproc 2>/dev/null)"

echo "== GPUs and RDMA devices"
ls /dev/nvidia[0-9]* >/dev/null 2>&1 && ok "GPU device nodes: $(ls /dev/nvidia[0-9]* | tr '\n' ' ')" || warn "no /dev/nvidiaN (no --gpus / CDI device)"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L | sed 's/^/        /'
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
[ -n "${NVIDIA_VISIBLE_DEVICES:-}" ] && echo "  NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES"
if [ -d /dev/infiniband ]; then ok "/dev/infiniband: $(ls /dev/infiniband | tr '\n' ' ')"
else warn "no /dev/infiniband: NCCL falls back to TCP sockets (pass --device=/dev/infiniband or use the RDMA device plugin)"; fi

echo "== network namespace (NCCL_SOCKET_IFNAME, RoCE needs the host netdevs)"
for n in /sys/class/net/*; do printf '        %-16s %s\n' "${n##*/}" "$(cat $n/operstate 2>/dev/null)"; done
ls /sys/class/net | grep -qvE '^(lo|eth0)$' || warn "only lo/eth0 visible: this is a bridged netns. RoCE/IB netdevs are not here; consider --network=host"
env | grep -E '^NCCL_' | sed 's/^/        /'

echo "== shm write test (64 MiB + 128 MiB)"
for mb in 64 128; do
  if dd if=/dev/zero of=/dev/shm/osf_test bs=1M count=$mb 2>/dev/null; then ok "wrote ${mb} MiB to /dev/shm"
  else fail "writing ${mb} MiB to /dev/shm failed: this is the 'bus error in DataLoader worker' outage"; fi
  rm -f /dev/shm/osf_test
done
