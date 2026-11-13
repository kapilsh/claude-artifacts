#!/usr/bin/env bash
# Module 4 drill: checkpoint load cold vs warm page cache, and O_DIRECT vs buffered.
# Usage: bash drills/m4_io.sh [CKPT_PATH]
#   uses the Module 2 checkpoint (created if missing). sudo is used only for drop_caches;
#   without sudo it falls back to evicting just this file (posix_fadvise DONTNEED).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
CKPT="${1:-$HOME/osf_ckpt.safetensors}"
[ -f "$CKPT" ] || python3 "$HERE/m2_vm.py" make "$CKPT" 2
SIZE_MB=$(( $(stat -c %s "$CKPT") / 1048576 ))
hr() { printf '\n==== %s ====\n' "$*"; }
drop() {
  if sudo -n true 2>/dev/null; then sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'; echo "(dropped all caches)"
  else python3 "$HERE/m2_vm.py" evict "$CKPT" >/dev/null; echo "(evicted $CKPT only; run with passwordless sudo to drop all caches)"; fi
}
in_cache() { python3 - "$CKPT" <<'EOF'
import ctypes, os, sys
p = sys.argv[1]; sz = os.path.getsize(p); PAGE = os.sysconf("SC_PAGE_SIZE")
libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
fd = os.open(p, os.O_RDONLY)
addr = libc.mmap(None, sz, 1, 1, fd, 0)          # PROT_READ, MAP_SHARED; mapping does not read the file
n = (sz + PAGE - 1) // PAGE; vec = (ctypes.c_ubyte * n)()
libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(sz), vec)   # which pages are resident in page cache
print(f"  {100 * sum(v & 1 for v in vec) / n:.0f}% of file pages are in the page cache (mincore)")
EOF
}
t() { local s=$(date +%s.%N); "$@" >/dev/null 2>&1; local e=$(date +%s.%N); awk -v s=$s -v e=$e -v mb=$SIZE_MB 'BEGIN{printf "  %.2f s  = %.2f GiB/s\n", e-s, mb/1024/(e-s)}'; }

hr "File: $CKPT (${SIZE_MB} MiB) on $(df -T "$CKPT" | awk 'NR==2{print $2" "$1}')"
in_cache

hr "1. Buffered read, COLD"; drop; in_cache; t dd if="$CKPT" of=/dev/null bs=16M
in_cache
hr "2. Buffered read, WARM (served from page cache: memory bandwidth, no device I/O)"; t dd if="$CKPT" of=/dev/null bs=16M
hr "3. O_DIRECT read (bypasses page cache: always goes to the device, even when cached)"; t dd if="$CKPT" of=/dev/null bs=16M iflag=direct
hr "4. Full checkpoint load to GPU: safetensors, COLD then WARM"
cat > /tmp/osf_load.py <<'EOF'
import sys, time, torch
from safetensors.torch import load_file
t0 = time.perf_counter(); sd = load_file(sys.argv[1], device="cuda"); torch.cuda.synchronize()
print(f"  load_file(device='cuda'): {time.perf_counter() - t0:.2f}s for {sum(v.numel() * v.element_size() for v in sd.values()) / 2**30:.2f} GiB")
EOF
drop; python3 /tmp/osf_load.py "$CKPT"; python3 /tmp/osf_load.py "$CKPT"
hr "Device I/O stats while loading (if sysstat is installed: iostat -xz 1)"
command -v iostat >/dev/null && echo "  run: iostat -xz 1   in another terminal during step 1" || echo "  sudo apt install sysstat  for iostat/mpstat/pidstat/sar"
