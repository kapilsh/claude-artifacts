#!/usr/bin/env python3
"""Module 2 drill: virtual memory you can measure.

Subcommands
  info                 page size, THP settings, huge page pool, CUDA unified-memory attributes
  make  PATH [GB]      write a safetensors checkpoint of ~GB gigabytes (default 2)
  load  PATH MODE      load it and report page faults + time.  MODE is one of
                         read     f.read() into a private buffer (copies out of page cache)
                         mmap     mmap + touch one byte per page (zero-copy view of page cache)
                         mmap-rand  mmap + MADV_RANDOM, touch 64k random pages (no readahead)
                         st       safetensors.safe_open(...).get_tensor() for every tensor
  evict PATH           drop PATH from the page cache (posix_fadvise DONTNEED, no root needed)
  thp  [GB]            anonymous memory: first-touch faults and random-access time,
                       with MADV_NOHUGEPAGE vs MADV_HUGEPAGE

Typical session (see m2_vm.sh, which runs all of this):
  python3 drills/m2_vm.py make /tmp/ckpt.safetensors 2
  python3 drills/m2_vm.py evict /tmp/ckpt.safetensors
  /usr/bin/time -v python3 drills/m2_vm.py load /tmp/ckpt.safetensors mmap   # cold
  /usr/bin/time -v python3 drills/m2_vm.py load /tmp/ckpt.safetensors mmap   # warm
"""
import ctypes
import mmap
import os
import resource
import sys
import time

PAGE = os.sysconf("SC_PAGE_SIZE")


def faults():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_minflt, r.ru_majflt


def read_sys(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return "n/a"


# --------------------------------------------------------------------------- info
CUDA_ATTRS = [
    (18, "INTEGRATED", "GPU shares DRAM with CPU (1 on Spark)"),
    (88, "PAGEABLE_MEMORY_ACCESS", "GPU can access malloc/mmap memory (HMM or ATS)"),
    (100, "PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES", "1 = ATS: GPU walks the CPU page tables (hardware coherence)"),
    (89, "CONCURRENT_MANAGED_ACCESS", "CPU and GPU may touch managed memory concurrently"),
    (86, "HOST_NATIVE_ATOMIC_SUPPORTED", "atomics between CPU and GPU are native over the link"),
    (102, "VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED", "cuMemCreate / cuMemMap available"),
    (132, "MULTICAST_SUPPORTED", "cuMulticastCreate (NVLS multicast; needs NVSwitch)"),
    (116, "GPU_DIRECT_RDMA_SUPPORTED", "NIC may DMA to cudaMalloc memory"),
]


def cuda_attrs():
    try:
        cu = ctypes.CDLL("libcuda.so.1")
    except OSError:
        print("  libcuda.so.1 not found")
        return
    if cu.cuInit(0) != 0:
        print("  cuInit failed")
        return
    dev = ctypes.c_int()
    cu.cuDeviceGet(ctypes.byref(dev), 0)
    name = ctypes.create_string_buffer(128)
    cu.cuDeviceGetName(name, 128, dev)
    print(f"  device 0: {name.value.decode()}")
    for num, attr, meaning in CUDA_ATTRS:
        v = ctypes.c_int(-1)
        rc = cu.cuDeviceGetAttribute(ctypes.byref(v), num, dev)
        val = v.value if rc == 0 else f"err{rc}"
        print(f"  {attr:<46} {val!s:>4}   {meaning}")
    # VMM allocation granularity (what cuMemCreate rounds to)
    class Loc(ctypes.Structure):
        _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]

    class Prop(ctypes.Structure):
        _fields_ = [("type", ctypes.c_int), ("requestedHandleTypes", ctypes.c_int), ("location", Loc),
                    ("win32HandleMetaData", ctypes.c_void_p), ("compressionType", ctypes.c_ubyte),
                    ("gpuDirectRDMACapable", ctypes.c_ubyte), ("usage", ctypes.c_ushort),
                    ("reserved", ctypes.c_ubyte * 4)]
    p = Prop()
    p.type, p.location.type, p.location.id = 1, 1, dev.value  # PINNED, DEVICE
    g = ctypes.c_size_t()
    for flag, lab in ((0, "minimum"), (1, "recommended")):
        if cu.cuMemGetAllocationGranularity(ctypes.byref(g), ctypes.byref(p), flag) == 0:
            print(f"  cuMemCreate granularity ({lab}): {g.value >> 20} MiB")


def info():
    print(f"kernel {os.uname().release}  arch {os.uname().machine}")
    print(f"base page size: {PAGE} bytes ({PAGE >> 10} KiB)")
    print(f"THP enabled: {read_sys('/sys/kernel/mm/transparent_hugepage/enabled')}")
    print(f"THP defrag : {read_sys('/sys/kernel/mm/transparent_hugepage/defrag')}")
    print(f"THP PMD size: {int(read_sys('/sys/kernel/mm/transparent_hugepage/hpage_pmd_size') or 0) >> 20} MiB")
    mthp = "/sys/kernel/mm/transparent_hugepage"
    sizes = sorted(d for d in os.listdir(mthp) if d.startswith("hugepages-")) if os.path.isdir(mthp) else []
    if sizes:
        print("mTHP sizes:  " + ", ".join(f"{d[10:]}={read_sys(f'{mthp}/{d}/enabled').split('[')[1].split(']')[0]}"
                                          for d in sizes if '[' in read_sys(f'{mthp}/{d}/enabled')))
    for line in open("/proc/meminfo"):
        if line.split(":")[0] in ("MemTotal", "Cached", "AnonHugePages", "HugePages_Total", "Hugepagesize", "Mlocked"):
            print("  " + " ".join(line.split()))
    print("CUDA attributes:")
    cuda_attrs()


# --------------------------------------------------------------------------- checkpoint
def make(path, gb=2.0):
    import torch
    from safetensors.torch import save_file
    n_tensors = 16
    per = int(gb * (1 << 30) / n_tensors / 2)  # bf16
    tensors = {f"layer{i}.weight": torch.randn(per, dtype=torch.bfloat16) for i in range(n_tensors)}
    t0 = time.perf_counter()
    save_file(tensors, path)
    print(f"wrote {path}: {os.path.getsize(path) / 2**30:.2f} GiB in {time.perf_counter() - t0:.1f}s")


def evict(path):
    fd = os.open(path, os.O_RDONLY)
    os.fsync(fd) if os.access(path, os.W_OK) else None
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    print(f"evicted {path} from page cache (clean pages only; use 'sudo sh -c \"echo 1 > /proc/sys/vm/drop_caches\"' for all)")


def load(path, mode):
    size = os.path.getsize(path)
    f0 = faults()
    t0 = time.perf_counter()
    checksum = 0
    if mode == "read":
        with open(path, "rb", buffering=0) as f:
            buf = bytearray(size)
            f.readinto(buf)              # kernel copies page cache -> our private pages
        checksum = buf[::PAGE].count(0)
    elif mode == "mmap-rand":
        # MADV_RANDOM turns off readahead and fault-around: every page is its own fault,
        # and on a cold cache every fault is a major fault (a synchronous 4 KiB read)
        import random
        with open(path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            mm.madvise(mmap.MADV_RANDOM)
            n = min(size // PAGE, 65536)                 # sample 64k pages (256 MiB) to keep it short
            for pg in random.Random(0).sample(range(size // PAGE), n):
                checksum += mm[pg * PAGE]
            mm.close()
        size = n * PAGE
    elif mode == "mmap":
        with open(path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            mv = memoryview(mm)
            checksum = bytes(mv[::PAGE]).count(0)  # touch one byte per page -> one fault per page (or readahead batch)
            mv.release(); mm.close()
    elif mode == "st":
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                checksum += int(f.get_tensor(k).view(-1)[:: PAGE // 2].float().sum().item() != 0)
    else:
        sys.exit(f"unknown mode {mode}")
    dt = time.perf_counter() - t0
    f1 = faults()
    mi, ma = f1[0] - f0[0], f1[1] - f0[1]
    print(f"{mode:>4}: {size / 2**30:.2f} GiB in {dt:6.3f}s = {size / dt / 2**30:6.2f} GiB/s   "
          f"minor faults {mi:>8}   major faults {ma:>7}   pages in file {size // PAGE}")


# --------------------------------------------------------------------------- THP
def thp(gb=1.0):
    import random
    size = int(gb * (1 << 30))
    # Linux values (same on x86-64 and arm64); Python's mmap module doesn't export them
    hint = {"MADV_NOHUGEPAGE": 15, "MADV_HUGEPAGE": 14}
    try:
        import numpy as np
    except ImportError:
        np = None
    for name, adv in hint.items():
        mm = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        if adv is not None:
            mm.madvise(adv)
        f0 = faults()
        t0 = time.perf_counter()
        mv = memoryview(mm)
        # first touch: write one byte per 4 KiB so every base page is faulted in
        if np is not None:
            a = np.frombuffer(mm, dtype=np.uint8)
            a[::4096] = 1
        else:
            for off in range(0, size, 4096):
                mm[off] = 1
        t_touch = time.perf_counter() - t0
        f1 = faults()
        ahp = [l for l in open(f"/proc/{os.getpid()}/smaps_rollup") if l.startswith("AnonHugePages")]
        rand = ""
        if np is not None:
            a64 = np.frombuffer(mm, dtype=np.int64)
            idx = np.random.default_rng(0).integers(0, a64.size, 20_000_000)
            a64[idx].sum()  # warm
            t1 = time.perf_counter(); a64[idx].sum(); t_rand = time.perf_counter() - t1
            rand = f"random 8B reads: {t_rand / idx.size * 1e9:5.1f} ns each"
        print(f"{name:<16} first-touch {t_touch * 1e3:7.1f} ms  minor faults {f1[0] - f0[0]:>7}   "
              f"{ahp[0].split()[1] if ahp else '?':>8} kB in huge pages   {rand}")
        mv.release(); del mv
        if np is not None:
            del a
            try: del a64
            except NameError: pass
        mm.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "info": info()
    elif cmd == "make": make(args[0], float(args[1]) if len(args) > 1 else 2.0)
    elif cmd == "evict": evict(args[0])
    elif cmd == "load": load(args[0], args[1] if len(args) > 1 else "mmap")
    elif cmd == "thp": thp(float(args[0]) if args else 1.0)
    else: sys.exit(__doc__)
