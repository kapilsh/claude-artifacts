#!/usr/bin/env python3
"""Module 5 drill: allocate a buffer, first-touch it from this CPU, and report which NUMA
node(s) its pages landed on, by reading /proc/self/numa_maps.

On a GB200/GB300 tray in NUMA mode, the default policy puts pages on the touching CPU's
node, but under memory pressure (or with numactl --preferred/--membind) they can land on a
GPU's HBM node. In CDMM mode the GPU nodes have no memory, so that can't happen.

Usage: python3 drills/m5_numa_place.py [--mib 2048] [--quiet]
"""
import argparse, mmap, os, re, time

ap = argparse.ArgumentParser()
ap.add_argument("--mib", type=int, default=2048)
ap.add_argument("--quiet", action="store_true")
a = ap.parse_args()

size = a.mib << 20
buf = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
page = os.sysconf("SC_PAGE_SIZE")
t = time.perf_counter()
for off in range(0, size, page):          # first touch: one write per page
    buf[off] = 1
t = time.perf_counter() - t

addr = None
with open("/proc/self/maps") as f:       # find our mapping by its size
    for line in f:
        lo, hi = (int(x, 16) for x in line.split()[0].split("-"))
        if hi - lo == size:
            addr = lo
nodes = {}
with open("/proc/self/numa_maps") as f:
    for line in f:
        if addr is not None and int(line.split()[0], 16) == addr:
            for n, pages in re.findall(r"\bN(\d+)=(\d+)", line):
                nodes[int(n)] = int(pages)
            policy = line.split()[1]
cpu = os.sched_getaffinity(0)
total = sum(nodes.values()) or 1
desc = ", ".join(f"node {n}: {p * page / 2**20:,.0f} MiB ({100 * p / total:.0f}%)" for n, p in sorted(nodes.items()))
if not a.quiet:
    print(f"  {a.mib} MiB anonymous buffer, first-touched by this process (allowed CPUs: {len(cpu)})")
    print(f"  page size {page} B, first touch {t:.2f} s ({size / page / t / 1e3:,.0f}k pages/s)")
print(f"  policy={policy if nodes else '?'}  ->  {desc or 'numa_maps not readable'}")
