#!/usr/bin/env python3
"""Module 1 drill: pinned vs pageable host<->device copies.

Measures, for several buffer sizes:
  * H2D / D2H bandwidth from pageable memory (torch.empty on CPU)
  * H2D / D2H bandwidth from pinned memory (pin_memory=True -> cudaHostAlloc)
  * how long the CPU thread is blocked inside a non_blocking=True copy
    (the "is it really async?" test)

Run it plain, or under taskset/numactl to see how CPU placement changes
the pageable path (the CPU does a memcpy into a staging buffer) but barely
touches the pinned path (the GPU copy engine does all the work by DMA).

    python3 drills/m1_h2d_bandwidth.py
    taskset -c 0   python3 drills/m1_h2d_bandwidth.py --sizes 256M
    taskset -c 16  python3 drills/m1_h2d_bandwidth.py --sizes 256M
"""
import argparse
import os
import time

import torch


def parse_size(s: str) -> int:
    s = s.strip().upper()
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    if s[-1] in mult:
        return int(float(s[:-1]) * mult[s[-1]])
    return int(s)


def fmt_size(n: int) -> str:
    for unit, div in (("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if n >= div:
            return f"{n / div:g}{unit}"
    return f"{n}B"


def time_copy(dst, src, iters: int, non_blocking: bool) -> float:
    """Return seconds per copy, measured with CUDA events around the whole loop."""
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dst.copy_(src, non_blocking=non_blocking)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1e3 / iters


def cpu_blocked_time(dst, src) -> float:
    """Seconds the calling CPU thread spends inside one non_blocking copy call."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    dst.copy_(src, non_blocking=True)
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return t1 - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1M,16M,256M,1G",
                    help="comma-separated buffer sizes (K/M/G suffix)")
    ap.add_argument("--iters", type=int, default=0,
                    help="copies per measurement (0 = auto, ~4 GB moved per size)")
    args = ap.parse_args()

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    try:
        affinity = sorted(os.sched_getaffinity(0))
        aff = f"{affinity[0]}-{affinity[-1]}" if len(affinity) > 1 and \
            affinity == list(range(affinity[0], affinity[-1] + 1)) else ",".join(map(str, affinity))
    except AttributeError:
        aff = "n/a"
    print(f"GPU: {props.name}   torch {torch.__version__}   CUDA {torch.version.cuda}")
    print(f"integrated (shares DRAM with CPU): {bool(getattr(props, 'is_integrated', False))}")
    print(f"this process may run on CPUs: {aff}")
    print()
    hdr = f"{'size':>6} | {'H2D pageable':>13} {'H2D pinned':>11} | {'D2H pageable':>13} {'D2H pinned':>11} | {'CPU blocked in non_blocking H2D':>32}"
    print(hdr)
    print(f"{'':>6} | {'GB/s':>13} {'GB/s':>11} | {'GB/s':>13} {'GB/s':>11} | {'pageable':>15} {'pinned':>16}")
    print("-" * len(hdr))

    for size in [parse_size(s) for s in args.sizes.split(",")]:
        n = size // 4
        iters = args.iters or max(3, min(200, (4 << 30) // size))
        pageable = torch.ones(n, dtype=torch.float32)            # malloc'd, pageable
        pinned = torch.ones(n, dtype=torch.float32, pin_memory=True)  # cudaHostAlloc
        gpu = torch.empty(n, dtype=torch.float32, device=dev)

        # warm-up: first touch, driver staging buffers, lazy init
        for src in (pageable, pinned):
            gpu.copy_(src); src.copy_(gpu)
        torch.cuda.synchronize()

        gbps = lambda t: size / t / 1e9
        h2d_pg = gbps(time_copy(gpu, pageable, iters, non_blocking=True))
        h2d_pn = gbps(time_copy(gpu, pinned, iters, non_blocking=True))
        d2h_pg = gbps(time_copy(pageable, gpu, iters, non_blocking=True))
        d2h_pn = gbps(time_copy(pinned, gpu, iters, non_blocking=True))

        blk_pg = min(cpu_blocked_time(gpu, pageable) for _ in range(3))
        blk_pn = min(cpu_blocked_time(gpu, pinned) for _ in range(3))
        dma_time = size / (h2d_pn * 1e9)

        def blk(t):
            return f"{t * 1e3:8.3f} ms ({100 * t / dma_time:3.0f}%)"

        print(f"{fmt_size(size):>6} | {h2d_pg:13.1f} {h2d_pn:11.1f} | {d2h_pg:13.1f} {d2h_pn:11.1f} | {blk(blk_pg):>15} {blk(blk_pn):>16}")
        del pageable, pinned, gpu
        torch.cuda.empty_cache()

    print()
    print("CPU-blocked % is relative to the pinned copy time for that size.")
    print("~100%  => the call held the CPU thread for the whole transfer (not really async).")
    print("~0%    => the call only enqueued work; the copy engine moved the bytes by DMA.")


if __name__ == "__main__":
    main()
