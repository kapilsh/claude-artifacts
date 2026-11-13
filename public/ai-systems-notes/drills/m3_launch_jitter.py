#!/usr/bin/env python3
"""Module 3 drill: CPU launch-thread jitter becomes GPU idle time.

Launches a long chain of tiny kernels (each a few µs on the GPU), so the GPU is only
busy while the CPU keeps up. Measures:
  * wall time per "step" of K kernels (eager launches)
  * the distribution of CPU time between consecutive launches (p50 / p99 / max)
  * the same step captured once as a CUDA graph and replayed

Run it quiet, then with CPU noise, then with the launch thread pinned away from the noise:

  python3 drills/m3_launch_jitter.py
  python3 drills/m3_launch_jitter.py --noise 32                       # 32 busy procs on all CPUs
  python3 drills/m3_launch_jitter.py --noise 31 --noise-cpus 1-31 --pin 0   # sibling CPU 1 still noisy
  python3 drills/m3_launch_jitter.py --noise 30 --noise-cpus 2-31 --pin 0   # whole core 0 quiet (x86: CPUs 0,1 are SMT siblings)
"""
import argparse
import multiprocessing as mp
import os
import statistics
import time

import torch


def burn(cpus):
    if cpus:
        os.sched_setaffinity(0, cpus)
    x = 0
    while True:
        x += 1


def parse_cpus(s):
    out = set()
    for part in s.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p / 100 * len(v)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", type=int, default=2000, help="kernels per step")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--noise", type=int, default=0, help="number of CPU-burning processes")
    ap.add_argument("--noise-cpus", default="", help="restrict noise to these CPUs, e.g. 1-31")
    ap.add_argument("--pin", default="", help="pin this (launch) process to these CPUs, e.g. 0")
    args = ap.parse_args()

    if args.pin:
        os.sched_setaffinity(0, parse_cpus(args.pin))
    noise = [mp.Process(target=burn, args=(parse_cpus(args.noise_cpus) if args.noise_cpus else None,), daemon=True)
             for _ in range(args.noise)]
    for p in noise:
        p.start()

    dev = torch.device("cuda")
    x = torch.zeros(1024, device=dev)

    def step():
        for _ in range(args.kernels):
            x.add_(1.0)            # one tiny kernel; GPU time ~2-3 µs, CPU launch cost ~5 µs+

    for _ in range(3):
        step()
    torch.cuda.synchronize()

    # --- eager: wall time per step + per-launch CPU gaps
    walls, gaps = [], []
    for _ in range(args.steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        last = t0
        for _ in range(args.kernels):
            x.add_(1.0)
            now = time.perf_counter_ns()
            gaps.append(now - last)
            last = now
        torch.cuda.synchronize()
        walls.append((time.perf_counter_ns() - t0) / 1e6)

    # GPU-only time for the same work (events around a step)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # --- CUDA graph: capture once, replay with one launch
    g = torch.cuda.CUDAGraph()
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        step()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        step()
    g.replay(); torch.cuda.synchronize()
    gwalls = []
    for _ in range(args.steps):
        t0 = time.perf_counter_ns()
        g.replay()
        torch.cuda.synchronize()
        gwalls.append((time.perf_counter_ns() - t0) / 1e6)
    s.record(); g.replay(); e.record(); e.synchronize()
    gpu_ms = s.elapsed_time(e)

    for p in noise:
        p.terminate()

    aff = sorted(os.sched_getaffinity(0))
    print(f"launch thread CPUs: {aff[0]}-{aff[-1]} ({len(aff)})   noise procs: {args.noise} on {args.noise_cpus or 'all CPUs'}")
    print(f"{args.kernels} kernels/step, {args.steps} steps")
    print(f"  GPU busy time per step (graph, back-to-back): {gpu_ms:7.2f} ms")
    print(f"  eager step wall time   median {statistics.median(walls):7.2f} ms   worst {max(walls):7.2f} ms"
          f"   -> GPU idle ≈ {100 * (1 - gpu_ms / statistics.median(walls)):.0f}% of the step")
    print(f"  CPU gap between launches  p50 {pct(gaps, 50) / 1e3:6.1f} µs   p99 {pct(gaps, 99) / 1e3:7.1f} µs"
          f"   p99.9 {pct(gaps, 99.9) / 1e3:8.1f} µs   max {max(gaps) / 1e3:8.1f} µs")
    print(f"  CUDA graph step wall    median {statistics.median(gwalls):7.2f} ms   worst {max(gwalls):7.2f} ms")


if __name__ == "__main__":
    main()
