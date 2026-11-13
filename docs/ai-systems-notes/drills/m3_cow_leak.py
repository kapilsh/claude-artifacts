#!/usr/bin/env python3
"""Module 3 drill: the DataLoader "memory leak" is copy-on-write + refcounts.

A Dataset holds N Python objects (strings in a list). With fork, workers share the
parent's pages copy-on-write. Merely *reading* an item writes its refcount (and the GC
header), which dirties the 4 KiB page it lives on, so the kernel copies the page into
the worker. Over an epoch every worker ends up with a private copy of the whole list.

The same data stored as one numpy buffer has a single object header, so reads don't
dirty data pages and workers keep sharing.

    python3 drills/m3_cow_leak.py --storage list    # private memory grows per worker
    python3 drills/m3_cow_leak.py --storage numpy   # stays flat

Columns: RSS = resident pages (shared counted fully), PSS = shared pages divided among
sharers, USS = Private_Clean+Private_Dirty (memory only this worker holds).
"""
import argparse
import os
import threading
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ListDataset(Dataset):
    def __init__(self, n):
        # many small Python objects -> refcounts sprinkled over every page
        self.items = [f"sample-{i:09d}-" + "x" * 40 for i in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s = self.items[i]          # INCREF on the str object -> page write -> COW copy
        return len(s)


class NumpyDataset(Dataset):
    def __init__(self, n):
        # same bytes in one contiguous array: one object header, data pages never written
        self.items = np.frombuffer(b"".join(f"sample-{i:09d}-".encode() + b"x" * 40 for i in range(n)),
                                   dtype=np.uint8).reshape(n, -1).copy()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return int(self.items[i, 0])


def mem_kb(pid):
    d = {}
    try:
        for line in open(f"/proc/{pid}/smaps_rollup"):
            parts = line.split()
            if len(parts) == 3 and parts[2] == "kB":   # skip the header line
                d[parts[0].rstrip(":")] = int(parts[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    return d.get("Rss", 0), d.get("Pss", 0), d.get("Private_Clean", 0) + d.get("Private_Dirty", 0)


def children(pid):
    out = []
    for tid in os.listdir(f"/proc/{pid}/task"):
        try:
            out += [int(x) for x in open(f"/proc/{pid}/task/{tid}/children").read().split()]
        except OSError:
            pass
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage", choices=["list", "numpy"], default="list")
    ap.add_argument("--n", type=int, default=4_000_000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--context", default="fork", help="fork | forkserver | spawn")
    args = ap.parse_args()

    t0 = time.time()
    ds = ListDataset(args.n) if args.storage == "list" else NumpyDataset(args.n)
    rss, pss, uss = mem_kb(os.getpid())
    print(f"storage={args.storage} n={args.n:,} context={args.context}  parent RSS after build: {rss / 1024:.0f} MiB "
          f"({time.time() - t0:.1f}s)")

    loader = DataLoader(ds, batch_size=1024, shuffle=True, num_workers=args.workers,
                        multiprocessing_context=args.context, persistent_workers=True)
    samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            kids = children(os.getpid())
            vals = [m for m in (mem_kb(k) for k in kids) if m]
            if vals:
                samples.append((time.time(), vals))
            time.sleep(0.25)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    batches = len(loader)
    print(f"{'progress':>9} | {'per-worker RSS':>14} {'PSS':>8} {'USS (private)':>14} | {'sum of worker USS':>17}")
    shown = set()
    for b, _ in enumerate(loader):
        pct = int(100 * (b + 1) / batches)
        if pct // 20 not in shown and samples:
            shown.add(pct // 20)
            _, vals = samples[-1]
            r = sum(v[0] for v in vals) / len(vals) / 1024
            p = sum(v[1] for v in vals) / len(vals) / 1024
            u = sum(v[2] for v in vals) / len(vals) / 1024
            print(f"{pct:>8}% | {r:>11.0f} MiB {p:>5.0f} MiB {u:>11.0f} MiB | {u * len(vals):>13.0f} MiB")
    stop.set()
    th.join()
    print("\nRSS stays ~flat in both cases (pages were already resident via sharing).")
    print("USS is the real cost: with --storage list each worker copies every page it reads.")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
