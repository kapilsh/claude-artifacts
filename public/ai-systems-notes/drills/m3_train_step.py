#!/usr/bin/env python3
"""Module 3 drill: a small but realistic training loop to profile with strace / perf trace.

  strace -f -c -o /tmp/strace.txt python3 drills/m3_train_step.py ; sort -k2 -nr /tmp/strace.txt | head -25
  sudo perf trace -s python3 drills/m3_train_step.py 2>&1 | tail -60
  nsys profile --trace=cuda,nvtx,osrt -o /tmp/m3 python3 drills/m3_train_step.py

What to expect: futex dominates time (threads waiting for each other: pin-memory thread,
DataLoader result queue, CUDA driver callbacks), ioctl is the CUDA driver talking to
/dev/nvidia*, read/write/poll/sendmsg come from worker <-> main-process queues, and
mmap/munmap/madvise come from allocators.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class Synthetic(Dataset):
    def __len__(self):
        return 64 * 64

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        return torch.randn(3, 64, 64, generator=g), i % 10


def main():
    dev = torch.device("cuda")
    model = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10)).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = DataLoader(Synthetic(), batch_size=64, num_workers=4, pin_memory=True)
    for step, (x, y) in enumerate(loader):
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    print(f"done: {step + 1} steps, last loss {loss.item():.3f}")


if __name__ == "__main__":
    main()
