#!/usr/bin/env python3
"""
RCCL Bandwidth Test via PyTorch Distributed (NCCL/RCCL backend).

Measures all-reduce bandwidth across GPUs, similar to rccl-tests all_reduce_perf.
Ring all-reduce: total bytes = 2 * (n-1) * size * 4 bytes (float32).
Bandwidth (GB/s) = total_bytes / time / 1e9
"""
import os
import time
import torch
import torch.distributed as dist

# Message sizes in elements (float32 = 4 bytes). Similar to nccl-tests: -b 8 -e 10G -f 2
SIZES = [
    8, 64, 256, 1024, 4096, 8192, 16384, 32768, 65536,
    131072, 262144, 524288, 1048576, 4194304, 16777216,
    67108864, 134217728, 268435456,
]
WARMUP_ITERS = 5
BENCH_ITERS = 20


def run_allreduce(size: int, rank: int, local_rank: int, world_size: int) -> float:
    """Run one all-reduce and return elapsed time in seconds."""
    t = torch.ones(size, device=f"cuda:{local_rank}", dtype=torch.float32) * (rank + 1)
    torch.cuda.synchronize()
    start = time.perf_counter()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", 8))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    if rank == 0:
        print(f"# RCCL Bandwidth Test (PyTorch NCCL/RCCL backend, {world_size} GPUs)")
        print(f"# Ring all-reduce: total_bytes = 2*(n-1)*size*4, BW = total_bytes/time")
        print(f"{'Size (bytes)':>14} {'Size (elements)':>16} {'Algbw (GB/s)':>14} {'BusBW (GB/s)':>14} {'Time (ms)':>12}")
        print("-" * 72)

    for size in SIZES:
        # Warmup
        for _ in range(WARMUP_ITERS):
            run_allreduce(size, rank, local_rank, world_size)

        times = []
        for _ in range(BENCH_ITERS):
            elapsed = run_allreduce(size, rank, local_rank, world_size)
            times.append(elapsed)

        if rank == 0:
            avg_time = sum(times) / len(times)
            bytes_per_elem = 4
            total_bytes = 2 * (world_size - 1) * size * bytes_per_elem
            alg_bw = total_bytes / avg_time / 1e9
            bus_bw = alg_bw * world_size
            size_bytes = size * bytes_per_elem
            print(f"{size_bytes:>14} {size:>16} {alg_bw:>14.2f} {bus_bw:>14.2f} {avg_time*1000:>12.2f}")

    if rank == 0:
        print("-" * 72)
        print("# Test complete")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
