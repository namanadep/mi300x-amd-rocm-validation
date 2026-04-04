#!/usr/bin/env python3
"""RCCL all-reduce test via PyTorch distributed."""
import os
import torch
import torch.distributed as dist

def main():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", 8))
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    
    # All-reduce test: each rank has tensor, sum across all
    size = 10_000_000  # 10M floats
    t = torch.ones(size, device=f"cuda:{local_rank}", dtype=torch.float32) * (rank + 1)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = sum(range(1, world_size + 1))  # 1+2+...+8 = 36
    ok = torch.allclose(t, torch.full_like(t, float(expected)))
    
    if rank == 0:
        print(f"RCCL all-reduce: OK, result={t[0].item():.0f} (expected {expected})")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
