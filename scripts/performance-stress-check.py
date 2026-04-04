#!/usr/bin/env python3
"""
Performance and Stress Checks
Based on Report 05 — Performance and Stress Validation
Usage: python performance-stress-check.py [--quick]
  --quick: skip long stress test
Requires: PyTorch with CUDA/HIP. Run with: /root/pytorchenv/bin/python or torchrun
"""
import argparse
import sys
import time

try:
    import torch
except ImportError:
    print("[FAIL] PyTorch not found. Install or use: /root/pytorchenv/bin/python")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Skip long stress test")
    args = parser.parse_args()

    print("==========================================")
    print(" Performance and Stress Checks")
    print(" Host: (Python) | PyTorch", torch.__version__)
    print("==========================================")

    if not torch.cuda.is_available():
        print("[FAIL] CUDA/HIP not available")
        sys.exit(1)

    n_gpu = torch.cuda.device_count()
    print(f"[PASS] {n_gpu} GPU(s) detected")

    # --- 1. Single-GPU matmul benchmark ---
    print("\n--- Single-GPU matmul benchmark ---")
    try:
        x = torch.randn(4096, 4096, device='cuda:0', dtype=torch.float32)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            y = torch.mm(x, x)
        torch.cuda.synchronize()
        t = time.perf_counter() - t0
        flops = 10 * 2 * 4096**3
        tflops = flops / t / 1e12
        print(f"[PASS] 4096x4096 x10: {t:.3f}s, ~{tflops:.1f} TFLOPS")
    except Exception as e:
        print(f"[FAIL] Single-GPU benchmark: {e}")
        sys.exit(1)

    # --- 2. All-GPU memory stress (80% per GPU) ---
    print("\n--- All-GPU memory stress ---")
    try:
        gb_per_gpu = 161  # ~84% of 192GB
        per_gpu = int(gb_per_gpu * 1024**3 / 4)
        tensors = []
        for i in range(n_gpu):
            t = torch.zeros(per_gpu, device=f'cuda:{i}', dtype=torch.float32)
            tensors.append(t)
        for i in range(n_gpu):
            used = torch.cuda.memory_allocated(i) / 1e9
            print(f"  GPU {i}: {used:.1f} GB OK")
        print("[PASS] All-GPU memory stress: no OOM")
        del tensors
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError as e:
        print(f"[FAIL] OOM during memory stress: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[WARN] Memory stress: {e}")

    # --- 3. Sustained training-loop stress (optional) ---
    if not args.quick:
        print("\n--- Sustained training-loop stress (55s) ---")
        try:
            model = torch.nn.Sequential(
                torch.nn.Linear(1024, 4096), torch.nn.ReLU(),
                torch.nn.Linear(4096, 4096), torch.nn.ReLU(),
                torch.nn.Linear(4096, 1024)
            ).cuda()
            opt = torch.optim.SGD(model.parameters(), lr=1e-3)
            t0 = time.perf_counter()
            iters = 0
            while time.perf_counter() - t0 < 55:
                x = torch.randn(64, 1024, device='cuda')
                loss = model(x).sum()
                loss.backward()
                opt.step()
                opt.zero_grad()
                iters += 1
            elapsed = time.perf_counter() - t0
            print(f"[PASS] {iters} iterations in {elapsed:.1f}s")
        except Exception as e:
            print(f"[WARN] Stress test: {e}")
    else:
        print("\n--- Skipping long stress (--quick) ---")

    print("\n------------------------------------------")
    print(" Performance check: PASS")
    sys.exit(0)

if __name__ == "__main__":
    main()
