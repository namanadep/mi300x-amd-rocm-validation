# Peak Load Summary — 8× MI300X VF

**Host:** dnd-amd-8gpu-venu-19mar  
**Date:** March 19 2026  
**Test:** 8× concurrent 8192×8192 FP32 matmul, 50-second sustained, via threading

---

## GPU Utilization (during stress, ~15 s in)

| GPU | Utilization | Junction (°C) | Memory (°C) | Power (W) |
|-----|-------------|---------------|-------------|-----------|
| 0 | 99% | 86 | 49 | 746 |
| 1 | 99% | 72 | 44 | 749 |
| 2 | 99% | 74 | 44 | 745 |
| 3 | 99% | 83 | 47 | 750 |
| 4 | 99% | 83 | 48 | 750 |
| 5 | 100% | 80 | 43 | 750 |
| 6 | 99% | 81 | 48 | 748 |
| 7 | 99% | 81 | 45 | 749 |

**Total system power: ~5.99 kW (8 GPUs × ~750 W)**

---

## Thermal Throttle Status (GPU 0, representative)

| Counter | Value | Meaning |
|---------|-------|---------|
| prochot_residency_acc | 0 | No CPU thermal throttle |
| socket_thm_residency_acc | 0 | No socket thermal throttle |
| vr_thm_residency_acc | 0 | No VR throttle |
| hbm_thm_residency_acc | 0 | No HBM thermal throttle |
| ppt_residency_acc | 53994 | Package power limit hit (expected at 750W cap) |

**All critical throttle counters = 0. System ran at full power cap without frequency reduction.**

---

## Concurrent 8-GPU per-GPU timing

| GPU | 100× 8192² matmul (s) |
|-----|----------------------|
| 0 | 0.65 |
| 1 | 0.62 |
| 2 | 0.63 |
| 3 | 0.64 |
| 4 | 0.65 |
| 5 | 0.66 |
| 6 | 0.65 |
| 7 | 0.70 |
| **Wall time** | **1.6 s** |

Coefficient of variation: ~4%. Confirms symmetric XGMI bandwidth across all GPU pairs.

---

## Memory Stress (161 GB per GPU)

| GPU | Allocated (GB) | % of 192 GB | OOM? |
|-----|----------------|-------------|------|
| 0–7 | 161.1 each | 84% | No |
| **Total** | **1,288.8 GB (~1.29 TB)** | 84% | **No OOM** |
