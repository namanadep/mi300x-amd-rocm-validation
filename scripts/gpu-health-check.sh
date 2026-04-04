#!/bin/bash
#
# GPU/ROCm/PyTorch Health Check Script
# 8-GPU AMD Ubuntu VM — Quick post-deploy verification
# Usage: ./gpu-health-check.sh [--pytorch]  (--pytorch requires sudo for /root/pytorchenv)
#

set -e
PASS=0
FAIL=0
WARN=0
PYTORCH_CHECK=0
[[ "$1" == "--pytorch" ]] && PYTORCH_CHECK=1

echo "=========================================="
echo " GPU/ROCm/PyTorch Health Check"
echo " Host: $(hostname) | $(date -Iseconds)"
echo "=========================================="

# --- 1. rocm-smi (GPUs) ---
if command -v rocm-smi &>/dev/null; then
    GPU_COUNT=$(rocm-smi --showproductname 2>/dev/null | grep -oE 'GPU\[[0-9]+\]' | sort -u | wc -l)
    GPU_COUNT=${GPU_COUNT:-0}
    if [[ "$GPU_COUNT" -ge 8 ]]; then
        echo "[PASS] rocm-smi: $GPU_COUNT GPUs detected"
        PASS=$((PASS+1))
    elif [[ "$GPU_COUNT" -gt 0 ]]; then
        echo "[WARN] rocm-smi: $GPU_COUNT GPUs (expected 8)"
        WARN=$((WARN+1))
    else
        echo "[FAIL] rocm-smi: No GPUs or error"
        FAIL=$((FAIL+1))
    fi
else
    echo "[FAIL] rocm-smi: not found"
    FAIL=$((FAIL+1))
fi

# --- 2. /dev/kfd ---
if [[ -c /dev/kfd ]]; then
    echo "[PASS] /dev/kfd exists"
    PASS=$((PASS+1))
else
    echo "[FAIL] /dev/kfd not found"
    FAIL=$((FAIL+1))
fi

# --- 3. render group (current user) ---
if groups | grep -q render; then
    echo "[PASS] User in render group (ROCm access OK)"
    PASS=$((PASS+1))
else
    echo "[WARN] User not in render group (rocminfo/HIP may fail)"
    WARN=$((WARN+1))
fi

# --- 4. amd-metrics-exporter ---
if systemctl is-active --quiet amd-metrics-exporter 2>/dev/null; then
    echo "[PASS] amd-metrics-exporter: active"
    PASS=$((PASS+1))
else
    echo "[WARN] amd-metrics-exporter: not active"
    WARN=$((WARN+1))
fi

# --- 5. Metrics endpoint ---
if curl -sf --connect-timeout 2 http://localhost:9400/metrics >/dev/null 2>&1; then
    echo "[PASS] Metrics endpoint localhost:9400/metrics reachable"
    PASS=$((PASS+1))
else
    echo "[WARN] Metrics endpoint localhost:9400/metrics not reachable"
    WARN=$((WARN+1))
fi

# --- 6. ROCm version ---
if [[ -f /opt/rocm-6.4.1/.info/version ]]; then
    ROCM_VER=$(cat /opt/rocm-6.4.1/.info/version 2>/dev/null || echo "?")
    echo "[PASS] ROCm version: $ROCM_VER"
    PASS=$((PASS+1))
else
    echo "[WARN] ROCm version file not found"
    WARN=$((WARN+1))
fi

# --- 7. rocminfo (requires render group) ---
if rocminfo 2>/dev/null | grep -q "Agent 1"; then
    echo "[PASS] rocminfo: HSA agents detected"
    PASS=$((PASS+1))
else
    echo "[WARN] rocminfo: failed or no agents (check render group)"
    WARN=$((WARN+1))
fi

# --- 8. PyTorch (optional, needs root/sudo for /root/pytorchenv) ---
if [[ "$PYTORCH_CHECK" -eq 1 ]]; then
    PYTORCH_PY="/root/pytorchenv/bin/python"
    if sudo test -x "$PYTORCH_PY" 2>/dev/null && sudo -n true 2>/dev/null; then
        OUT=$(sudo "$PYTORCH_PY" -c "
import torch
cuda = torch.cuda.is_available()
count = torch.cuda.device_count() if cuda else 0
print('OK' if cuda and count >= 8 else 'FAIL', count)
" 2>/dev/null)
        if [[ "$OUT" =~ OK ]]; then
            echo "[PASS] PyTorch: CUDA available, $OUT"
            PASS=$((PASS+1))
        else
            echo "[FAIL] PyTorch: $OUT"
            FAIL=$((FAIL+1))
        fi
    else
        echo "[WARN] PyTorch: skip (need sudo + /root/pytorchenv)"
        WARN=$((WARN+1))
    fi
fi

# --- Summary ---
echo "------------------------------------------"
echo " Summary: $PASS pass, $WARN warnings, $FAIL failures"
if [[ $FAIL -gt 0 ]]; then
    echo " Overall: FAIL"
    exit 1
elif [[ $WARN -gt 0 ]]; then
    echo " Overall: PASS (with warnings)"
    exit 0
else
    echo " Overall: PASS"
    exit 0
fi
