# AMD ROCm validation image — mirrors the dnd-amd-8gpu-venu-19mar environment
FROM rocm/pytorch:rocm6.4_ubuntu24.04_py3.12_pytorch_release_2.4.1

WORKDIR /workspace

COPY scripts/ scripts/
COPY configs/ configs/
COPY pyproject.toml .

RUN pip install -e ".[dev]"

# Default: health check (no GPU required for syntax validation)
CMD ["bash", "scripts/gpu-health-check.sh"]
