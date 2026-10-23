# RAVEL V2.1 — measured results

Everything here was measured in this repository on a cloud container with **4 CPU cores, 15 GB RAM, no GPU**
(PyTorch 2.9.0+cpu, Python 3.13). The target GPU (RTX 5070, 12 GB) was not available, so **no GPU, VRAM or
GPU-latency claim is made**. Raw outputs are in `results/` and training logs in `results/train_logs/`. Commands are in
`scripts/run_experiments.sh` and `scripts/jobs_main.txt`.

Because of the CPU budget, the ablations use **RAVEL-Micro** (same architecture, smaller widths, ~8 M parameters,
`ravel/config.py:micro_overrides`). Only Stage 0, Stage 1 and the latency benchmark use RAVEL-Tiny (51.2 M).

__RESULTS_BODY__
