#!/usr/bin/env bash
# Reproduces every experiment reported in RESULTS.md (CPU, 4 cores: roughly 8-9 hours). Run from the repo root.
# Data: LibriSpeech dev-clean (CC BY 4.0, https://www.openslr.org/12) and FSDD (CC BY-SA 4.0,
# https://github.com/Jakobovski/free-spoken-digit-dataset).
set -euo pipefail
LIBRI_RAW=${LIBRI_RAW:-/home/user/ravel_data/librispeech/LibriSpeech/dev-clean}
LIBRI=${LIBRI:-/home/user/ravel_data/libri24k}
FSDD=${FSDD:-/home/user/ravel_data/fsdd_dl/fsdd/recordings}
export OMP_NUM_THREADS=1   # 1 thread per job, 4 jobs in parallel. Oversubscribing threads makes CPU training ~30x slower.

# 0. tests + Stage 0
python -m pytest -q tests
python scripts/check_gpu.py || true
OMP_NUM_THREADS=4 python scripts/stage0_validate.py --device cpu --shapes 2x25,4x50,8x50

# 1. data
python scripts/prepare_librispeech.py --root "$LIBRI_RAW" --out "$LIBRI"
python training/make_sim_data.py --libri "$LIBRI" --split test  --n 200  --policy random_heuristic --out data/sim_test.pt
python training/make_sim_data.py --libri "$LIBRI" --split train --n 2000 --policy random_heuristic --out data/sim_train.pt
python training/make_sim_data.py --libri "$LIBRI" --split train --n 2000 --policy confounded --seed 5 --out data/sim_train_conf.pt
python training/digit_qa.py prepare --fsdd "$FSDD" --out data/fsdd24k.pt
python training/digit_qa.py classifier --data data/fsdd24k.pt --out checkpoints/digit_clf.pt

# 2. all training runs: E1 (B / Bcrit / C x 3 seeds), E3 (C on confounded logs x 3), digit QA, Stage 1 (RAVEL vs Baseline A)
python scripts/job_queue.py --jobs scripts/jobs_main.txt --parallel 4

# 3. E1 closed-loop ablation (tunes theta/beta on validation episodes from training speakers, reports test)
python evaluation/ablations.py --threads 1 --tag e1 --variants B Bcrit C --seeds 0 1 2 --n_eval 200 --out results/e1_closed_loop.json
python evaluation/rate_analysis.py --ckpts checkpoints/sim/e1_C_s{0,1,2}/final.pt checkpoints/sim/e1_Bcrit_s0/final.pt --n 150 --out results/rate_analysis.json
python evaluation/planner_analysis.py --ckpt checkpoints/sim/e1_C_s0/final.pt --n 100 --out results/planner_analysis_C_s0.json

# 4. E2 interventional accuracy (logger-driven and model-driven states)
CK="checkpoints/sim/e1_C_s0/final.pt checkpoints/sim/e1_C_s1/final.pt checkpoints/sim/e1_C_s2/final.pt checkpoints/sim/e1_Bcrit_s0/final.pt checkpoints/sim/e1_Bcrit_s1/final.pt checkpoints/sim/e1_Bcrit_s2/final.pt"
python evaluation/run_counterfactual.py --threads 1 --R 32 --every 5 --n_episodes 40 --driver logger --ckpts $CK --out results/e2_counterfactual_logger.json
python evaluation/run_counterfactual.py --threads 1 --R 32 --every 5 --n_episodes 40 --driver model  --ckpts $CK --out results/e2_counterfactual_model.json

# 5. E3 confounded logs
python evaluation/ablations.py --threads 1 --no_refs --tag e3 --variants C --seeds 0 1 2 --n_eval 200 --out results/e3_closed_loop.json
python evaluation/run_counterfactual.py --threads 1 --R 32 --every 5 --n_episodes 40 --driver logger --logger confounded \
  --ckpts checkpoints/sim/e3_C_s0/final.pt checkpoints/sim/e3_C_s1/final.pt checkpoints/sim/e3_C_s2/final.pt --out results/e3_counterfactual_conf.json
python evaluation/rate_analysis.py --ckpts checkpoints/sim/e3_C_s0/final.pt --n 150 --out results/rate_analysis_e3.json

# 6. digit QA evaluation (closed loop, audio only, independent classifier)
python training/digit_qa.py score --ckpt checkpoints/digit_qa/final.pt --n 200 --out results/digit_qa_scoring.json
python training/digit_qa.py eval  --ckpt checkpoints/digit_qa/final.pt --n 250 --mode policy --out results/digit_qa_policy.json
python training/digit_qa.py eval  --ckpt checkpoints/digit_qa/final.pt --n 250 --mode forced --out results/digit_qa_forced.json

# 7. latency (run with the machine otherwise idle)
OMP_NUM_THREADS=4 python evaluation/latency.py --device cpu --threads 4 --codec --out results/latency_cpu_tiny_codec.json
OMP_NUM_THREADS=4 python evaluation/latency.py --device cpu --threads 4 --out results/latency_cpu_tiny_nocodec.json
OMP_NUM_THREADS=4 python evaluation/latency.py --device cpu --threads 4 --micro --codec --out results/latency_cpu_micro_codec.json

python scripts/collect_results.py > results/tables.md
