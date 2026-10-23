#!/usr/bin/env bash
# Reproduces every experiment reported in RESULTS.md (CPU, 4 cores; ~hours). Run from the repo root.
# Data: LibriSpeech dev-clean (CC BY 4.0) from https://www.openslr.org/12
set -euo pipefail
LIBRI_RAW=${LIBRI_RAW:-/home/user/ravel_data/librispeech/LibriSpeech/dev-clean}
LIBRI=${LIBRI:-/home/user/ravel_data/libri24k}
export OMP_NUM_THREADS=2   # two jobs in parallel on 4 cores; oversubscription makes PyTorch CPU training ~30x slower

# 0. tests + Stage 0
python -m pytest -q tests
python scripts/check_gpu.py || true
python scripts/stage0_validate.py --device cpu --shapes 2x25,4x50,8x50

# 1. data
python scripts/prepare_librispeech.py --root "$LIBRI_RAW" --out "$LIBRI"
python training/make_sim_data.py --libri "$LIBRI" --split train --n 2000 --policy random_heuristic --out data/sim_train.pt
python training/make_sim_data.py --libri "$LIBRI" --split test  --n 200  --policy random_heuristic --out data/sim_test.pt
python training/make_sim_data.py --libri "$LIBRI" --split train --n 2000 --policy confounded --seed 5 --out data/sim_train_conf.pt

# 2. Stage 1: RAVEL-Tiny vs parameter-matched Baseline A (conventional streaming AR codec LM)
python training/train_acoustic.py --model ravel      --steps 400 --batch 8 --slots 50 --threads 2 --out checkpoints/stage1_ravel &
python training/train_acoustic.py --model baseline_a --steps 400 --batch 8 --slots 50 --threads 2 --out checkpoints/stage1_baseA &
wait

# 3. E1: RIF ablation (B = no RIF / behaviour cloning, Bcrit = cost-only critic, C = full RIF), 3 seeds
for s in 0 1 2; do
  for v in B Bcrit C; do
    python training/train_influence.py --variant $v --seed $s --steps 1000 --threads 2 \
      --train data/sim_train.pt --test data/sim_test.pt --out checkpoints/sim/e1_${v}_s${s} &
    while [ "$(jobs -r | wc -l)" -ge 2 ]; do sleep 5; done
  done
done
wait
python evaluation/ablations.py --runs checkpoints/sim --tag e1 --variants B Bcrit C --seeds 0 1 2 --n_eval 200 \
  --betas 0.0 0.5 --out results/e1_closed_loop.json

# 4. E3: confounded logging policy
for s in 0 1 2; do
  python training/train_influence.py --variant C --seed $s --steps 1000 --threads 2 \
    --train data/sim_train_conf.pt --test data/sim_test.pt --out checkpoints/sim/e3_C_s${s} &
  while [ "$(jobs -r | wc -l)" -ge 2 ]; do sleep 5; done
done
wait
python evaluation/ablations.py --runs checkpoints/sim --tag e3 --variants C --seeds 0 1 2 --n_eval 200 --no_refs \
  --out results/e3_confounded_closed_loop.json

# 5. E2: counterfactual accuracy of RIF predictions under intervention
python evaluation/run_counterfactual.py --ckpts checkpoints/sim/e1_C_s0/final.pt checkpoints/sim/e1_Bcrit_s0/final.pt \
  checkpoints/sim/e3_C_s0/final.pt --n_episodes 30 --out results/e2_counterfactual.json

# 6. latency
python evaluation/latency.py --device cpu --codec --out results/latency_cpu_tiny.json
python evaluation/latency.py --device cpu --micro --out results/latency_cpu_micro.json
