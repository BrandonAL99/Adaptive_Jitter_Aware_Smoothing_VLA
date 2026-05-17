# VLA Jitter Smoothing — AJAS for SmolVLA

Thesis project: *Adaptive Jitter-Aware Action Smoothing (AJAS) for Vision-Language-Action models on a 6-DOF robot arm.*

SmolVLA and similar VLA models generate action chunks that exhibit high-frequency jitter — rapid oscillations that are physically unnecessary and reduce task success. This project proposes a smoothing approach inspired by the pure-pursuit controller used in mobile robotics: in high-jitter regions the lookahead distance is increased (skipping noisy waypoints), while in smooth regions it is kept small (preserving intended motion).

---

## Contribution summary

| Component | Description |
|-----------|-------------|
| **Temporal consistency metric (Sm)** | Weighted mean frequency of the action FFT — measures jitter without requiring ground truth. Lower = smoother. |
| **AJAS (Adaptive Jitter-Aware Action Smoothing)** | Chunk-level processor: slides a window over the VLA action chunk, computes local Sm, and uses it to set an adaptive keyframe stride. Keyframes are connected with a cubic Hermite spline (C1 continuous, no overshoot). |
| **Low-pass baseline** | Causal 2nd-order Butterworth IIR filter applied per action step. Conventional comparison baseline. |
| **Evaluation framework** | Scripts for LIBERO simulation and SO101 real-robot evaluation, with automated TC metric computation and HuggingFace dataset upload. |

---

## Repository structure

```
vla_jitter_smoothing/
├── libero_analysis/         # Core contribution: metrics + processors
│   ├── processors/          # AJAS, LP filter, identity, pure-pursuit
│   ├── scripts/             # Analysis, tuning, and comparison plots
│   ├── temporal_consistency.py   # Sm and SER metric definitions
│   └── compute_metrics_all.py    # Batch metric computation
│
├── simulation/              # Custom MuJoCo SO101 environment
│   ├── envs/                # pick_place, stacking, sorting tasks
│   ├── mjcf/                # MuJoCo XML scenes
│   └── scripts/
│       ├── eval_libero_processed.py  # Main LIBERO eval with processor
│       └── eval_libero.py            # Original LIBERO eval (no processor)
│
├── real_robot/              # SO101 physical arm scripts
│   ├── config.py            # Arm + camera configuration (set your serials)
│   ├── eval_real.py         # Real-robot evaluation (one episode per trigger)
│   ├── teleop.py            # Teleoperation interface
│   └── collect_dataset.py   # Dataset collection pipeline
│
├── training/                # Training scripts
│   ├── train.py             # Fine-tune SmolVLA on real robot dataset
│   └── train_sim.py         # Fine-tune SmolVLA on simulation dataset
│
├── dataset_scripts/         # Dataset utilities (merge, rename cameras, etc.)
├── eval_scripts/            # Shell scripts for running evaluations
├── figures/                 # Report figures (processor comparisons, metric plots)
└── results/
    └── training_runs_summary.md
```

---

## Quick start

### 1. Install dependencies

```bash
# Clone lerobot (required by all scripts)
git clone https://github.com/huggingface/lerobot.git
pip install -e "lerobot[smolvla]"

# Python dependencies for this repo
pip install -r requirements.txt
```

### 2. Run LIBERO simulation evaluation

```bash
# Baseline (no smoothing)
python simulation/scripts/eval_libero_processed.py \
    --model BrandonAL/my_smolvla_all_1505 \
    --task libero_object --episodes 10 \
    --processor identity \
    --output-dir results/baseline

# AJAS Spline (C=100, main method)
python simulation/scripts/eval_libero_processed.py \
    --model BrandonAL/my_smolvla_all_1505 \
    --task libero_object --episodes 10 \
    --processor ajas_spline --spline-c 100 \
    --output-dir results/ajas_spline

# 4 Hz low-pass filter
python simulation/scripts/eval_libero_processed.py \
    --model BrandonAL/my_smolvla_all_1505 \
    --task libero_object --episodes 10 \
    --processor lowpass --lowpass-cutoff 4.0 \
    --output-dir results/lp4hz
```

On a headless GPU add `MUJOCO_GL=osmesa` before the command.

### 3. Run real-robot evaluation (SO101 arm)

Edit the three constant blocks at the top of `real_robot/eval_real.py` to select processor, task, and dataset name. Then:

```bash
cd /path/to/vla_jitter_smoothing
HF_TOKEN=hf_... conda run -n lerobot python real_robot/eval_real.py
```

Place cubes and press Enter to trigger each episode. Results push to HuggingFace after every episode.

---

## Key results (LIBERO simulation, libero_object suite)

| Method | Success Rate | Sm (↓ better) | SER (↑ better) |
|--------|-------------|--------------|----------------|
| Baseline (no smoothing) | 54% | 0.01463 | 0.853 |
| Low-pass 4 Hz | — | — | — |
| AJAS Spline C=100 | 50% | 0.00721 | 0.913 |

*Real-robot results pending — see `results/training_runs_summary.md`.*

---

## Processors

All processors live in `libero_analysis/processors/` and follow a simple interface:

```python
from libero_analysis.processors import get_processor

# Per-action processor (low-pass filter)
proc = get_processor("lowpass", cutoff_hz=4.0, fs=30.0, order=2)
proc.reset()                         # call at episode start
action_out = proc(action_in)         # shape (action_dim,)

# Chunk processor (AJAS)
proc = get_processor("ajas_spline", C=100, k_min=1, k_max=40, dt=1/30)
proc.reset()
chunk_out = proc(chunk_in)           # shape (T, action_dim)
```

To plug a processor into a SmolVLA policy (hooks `_get_action_chunk`):

```python
from libero_analysis.processors import wrap_policy_with_chunk_processor
wrap_policy_with_chunk_processor(policy, proc)
```

---

## Training

The fine-tuned model `BrandonAL/my_smolvla_all_1505` was trained via:

```bash
python training/train.py
# or on vast.ai cloud:
bash eval_scripts/run_real_1505.sh
```

See `results/training_runs_summary.md` for the full training history.

---

## Citation

Brandon Lawrence. *Adaptive Jitter-Aware Action Smoothing for Vision-Language-Action Models*. GENG4411 Thesis, University of Western Australia, 2026.

Model on HuggingFace: [`BrandonAL/my_smolvla_all_1505`](https://huggingface.co/BrandonAL/my_smolvla_all_1505)  
Dataset: [`BrandonAL/my_so101_all_1505`](https://huggingface.co/datasets/BrandonAL/my_so101_all_1505)
