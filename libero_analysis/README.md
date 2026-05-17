# Libero Analysis

This folder contains analysis code for evaluating and comparing the temporal consistency
of action sequences produced by SmolVLA on the LIBERO simulation benchmark, before and
after applying the pure-pursuit-style smoothing function.

---

## Goal

Use the LIBERO eval datasets (collected from `HuggingFaceVLA/smolvla_libero`) as a test
bed for the smoothing method, because:

- The HF model already performs well (87-90% success rate), so any regression from
  smoothing is easy to detect.
- Pre-collected eval datasets are already saved to HuggingFace, so no re-running of
  the policy is needed to get baseline temporal consistency numbers.
- LIBERO is a cleaner benchmark than the SO-101 sim (which had poor performance even
  at 200k steps).

The comparison will be: **success rate + Sm/SER before smoothing vs after smoothing**.

---

## Models

| Model | Description |
|-------|-------------|
| `HuggingFaceVLA/smolvla_libero` | Official HF pre-trained SmolVLA on full LIBERO suite. Used for all HF eval datasets below. |
| `BrandonAL/my_smolvla_libero` | Fine-tuned from `lerobot/smolvla_base` on `HuggingFaceVLA/libero` for 50k steps (see training details below). Originally named `BrandonAL/libero-full`. |

---

## Training Details (BrandonAL/my_smolvla_libero)

| Parameter | Value |
|-----------|-------|
| Base model | `lerobot/smolvla_base` |
| Training dataset | `HuggingFaceVLA/libero` (273K frames, 1,693 episodes) |
| Steps | 50,000 |
| Batch size | 32 |
| Optimiser | AdamW, lr=1e-4, cosine decay, 1000 warmup steps |
| Training time | ~4h 41min |
| Initial loss | 0.403 |
| Final loss | 0.072 |
| Checkpoints saved | 20K, 40K, 50K steps |
| Camera rename | `observation.images.image` → `camera1`, `observation.images.image2` → `camera2` |

---

## LIBERO Environment Details

- **Simulator:** LIBERO (MuJoCo-based)
- **Robot:** 7-DOF manipulator (actions shape `(7,)`)
- **Control mode:** Relative end-effector control
- **FPS:** 30 Hz
- **Cameras:** `agentview_image` (360x360) + `robot0_eye_in_hand_image` (360x360)
- **State:** EEF pos/quat/mat, gripper qpos/vel, joint pos/vel → flattened to shape `(8,)`
- **Task suites:** `libero_object`, `libero_spatial`, `libero_goal`, `libero_long`
  (10 tasks each, binary success reward)

---

## Eval Datasets on HuggingFace

These are the **keeper** datasets. All evaluated using `HuggingFaceVLA/smolvla_libero`
unless noted.

| Dataset (HF repo) | Model evaluated | Task suite | Episodes | Notes |
|-------------------|-----------------|------------|----------|-------|
| `BrandonAL/eval_HFvla_smolvla_libero_object` | `HuggingFaceVLA/smolvla_libero` | libero_object | 10 eps x 10 tasks = 100 | Renamed from `my_smolvla_libero_object` |
| `BrandonAL/eval_HFvla_smolvla_libero_spatial` | `HuggingFaceVLA/smolvla_libero` | libero_spatial | 10 eps x 10 tasks = 100 | Renamed from `my_smolvla_libero_spatial` |
| `BrandonAL/eval_HFvla_smolvla_libero_goal` | `HuggingFaceVLA/smolvla_libero` | libero_goal | 10 eps x 10 tasks = 100 | Renamed from `my_smolvla_libero_goal` |
| `BrandonAL/eval_HFvla_smolvla_libero_long` | `HuggingFaceVLA/smolvla_libero` | libero_long | 10 eps x 10 tasks = 100 | Renamed from `my_smolvla_libero_long` |
| `BrandonAL/eval_my_smolvla_libero_object` | `BrandonAL/my_smolvla_libero` | libero_object | 5 eps x 10 tasks = 50 | Fine-tuned model eval; renamed from `my_smolvla_libero_object_2` |
| `BrandonAL/libero_eval_06022026v2` | `HuggingFaceVLA/smolvla_libero` | libero_object | 1 ep x 10 tasks = 10 | Early eval run, small sample |

### Known Baseline Results

**`HuggingFaceVLA/smolvla_libero` on `libero_object`:**

| Run | Config | Result |
|-----|--------|--------|
| 2026-02-06 log | 4 eps x 10 tasks (40 total) | **87.5%** (35/40) |
| 2026-01-17 eval_info.json | 3 eps x 10 tasks (30 total) | **90.0%** (27/30) |

Per-task breakdown (2026-01-17, 3 eps each):

| Task | Successes | Rate |
|------|-----------|------|
| task_0 | 3/3 | 100% |
| task_1 | 2/3 | 67% |
| task_2 | 2/3 | 67% |
| task_3 | 2/3 | 67% |
| task_4 | 3/3 | 100% |
| task_5 | 3/3 | 100% |
| task_6 | 3/3 | 100% |
| task_7 | 3/3 | 100% |
| task_8 | 3/3 | 100% |
| task_9 | 3/3 | 100% |

Local eval output stored at: `results/Libero_outputs/06-42-12_libero_smolvla/`

---

## Temporal Consistency Metrics

Defined in `temporal_consistency.py`. Both metrics are computed per-joint from the FFT
of the action sequence.

**Sm — weighted mean frequency**
```
Sm = (2 / (n * fs)) * sum( |F(f)| * f )
```
Range [0, 1]. Lower = smoother (energy concentrated at low frequencies).

**SER — Signal Energy Ratio**
```
SER = energy below 1 Hz / total energy
```
Range [0, 1]. Higher = smoother.

### Quick usage

```python
from libero_analysis.temporal_consistency import compute_metrics, print_metrics

# actions: np.ndarray shape (n_joints, T) or (T, n_joints)
m = compute_metrics(actions, dt=1/30)
print_metrics(m, label="baseline HF model")

# windowed version — useful for the pure-pursuit lookahead
from libero_analysis.temporal_consistency import compute_metrics_windowed
windows = compute_metrics_windowed(actions, dt=1/30, window=20, step=5)
```

---

## Planned Analysis Scripts

- `analyse_baseline.py` — load each HF eval dataset, compute Sm/SER per episode,
  save summary to `results/libero_analysis_baseline.json`
- `apply_smoothing.py` — run the pure-pursuit smoother over the baseline action
  sequences, save smoothed episodes
- `compare_results.py` — plot Sm, SER, and success rate before vs after smoothing

---

## Notes

- The SO-101 MuJoCo sim approach was abandoned due to poor policy performance even
  at 200k training steps. LIBERO is now the primary test bed for the smoothing work.
- The HF `smolvla_libero` model operates at 30 Hz with an action chunk size of 50.
- The `my_smolvla_libero` fine-tune reached loss 0.072 but has not been evaluated
  as thoroughly as the official HF model.
- For the smoothing comparison, `eval_HFvla_smolvla_libero_object` (100 episodes) is
  the primary dataset to use — largest sample, best-documented baseline results.
