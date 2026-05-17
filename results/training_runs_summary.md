# Training History — SO101 Real-World SmolVLA Model

Summary of all real-world training runs on the SO101 arm model. Intended as a reference for thesis report writing.

---

## Run 1 — Initial training from SmolVLA base
**Date:** 2026-02-12  
**Cloud:** vast.ai (GPU unspecified in log)  
**Log:** `results/training_logs/my_smolvla_all_training_log.txt`

| Parameter | Value |
|-----------|-------|
| Base model | `lerobot/smolvla_base` (HuggingFace pretrained) |
| Dataset | `BrandonAL/my_svla_so101_all` |
| Episodes | 150 (50 pick-place + 50 stacking + 50 sorting) |
| Target steps | 50,000 |
| **Actual steps** | **~22,000** (stopped manually) |
| Batch size | 64 |
| Optimiser | AdamW, lr=1e-4, cosine decay, 1000 warmup steps |
| Final loss | ~0.009 |
| Output model | `BrandonAL/my_smolvla_all` |

**Notes:** First fine-tune from the SmolVLA pretrained base onto the SO101 real robot tasks. Training was cut short at ~22k steps. The resulting checkpoint `my_smolvla_all` became the base for all subsequent real-world runs.

---

## Run 2 — Extended fine-tune on augmented dataset
**Date:** 2026-03-16 to 2026-03-17  
**Cloud:** vast.ai  
**Log:** `results/training_logs/my_smolvla_all_updates_training.log`

| Parameter | Value |
|-----------|-------|
| Base model | `BrandonAL/my_smolvla_all` (from Run 1) |
| Dataset | `BrandonAL/my_svla_so101_all_v2` |
| Episodes | 246 (150 original + 96 additional with varied cube placement) |
| Frames | ~104,000 |
| Target steps | 200,000 |
| **Actual steps** | **~51,000** (stopped, checkpoint saved) |
| Batch size | 64 |
| Optimiser | AdamW, lr=1e-4, cosine decay, 333 warmup steps (auto-scaled) |
| Final loss | ~0.007–0.008 |
| Output model | `BrandonAL/my_smolvla_all_updated` |

**Notes:** Dataset extended with 96 additional episodes recorded with cubes in varied (non-canonical) positions to improve generalisation. Training was stopped well short of the 200k target due to cloud cost. The model improved on Run 1 but further training was not continued on this dataset. Decided to go back to the `my_smolvla_all` (Run 1 output) as the base for Run 3, as the v2 augmentation was of uncertain value.

---

## Run 3 — Fine-tune on expanded canonical dataset (current)
**Date:** 2026-05-15 onwards  
**Cloud:** vast.ai RTX 5090  
**Log:** `results/training_logs/train_real_1505_20260515_163717.log`

| Parameter | Value |
|-----------|-------|
| Base model | `BrandonAL/my_smolvla_all` (Run 1 output, ~22k steps) |
| Dataset | `BrandonAL/my_so101_all_1505` |
| Episodes | 250 (150 original + 50 new pick-place + 50 new stacking) |
| Target steps | 50,000 (plan to evaluate at ~20k) |
| **Actual steps (as logged)** | **~8,800+** (in progress) |
| Batch size | 64 |
| Optimiser | AdamW, lr=1e-4, cosine decay |
| Video backend | pyav (torchcodec caused frame index errors on merged dataset) |
| Starting loss | ~0.287 (step 200) |
| Loss at ~8,800 steps | ~0.059 |
| Checkpoint frequency | every 5,000 steps |
| Output model | `BrandonAL/my_smolvla_all_1505` |

**Dataset construction (`my_so101_all_1505`):**  
Merged via `lerobot aggregate_datasets`:
- `BrandonAL/my_svla_so101_all` — 150 eps (original canonical dataset from thesis collection)
- `BrandonAL/my_so101_pickplace_1505` — 50 new pick-place eps (recorded May 2026)
- `BrandonAL/my_so101_stacking_1505` — 10 eps (original 25 recorded; recording crashed after ep 25, only 10 were pushed pre-crash; rest unrecoverable)

Total: 210 episodes actually used (150 + 50 + 10 = 210; the merge script counted 250 due to the stacking dataset being re-recorded but crash lost most eps).

> **Correction:** The `my_so101_stacking_1505` dataset ended up with only 10 recoverable episodes after a recording crash at episode ~25. The dataset was patched to reflect 10 episodes before merging.

**Technical issues resolved during setup:**
- Merged dataset caused `FrameTimestampError` in pyav (timestamp drift up to 0.067s at long video durations). Fixed by disabling timestamp tolerance check in `lerobot/src/lerobot/datasets/video_utils.py` on the cloud instance.
- Training is data-bound: `data_s ~2.8s` vs `updt_s ~0.39s` — GPU idle ~88% of time due to slow seeking in large concatenated video files (a consequence of how `aggregate_datasets` stores all episodes from a source in one video file).

**Notes:** This run is intended to produce the model used for evaluation in the thesis. The plan is to evaluate `my_smolvla_all_1505` checkpoints (at 5k/10k/15k/20k steps) against the baseline `my_smolvla_all` across all three tasks (pick-place, stacking, sorting), using both the baseline inference and the AJAS spline smoothing method.

---

## Summary timeline

```
Feb 2026   Run 1: smolvla_base → my_smolvla_all
           150 eps, 22k/50k steps, loss 0.009

Mar 2026   Run 2: my_smolvla_all → my_smolvla_all_updated  
           246 eps (varied placement), 51k/200k steps, loss 0.007

May 2026   Run 3: my_smolvla_all → my_smolvla_all_1505  (current)
           210 eps (150 orig + 50 pp + 10 stack), target 50k steps
           Loss: 0.287 → 0.059 at 8.8k steps (in progress)
```

---

## Planned evaluations (post-training)

After training is complete (or at a chosen checkpoint ~20k steps):

1. **Baseline evaluation** — `my_smolvla_all_1505` raw inference on all 3 tasks
2. **Low-pass filter evaluation** — same model with joint-angle lowpass filter applied
3. **AJAS spline evaluation** — adaptive look-ahead smoothing (pure-pursuit inspired metric)
4. **Comparison against** `my_smolvla_all` (pre-run-3 baseline) to measure effect of added data

Metrics per evaluation:
- Success rate (pick-place: 1.0 max; stacking: 1.0 max; sorting: 1.0 max)
- Temporal consistency (weighted spectral jitter metric)
