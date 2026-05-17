"""Real-robot evaluation script for SO101 SmolVLA.

One episode per Enter-key trigger. Episodes push to HuggingFace immediately
after each save so a crash never loses recorded data.

Usage
-----
    cd /path/to/vla_jitter_smoothing
    HF_TOKEN=hf_... conda run -n lerobot python real_robot/eval_real.py

Configuration
-------------
Edit the three constant blocks below:
  1. Uncomment ONE PROCESSOR block
  2. Uncomment ONE TASK line
  3. Uncomment ONE DATASET line (must match your condition + task choice)

Processor matching sim tests (eval_libero_processed.py)
-------------------------------------------------------
  identity   → no processor (baseline)
  lowpass    → LowPassFilterProcessor (ActionProcessor, causal IIR per-action)
  ajas_spline → AJASSplineProcessor (ChunkProcessor, hooked into _get_action_chunk)
"""

import csv
import json
import os
import sys
import time
import threading
from pathlib import Path

import numpy as np
import torch

# ── project path setup ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lerobot" / "src"))

import config
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from huggingface_hub import login

from libero_analysis.processors import (
    get_processor, wrap_policy_with_chunk_processor,
    ActionProcessor, ChunkProcessor,
)
from libero_analysis.temporal_consistency import compute_metrics

# ══════════════════════════════════════════════════════════════════════════════
# 1. PROCESSOR — uncomment ONE block
# ══════════════════════════════════════════════════════════════════════════════

# ── Baseline: no smoothing (identity) ─────────────────────────────────────────
PROCESSOR_NAME   = "identity"
PROCESSOR_KWARGS = {}

# ── 4 Hz Butterworth low-pass filter (ActionProcessor, per-action causal IIR) ─
# PROCESSOR_NAME   = "lowpass"
# PROCESSOR_KWARGS = {"cutoff_hz": 4.0, "fs": 30.0, "order": 2}

# ── AJAS Spline (ChunkProcessor, adaptive keyframe + cubic spline) ─────────────
# Matches: eval_libero_processed.py --processor ajas_spline --spline-c 100
# PROCESSOR_NAME   = "ajas_spline"
# PROCESSOR_KWARGS = {"C": 100.0, "k_min": 1, "k_max": 40, "dt": 1/30,
#                     "skip_joints": (5,), "kf_window": 2, "sm_window": 15}
# Note: skip_joints=(5,) skips the gripper (index 5 for 6-DOF SO101, no 7th joint)

# ══════════════════════════════════════════════════════════════════════════════
# 2. TASK — uncomment ONE line
# ══════════════════════════════════════════════════════════════════════════════

TASK = "Pick up the red cube and place it in the box."       # Pick-place
# TASK = "Put the red cube on top of the blue cube."          # Stacking
# TASK = "Put the red cube in the right box and the blue cube in the left box."  # Sorting

# ══════════════════════════════════════════════════════════════════════════════
# 3. DATASET — uncomment the line matching your condition + task above
# ══════════════════════════════════════════════════════════════════════════════
# Baseline (no processor)
DATASET = "BrandonAL/eval_real_baseline_pickplace"
# DATASET = "BrandonAL/eval_real_baseline_stacking"
# DATASET = "BrandonAL/eval_real_baseline_sorting"

# 4 Hz low-pass filter
# DATASET = "BrandonAL/eval_real_lp4hz_pickplace"
# DATASET = "BrandonAL/eval_real_lp4hz_stacking"
# DATASET = "BrandonAL/eval_real_lp4hz_sorting"

# AJAS Spline
# DATASET = "BrandonAL/eval_real_ajas_pickplace"
# DATASET = "BrandonAL/eval_real_ajas_stacking"
# DATASET = "BrandonAL/eval_real_ajas_sorting"

# ══════════════════════════════════════════════════════════════════════════════
# Fixed config (don't need to change these)
# ══════════════════════════════════════════════════════════════════════════════
MODEL        = "BrandonAL/my_smolvla_all_1505"
FPS          = 30
EPISODE_TIME = 90       # seconds per episode (robot stops after this)
DT           = 1.0 / FPS
LOG_DIR      = ROOT / "results" / "eval_real_robot"

HF_TOKEN = os.environ.get("HF_TOKEN", "")


# ── recording wrapper: captures actions sent to robot for TC metrics ────────────

class _RecordingPostprocessor:
    """Chains lerobot's postprocessor → optional LP ActionProcessor → records."""

    def __init__(self, inner, action_processor: ActionProcessor | None = None):
        self._inner  = inner
        self._ap     = action_processor
        self._buf: list[np.ndarray] = []

    def reset(self):
        self._buf = []
        if self._ap is not None:
            self._ap.reset()

    def __call__(self, action_tensor):
        result    = self._inner(action_tensor)               # denormalize
        action_np = result.squeeze(0).cpu().numpy().astype(np.float32)

        if self._ap is not None:                             # LP filter
            action_np = self._ap(action_np)
            result = (torch.from_numpy(action_np)
                      .unsqueeze(0)
                      .to(dtype=result.dtype, device=result.device))

        self._buf.append(action_np.copy())
        return result

    def get_actions(self) -> np.ndarray:
        return np.array(self._buf, dtype=np.float32) if self._buf else np.empty((0, 6))


# ── helpers ────────────────────────────────────────────────────────────────────

def _tc_summary(actions: np.ndarray) -> dict:
    """Compute Sm and SER over the executed actions (shape T×6)."""
    if actions.shape[0] < 4:
        return {"sm_mean": float("nan"), "ser_mean": float("nan"), "T": len(actions)}
    m = compute_metrics(actions[:, :6].T, dt=DT)  # exclude gripper for TC
    return {"sm_mean": m["sm_mean"], "ser_mean": m["ser_mean"], "T": m["T"]}


def _prompt_score(ep_idx: int, task: str) -> float:
    """Print scoring guide and collect score from user."""
    print()
    print(f"  Episode {ep_idx} finished.")

    if "box" in task.lower() and "right" not in task.lower():
        print("  Score guide  →  0.5 grasp cube  +  0.5 place in box  (max 1.0)")
    elif "on top" in task.lower():
        print("  Score guide  →  0.5 grasp red cube  +  0.5 stack on blue  (max 1.0)")
    elif "right box" in task.lower():
        print("  Score guide  →  0.25 grasp red  +  0.25 sort red  +  0.25 grasp blue  +  0.25 sort blue  (max 1.0)")

    while True:
        raw = input("  Enter score (0–1, or sub-scores like 0.5): ").strip()
        try:
            score = float(raw)
            if 0.0 <= score <= 1.0:
                return score
        except ValueError:
            pass
        print("  Please enter a number between 0 and 1.")


def _prompt_meta() -> tuple[int, str, int]:
    """Ask user for position, config, trial number."""
    print()
    pos = input("  Position (1-5): ").strip()
    cfg = input("  Config / orientation (A or B): ").strip().upper()
    rep = input("  Trial rep (1 or 2): ").strip()
    return int(pos), cfg, int(rep)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    login(token=HF_TOKEN, add_to_git_credential=False)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── build processor ────────────────────────────────────────────────────────
    processor = get_processor(PROCESSOR_NAME, **PROCESSOR_KWARGS)
    is_chunk  = isinstance(processor, ChunkProcessor)
    print(f"\nProcessor : {processor}")
    print(f"Type      : {'ChunkProcessor (policy hook)' if is_chunk else 'ActionProcessor (per-action)'}")
    print(f"Task      : {TASK}")
    print(f"Dataset   : {DATASET}")
    print(f"Model     : {MODEL}")
    print()

    # ── robot & cameras ────────────────────────────────────────────────────────
    camera_configs, _, robot_config = config.autoConfig()
    robot = SO101Follower(robot_config)

    # ── policy ─────────────────────────────────────────────────────────────────
    policy_cfg = SmolVLAConfig.from_pretrained(MODEL)
    policy_cfg.pretrained_path = MODEL
    policy_cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = make_policy(cfg=policy_cfg)
    policy.to(policy_cfg.device)
    policy.eval()

    preprocessor, inner_postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=MODEL,
    )

    # ── hook chunk processor (AJAS) into policy before any episode ─────────────
    if is_chunk:
        wrap_policy_with_chunk_processor(policy, processor)
        policy.config.n_action_steps = 10   # execute 10 spline steps, then re-plan
        print(f"  → n_action_steps set to {policy.config.n_action_steps} for chunk processor")

    # ── recording postprocessor (wraps LP or just records for baseline/AJAS) ───
    ap_arg = processor if isinstance(processor, ActionProcessor) else None
    rec_post = _RecordingPostprocessor(inner_postprocessor, action_processor=ap_arg)

    # ── dataset (resume if exists) ─────────────────────────────────────────────
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features    = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        repo_id=DATASET,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # ── CSV log ────────────────────────────────────────────────────────────────
    csv_path = LOG_DIR / f"{DATASET.split('/')[-1]}.csv"
    csv_exists = csv_path.exists()
    csv_fh = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_fh)
    if not csv_exists:
        csv_writer.writerow([
            "ep_idx", "processor", "task", "position", "config", "trial",
            "score", "sm_mean", "ser_mean", "T", "timestamp",
        ])
        csv_fh.flush()

    # ── keyboard listener & rerun ──────────────────────────────────────────────
    _, events = init_keyboard_listener()
    init_rerun(session_name="eval_real")
    robot.connect()

    ep_idx    = dataset.num_episodes  # continue numbering if resuming
    log_say(f"Ready. Processor={PROCESSOR_NAME}. Dataset={DATASET}")

    try:
        while True:
            print(f"\n{'─'*60}")
            print(f"Episode {ep_idx}  |  {PROCESSOR_NAME}  |  {TASK}")
            print("Place cubes in position, then press Enter to start  (q+Enter to quit).")
            cmd = input("> ").strip().lower()
            if cmd == "q":
                break

            pos, cfg_orient, trial = _prompt_meta()

            # reset per-episode state
            rec_post.reset()
            if is_chunk:
                processor.reset()
            policy.reset()
            log_say(f"Running episode {ep_idx}")

            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=rec_post,
                dataset=dataset,
                control_time_s=EPISODE_TIME,
                single_task=TASK,
                display_data=True,
            )

            dataset.save_episode()

            # ── push immediately ────────────────────────────────────────────────
            print("  Pushing to HuggingFace ...")
            try:
                dataset.push_to_hub(private=True)
                print(f"  Pushed: https://huggingface.co/datasets/{DATASET}")
            except Exception as e:
                print(f"  WARNING: push failed: {e}")

            # ── TC metrics ──────────────────────────────────────────────────────
            actions = rec_post.get_actions()
            tc = _tc_summary(actions)
            print(f"  TC  Sm={tc['sm_mean']:.5f}  SER={tc['ser_mean']:.4f}  T={tc['T']}")

            # ── manual score ───────────────────────────────────────────────────
            score = _prompt_score(ep_idx, TASK)

            # ── log to CSV ─────────────────────────────────────────────────────
            csv_writer.writerow([
                ep_idx, PROCESSOR_NAME, TASK, pos, cfg_orient, trial,
                score, round(tc["sm_mean"], 6), round(tc["ser_mean"], 4), tc["T"],
                time.strftime("%Y-%m-%dT%H:%M:%S"),
            ])
            csv_fh.flush()
            print(f"  Logged ep {ep_idx}: score={score}  Sm={tc['sm_mean']:.5f}")

            ep_idx += 1

    finally:
        robot.disconnect()
        csv_fh.close()
        print(f"\nDone. {ep_idx} total episodes. CSV at {csv_path}")


if __name__ == "__main__":
    main()
