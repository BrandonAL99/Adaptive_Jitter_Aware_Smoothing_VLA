"""
Evaluate a SmolVLA model on LIBERO task suites and record success rate + TC metrics.

Runs N episodes per task across all selected LIBERO suites.  After evaluation:
  - results/<run>/eval_info.json    – per-task success rates and TC summaries
  - results/<run>/episodes.npz      – raw actions per episode (for TC analysis)
  - (optional) HF dataset pushed as per-task splits (task_0, task_1, …)
    containing: action, observation.images.image, observation.images.image2,
                observation.robot_state (34-dim), episode_index, success

Usage:
    # HF model, libero_object, 10 eps per task
    python simulation/scripts/eval_libero.py \
        --model HuggingFaceVLA/smolvla_libero \
        --task libero_object \
        --episodes 10

    # Multiple suites
    python simulation/scripts/eval_libero.py \
        --model HuggingFaceVLA/smolvla_libero \
        --task libero_object libero_spatial libero_goal libero_long \
        --episodes 10 \
        --hf-dataset BrandonAL/eval_relabeled_smolvla_libero_object

    # Skip HF push, custom output dir
    python simulation/scripts/eval_libero.py \
        --model HuggingFaceVLA/smolvla_libero \
        --task libero_object \
        --episodes 5 \
        --output-dir results/my_eval \
        --no-push

Notes:
  - Requires the 'lerobot' conda env: conda activate lerobot
  - For GPU cloud: MUJOCO_GL=osmesa python simulation/scripts/eval_libero.py ...
"""

import argparse
import os
import json
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, Features, Image, Sequence, Value
from huggingface_hub import login
from tqdm import trange

# ── lerobot imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.utils import get_safe_torch_device

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from libero_analysis.temporal_consistency import compute_metrics

# ── constants ──────────────────────────────────────────────────────────────────
LIBERO_FPS   = 30
DT           = 1.0 / LIBERO_FPS
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# 34-dim robot_state layout stored in the HF dataset:
#   eef/mat(9) + eef/pos(3) + eef/quat(4) + gripper/qpos(2) + gripper/qvel(2)
#   + joints/pos(7) + joints/vel(7)
# (joints/pos are at indices 20:27 — used by TC analysis scripts)


# ── helpers ────────────────────────────────────────────────────────────────────

def _flatten_robot_state(rs: dict) -> np.ndarray:
    """Flatten nested robot_state dict to 34-dim numpy array (layout matches HF datasets).
    Layout: eef/mat(9) + eef/pos(3) + eef/quat(4) + gripper/qpos(2) + gripper/qvel(2)
            + joints/pos(7) + joints/vel(7)  = 34 dims total.
    TC analysis uses joints/pos at indices 20:27.
    """
    parts = [
        rs["eef"]["mat"].reshape(-1),
        rs["eef"]["pos"].reshape(-1),
        rs["eef"]["quat"].reshape(-1),
        rs["gripper"]["qpos"].reshape(-1),
        rs["gripper"]["qvel"].reshape(-1),
        rs["joints"]["pos"].reshape(-1),
        rs["joints"]["vel"].reshape(-1),
    ]
    return np.concatenate([p.astype(np.float32) for p in parts])


def _batch_robot_state(rs: dict) -> dict:
    """Add batch dim (dim 0) to every leaf array in the nested robot_state dict.
    Also converts numpy arrays to float32 tensors.
    LiberoProcessorStep expects (B, ...) shaped leaf tensors.
    """
    result = {}
    for k, v in rs.items():
        if isinstance(v, dict):
            result[k] = _batch_robot_state(v)
        elif isinstance(v, np.ndarray):
            t = torch.from_numpy(v).float()
            if t.dim() == 0:
                t = t.unsqueeze(0).unsqueeze(0)   # scalar → (1, 1)
            else:
                t = t.unsqueeze(0)                # (...) → (1, ...)
            result[k] = t
        else:
            result[k] = v
    return result


def _write_json(path: Path, data):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


# ── single-episode rollout ─────────────────────────────────────────────────────

def run_episode(
    env,
    policy,
    env_preprocessor,
    preprocessor,
    postprocessor,
    max_steps: int,
    device: torch.device,
    use_amp: bool = False,
    seed: int | None = None,
) -> tuple[bool, np.ndarray, list, list]:
    """Run one episode on a single (non-vectorized) LiberoEnv.

    Returns
    -------
    success     : bool
    actions     : np.ndarray  shape (T, action_dim)
    images      : list of (H, W, 3) uint8 np arrays  (agentview)
    images2     : list of (H, W, 3) uint8 np arrays  (wrist)
    robot_states: list of 34-dim float32 np arrays
    """
    obs, _ = env.reset(seed=seed)
    policy.reset()

    actions      = []
    rewards      = []
    dones        = []
    images_main  = []
    images_wrist = []
    robot_states = []
    success      = False

    amp_ctx = torch.autocast(device_type=device.type) if use_amp else nullcontext()

    for _ in range(max_steps):
        # ── capture raw data for dataset (before any lerobot preprocessing) ──
        images_main.append(obs["pixels"]["image"].copy())    # (H, W, 3) uint8
        images_wrist.append(obs["pixels"]["image2"].copy())  # (H, W, 3) uint8
        robot_states.append(_flatten_robot_state(obs["robot_state"]))  # (34,)

        # ── build batched obs dict for lerobot preprocessing pipeline ──────────
        # LiberoProcessorStep expects:
        #   "observation.images.image" / "image2" : (1, C, H, W) float32 [0,1]
        #   "observation.robot_state"             : nested dict with (1, ...) tensors
        #   "task"                                : list[str]
        img_main  = (torch.from_numpy(obs["pixels"]["image"])
                     .float().permute(2, 0, 1).unsqueeze(0) / 255.0)   # (1, C, H, W)
        img_wrist = (torch.from_numpy(obs["pixels"]["image2"])
                     .float().permute(2, 0, 1).unsqueeze(0) / 255.0)   # (1, C, H, W)

        processed = {
            "observation.images.image":  img_main,
            "observation.images.image2": img_wrist,
            "observation.robot_state":   _batch_robot_state(obs["robot_state"]),
            "task":                      [env.task_description],
        }

        # ── LiberoProcessorStep (flip images, flatten robot_state → 8-dim) ───
        processed = env_preprocessor(processed)

        # ── policy preprocessor (tokenise task, normalise state, move to GPU) ─
        with amp_ctx:
            policy_input = preprocessor(processed)
            with torch.inference_mode():
                action_raw = policy.select_action(policy_input)

        # ── postprocessor (denormalise action) ────────────────────────────────
        action_processed = postprocessor(action_raw)

        action_np = action_processed.squeeze(0).cpu().numpy().astype(np.float32)
        actions.append(action_np)

        # ── step env ──────────────────────────────────────────────────────────
        obs, reward, terminated, _truncated, info = env.step(action_np)
        rewards.append(float(reward))
        dones.append(bool(terminated))
        if info.get("is_success"):
            success = True
        if terminated:
            break

    return success, np.array(actions, dtype=np.float32), rewards, dones, images_main, images_wrist, robot_states


# ── HF dataset builder ─────────────────────────────────────────────────────────

def build_hf_dataset(episodes: list[dict]) -> Dataset:
    """Convert episode list to a HuggingFace Dataset row-per-timestep.

    Each episode dict must have:
        episode_index, task, actions, images_main, images_wrist, robot_states, success
    """
    rows = []
    global_idx = 0
    for ep in episodes:
        T = len(ep["actions"])
        for t in range(T):
            rows.append({
                "episode_index": int(ep["episode_index"]),
                "index":         global_idx,
                "frame_index":   t,
                "timestamp":     float(t / LIBERO_FPS),
                "action":        ep["actions"][t].astype(np.float32),
                "reward":        float(ep["rewards"][t]),
                "done":          bool(ep["dones"][t]),
                "success":       bool(ep["success"]),
                "observation.images.image":  ep["images_main"][t],
                "observation.images.image2": ep["images_wrist"][t],
                "observation.robot_state":   ep["robot_states"][t].astype(np.float32),
            })
            global_idx += 1

    action_dim = episodes[0]["actions"].shape[1]
    features = Features({
        "episode_index": Value("int32"),
        "index":         Value("int32"),
        "frame_index":   Value("int32"),
        "timestamp":     Value("float32"),
        "action":        Sequence(Value("float32"), length=action_dim),
        "reward":        Value("float32"),
        "done":          Value("bool"),
        "success":       Value("bool"),
        "observation.images.image":  Image(),
        "observation.images.image2": Image(),
        "observation.robot_state":   Sequence(Value("float32"), length=34),
    })
    return Dataset.from_list(rows, features=features)


# ── main evaluation loop ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA on LIBERO")
    parser.add_argument("--model",      required=True,
                        help="HF model id or local path to policy checkpoint")
    parser.add_argument("--task",       nargs="+",
                        default=["libero_object"],
                        choices=["libero_object", "libero_spatial", "libero_goal",
                                 "libero_long", "libero_10", "libero_90"],
                        help="LIBERO task suite(s) to evaluate on")
    parser.add_argument("--episodes",   type=int, default=10,
                        help="Episodes per task (default: 10)")
    parser.add_argument("--max-steps",  type=int, default=500,
                        help="Max steps per episode (default: 500)")
    parser.add_argument("--device",     default="cuda",
                        help="Device: cuda or cpu (default: cuda)")
    parser.add_argument("--output-dir", default="results/libero_eval",
                        help="Directory to save results (default: results/libero_eval)")
    parser.add_argument("--hf-dataset", default=None,
                        help="HF dataset repo id for pushing eval episodes (e.g. BrandonAL/eval_...)")
    parser.add_argument("--no-push",    action="store_true",
                        help="Skip pushing dataset to HuggingFace")
    parser.add_argument("--rename-map", default=None,
                        help='JSON rename map for obs keys e.g. \'{"observation.images.image":"observation.images.camera1"}\'')
    parser.add_argument("--use-amp",    action="store_true",
                        help="Use automatic mixed precision")
    parser.add_argument("--task-ids",   nargs="+", type=int, default=None,
                        help="Restrict to specific task IDs within the suite (e.g. --task-ids 0)")
    parser.add_argument("--seed",       type=int, default=None,
                        help="Base random seed; episode i gets seed+i for reproducibility")
    args = parser.parse_args()

    device = get_safe_torch_device(args.device, log=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    rename_map = {}
    if args.rename_map:
        rename_map = json.loads(args.rename_map)

    # ── load policy ────────────────────────────────────────────────────────────
    print(f"\nLoading policy: {args.model}")
    env_cfg = LiberoEnvConfig(
        task=args.task[0],          # used only for feature inference, not actual env creation
        obs_type="pixels_agent_pos",
    )
    policy_cfg = SmolVLAConfig.from_pretrained(args.model)
    policy_cfg.pretrained_path = args.model
    policy_cfg.device = str(device)

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=rename_map)
    policy.to(device)
    policy.eval()
    print(f"  Action dim : {policy.config.output_features}")

    # ── preprocessors ─────────────────────────────────────────────────────────
    preprocessor_overrides = {
        "device_processor":              {"device": str(device)},
        "rename_observations_processor": {"rename_map": rename_map},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.model,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, _env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )
    print("  Preprocessors loaded.")

    # ── HF login (only if pushing) ─────────────────────────────────────────────
    if not args.no_push and args.hf_dataset:
        print(f"\nLogging in to HuggingFace ...")
        login(token=HF_TOKEN, add_to_git_credential=False)

    # ── per-run accumulators ───────────────────────────────────────────────────
    eval_info = {
        "model":    args.model,
        "episodes": args.episodes,
        "per_task": [],
        "overall":  {},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    all_npz_episodes = []   # for saving episodes.npz at the end
    t_start = time.time()

    # ── loop over suites ───────────────────────────────────────────────────────
    for suite_name in args.task:
        print(f"\n{'='*60}")
        print(f"SUITE: {suite_name}")
        print(f"{'='*60}")

        # build envs for all tasks in this suite (n_envs=1 per task → 1 env/episode)
        suite_envs_dict = make_env(
            cfg=LiberoEnvConfig(
                task=suite_name,
                obs_type="pixels_agent_pos",
                observation_height=360,
                observation_width=360,
            ),
            n_envs=1,
            use_async_envs=False,
        )

        suite_task_envs = suite_envs_dict[suite_name]   # {task_id: SyncVectorEnv}
        if args.task_ids is not None:
            suite_task_envs = {k: v for k, v in suite_task_envs.items() if k in args.task_ids}
        suite_episodes  = []    # all episodes across tasks in this suite (for HF push)
        suite_task_infos = []   # per-task results
        global_ep_idx   = 0

        for task_id in sorted(suite_task_envs.keys()):
            vec_env = suite_task_envs[task_id]
            # Extract the single inner LiberoEnv for direct (non-vectorized) access
            inner_env = vec_env.envs[0]
            task_description = inner_env.task_description
            print(f"\n  Task {task_id}: {inner_env.task}")
            print(f"  Description: {task_description}")

            ep_successes = []
            ep_actions   = []
            ep_tc        = []
            task_ep_dicts = []  # for HF dataset

            for ep in trange(args.episodes, desc=f"  task_{task_id}", leave=False):
                ep_seed = args.seed + ep if args.seed is not None else None
                success, actions, rewards, dones, imgs_main, imgs_wrist, robot_states = run_episode(
                    env=inner_env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    max_steps=args.max_steps,
                    device=device,
                    use_amp=args.use_amp,
                    seed=ep_seed,
                )

                # TC metrics on 7-dim actions
                m = compute_metrics(actions.T, dt=DT)   # actions.T → (7, T)

                ep_successes.append(success)
                ep_actions.append(actions)
                ep_tc.append({
                    "sm_mean":  m["sm_mean"],
                    "sm_max":   m["sm_max"],
                    "ser_mean": m["ser_mean"],
                    "ser_min":  m["ser_min"],
                    "T":        m["T"],
                })

                task_ep_dicts.append({
                    "episode_index": global_ep_idx,
                    "task":          task_description,
                    "success":       success,
                    "actions":       actions,
                    "rewards":       rewards,
                    "dones":         dones,
                    "images_main":   imgs_main,
                    "images_wrist":  imgs_wrist,
                    "robot_states":  robot_states,
                })
                all_npz_episodes.append({
                    "suite":      suite_name,
                    "task_id":    task_id,
                    "episode":    ep,
                    "success":    success,
                    "actions":    actions,
                })

                global_ep_idx += 1
                sr_so_far = sum(ep_successes) / len(ep_successes)
                print(f"    ep {ep+1}/{args.episodes}  success={success}  "
                      f"sr={sr_so_far:.0%}  Sm={m['sm_mean']:.5f}  SER={m['ser_mean']:.4f}  "
                      f"T={m['T']}")

            suite_episodes.extend(task_ep_dicts)

            task_sr = float(np.mean(ep_successes))
            task_sm   = float(np.mean([t["sm_mean"]  for t in ep_tc]))
            task_ser  = float(np.mean([t["ser_mean"] for t in ep_tc]))
            task_info = {
                "suite":       suite_name,
                "task_id":     task_id,
                "task_name":   inner_env.task,
                "n_episodes":  args.episodes,
                "success_rate": task_sr,
                "n_success":   int(sum(ep_successes)),
                "tc": {
                    "sm_mean":  task_sm,
                    "ser_mean": task_ser,
                    "per_ep":   ep_tc,
                },
            }
            suite_task_infos.append(task_info)
            eval_info["per_task"].append(task_info)
            print(f"\n  Task {task_id} summary: sr={task_sr:.1%}  Sm={task_sm:.5f}  SER={task_ser:.4f}")

            # save incremental results
            _write_json(out_dir / "eval_info.json", eval_info)

        # ── close envs ─────────────────────────────────────────────────────────
        close_envs(suite_envs_dict)

        # ── push HF dataset for this suite ────────────────────────────────────
        if not args.no_push and args.hf_dataset:
            # push all tasks as separate splits: task_0, task_1, ...
            task_splits: dict[str, list] = defaultdict(list)
            for ep_d in suite_episodes:
                task_splits[f"task_{ep_d['episode_index'] // args.episodes}"].append(ep_d)

            # re-index per-task episodes starting from 0
            for split_name, split_episodes in task_splits.items():
                base_idx = split_episodes[0]["episode_index"]
                for e in split_episodes:
                    e = dict(e)   # copy
                    e["episode_index"] = e["episode_index"] - base_idx

            # build and push per task split
            task_episode_groups: dict[int, list] = defaultdict(list)
            for ep_d in suite_episodes:
                tid = ep_d["episode_index"] // args.episodes
                task_episode_groups[tid].append(ep_d)

            for tid, t_eps in sorted(task_episode_groups.items()):
                reindexed = []
                for i, ep_d in enumerate(t_eps):
                    ep_copy = dict(ep_d)
                    ep_copy["episode_index"] = i
                    reindexed.append(ep_copy)
                ds = build_hf_dataset(reindexed)
                split_name = f"task_{tid}"
                repo_id = args.hf_dataset
                print(f"\n  Pushing {len(reindexed)} eps to {repo_id} split={split_name} ...")
                try:
                    ds.push_to_hub(repo_id=repo_id, split=split_name)
                    print(f"  OK: https://huggingface.co/datasets/{repo_id}")
                except Exception as e:
                    print(f"  WARNING: HF push failed: {e}")

    # ── overall summary ────────────────────────────────────────────────────────
    all_sr  = [t["success_rate"] for t in eval_info["per_task"]]
    all_sm  = [t["tc"]["sm_mean"]  for t in eval_info["per_task"]]
    all_ser = [t["tc"]["ser_mean"] for t in eval_info["per_task"]]
    eval_info["overall"] = {
        "n_tasks":      len(eval_info["per_task"]),
        "success_rate": float(np.mean(all_sr)),
        "tc": {
            "sm_mean":  float(np.mean(all_sm)),
            "ser_mean": float(np.mean(all_ser)),
        },
        "eval_s": time.time() - t_start,
    }
    _write_json(out_dir / "eval_info.json", eval_info)

    # ── save NPZ for offline TC analysis ──────────────────────────────────────
    npz_actions   = [e["actions"] for e in all_npz_episodes]
    npz_ep_lens   = np.array([len(a) for a in npz_actions])
    npz_successes = np.array([e["success"] for e in all_npz_episodes])
    npz_task_ids  = np.array([e["task_id"] for e in all_npz_episodes])
    npz_suites    = np.array([e["suite"]   for e in all_npz_episodes])
    np.savez_compressed(
        out_dir / "episodes.npz",
        actions    = np.concatenate(npz_actions, axis=0),   # (sum_T, 7)
        ep_lengths = npz_ep_lens,
        successes  = npz_successes,
        task_ids   = npz_task_ids,
        suites     = npz_suites,
    )

    # ── print final table ──────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Suite':<20} {'Task':>4}  {'SR':>6}  {'Sm':>10}  {'SER':>8}  {'N':>4}")
    print(f"  {'-'*60}")
    for t in eval_info["per_task"]:
        print(f"  {t['suite']:<20} {t['task_id']:>4}  "
              f"{t['success_rate']:>6.1%}  "
              f"{t['tc']['sm_mean']:>10.6f}  "
              f"{t['tc']['ser_mean']:>8.4f}  "
              f"{t['n_episodes']:>4}")
    print(f"\n  Overall SR  : {eval_info['overall']['success_rate']:.1%}")
    print(f"  Overall Sm  : {eval_info['overall']['tc']['sm_mean']:.6f}")
    print(f"  Overall SER : {eval_info['overall']['tc']['ser_mean']:.4f}")
    print(f"\n  Time        : {elapsed/60:.1f} min")
    print(f"  Results dir : {out_dir.resolve()}")


if __name__ == "__main__":
    main()
