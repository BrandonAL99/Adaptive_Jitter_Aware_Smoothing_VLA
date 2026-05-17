"""
Compute temporal consistency metrics across all Libero eval datasets.
Results are written to libero_analysis/metrics_all.txt as they are computed.

Usage:
  python libero_analysis/compute_metrics_all.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from libero_analysis.temporal_consistency import compute_metrics

DATASETS = [
    "BrandonAL/eval_smolvla_chunk50_libero_object",
    "BrandonAL/eval_smolvla_chunk50_libero_spatial",
    "BrandonAL/eval_smolvla_chunk50_libero_goal",
    "BrandonAL/eval_smolvla_chunk50_libero_10",
    "BrandonAL/eval_smolvla_chunk50_lowpass_2hz_libero_object",
]

DT      = 1.0 / 30.0
OUT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics_all_chunk50.txt")
COL_W   = 10   # column width for per-joint values
JOINTS  = ["j0", "j1", "j2", "j3", "j4", "j5", "gripper"]

# ── HF helpers ────────────────────────────────────────────────────────────────
def _login():
    try:
        from huggingface_hub import login
        login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=False)
    except Exception:
        pass

def _stream(repo, split, cols):
    from datasets import load_dataset
    ds = load_dataset(repo, split=split, streaming=True)
    return ds.select_columns(cols)

def _build_index(repo):
    """Returns {global_ep: (split, local_ep)}, {global_ep: success}."""
    ep_index, success = {}, {}
    global_ep = 0
    for i in range(20):
        split = f"task_{i}"
        try:
            stream = _stream(repo, split, ["episode_index", "success"])
        except Exception:
            break
        local_seen = {}
        for row in stream:
            loc = int(row["episode_index"])
            if loc not in local_seen:
                local_seen[loc] = global_ep
                ep_index[global_ep] = (split, loc)
                global_ep += 1
            success[local_seen[loc]] = bool(row["success"])
        print(f"    indexed {split}: {len(local_seen)} episodes", flush=True)
    return ep_index, success

def _fetch_episode_actions(repo, split, local_ep):
    """Stream one episode's actions, stop as soon as episode changes."""
    stream = _stream(repo, split, ["episode_index", "action"])
    rows, started = [], False
    for row in stream:
        loc = int(row["episode_index"])
        if loc == local_ep:
            rows.append(row["action"])
            started = True
        elif started:
            break
    return np.array(rows).T  # (n_joints, T)

# ── Writing helpers ───────────────────────────────────────────────────────────
def _write(f, line=""):
    print(line)
    f.write(line + "\n")
    f.flush()

def _sep(f, char="─", width=100):
    _write(f, char * width)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    _login()
    from datasets import disable_progress_bar
    disable_progress_bar()

    all_dataset_summaries = []

    with open(OUT, "w") as f:
        _write(f, "LIBERO EVAL DATASETS — TEMPORAL CONSISTENCY METRICS")
        _write(f, "Sm  = weighted mean frequency  (lower = smoother)")
        _write(f, "SER = signal energy ratio below 1 Hz  (higher = smoother)")
        _write(f)

        for repo in DATASETS:
            _sep(f, "═")
            _write(f, f"DATASET: {repo}")
            _sep(f, "═")

            print(f"\n  Building episode index ...", flush=True)
            ep_index, success = _build_index(repo)
            n_eps = len(ep_index)
            _write(f, f"  Episodes: {n_eps}")
            _write(f)

            # Per-episode header
            joint_header = "".join(f"{'Sm_'+j:>{COL_W}}" for j in JOINTS)
            _write(f, f"  {'ep':>5}  {'success':>8}  {'steps':>6}  {'Sm_mean':>10}  {'SER_mean':>10}  {joint_header}")
            _sep(f, "─")

            ep_sms, ep_sers, ep_sucs = [], [], []

            for global_ep in sorted(ep_index.keys()):
                split, local_ep = ep_index[global_ep]
                actions = _fetch_episode_actions(repo, split, local_ep)
                m       = compute_metrics(actions, DT)
                suc     = success.get(global_ep)

                ep_sms.append(m["sm_mean"])
                ep_sers.append(m["ser_mean"])
                ep_sucs.append(suc)

                joint_sms = "".join(f"{v:>{COL_W}.6f}" for v in m["sm_per_joint"])
                suc_str   = "True " if suc else "False" if suc is False else "  -  "
                _write(f, f"  {global_ep:>5}  {suc_str:>8}  {actions.shape[1]:>6}  "
                           f"{m['sm_mean']:>10.6f}  {m['ser_mean']:>10.4f}  {joint_sms}")

            # Dataset summary
            _sep(f, "─")
            n_suc = sum(1 for s in ep_sucs if s)
            sr    = n_suc / n_eps if n_eps else 0
            sms   = np.array(ep_sms)
            sers  = np.array(ep_sers)
            _write(f, f"  SUMMARY  n={n_eps}  success_rate={sr:.1%} ({n_suc}/{n_eps})")
            _write(f, f"  Sm   mean={sms.mean():.6f}  std={sms.std():.6f}  "
                       f"min={sms.min():.6f}  max={sms.max():.6f}")
            _write(f, f"  SER  mean={sers.mean():.4f}   std={sers.std():.4f}   "
                       f"min={sers.min():.4f}   max={sers.max():.4f}")
            _write(f)

            all_dataset_summaries.append({
                "repo":    repo.split("/")[-1],
                "n_eps":   n_eps,
                "sr":      sr,
                "sm_mean": sms.mean(),
                "sm_std":  sms.std(),
                "ser_mean": sers.mean(),
                "ser_std":  sers.std(),
            })

        # Cross-dataset summary table
        _sep(f, "═")
        _write(f, "CROSS-DATASET SUMMARY")
        _sep(f, "═")
        _write(f, f"  {'dataset':<45}  {'n':>5}  {'success':>8}  "
                   f"{'Sm_mean':>10}  {'Sm_std':>8}  {'SER_mean':>10}  {'SER_std':>8}")
        _sep(f, "─")
        for s in all_dataset_summaries:
            _write(f, f"  {s['repo']:<45}  {s['n_eps']:>5}  {s['sr']:>7.1%}  "
                       f"{s['sm_mean']:>10.6f}  {s['sm_std']:>8.6f}  "
                       f"{s['ser_mean']:>10.4f}  {s['ser_std']:>8.4f}")
        _sep(f, "═")
        _write(f, f"Output written to: {OUT}")

    print(f"\nDone. Results saved to {OUT}")

if __name__ == "__main__":
    main()