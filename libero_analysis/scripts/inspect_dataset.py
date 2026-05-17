from huggingface_hub import login
from datasets import load_dataset, disable_progress_bar
import numpy as np
import matplotlib.pyplot as plt

login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=False)
disable_progress_bar()

repo = "BrandonAL/eval_my_smolvla_libero_object"
ds = load_dataset(repo, split="task_0", streaming=True)

# Print headers from first row
row = next(iter(ds))
print("Columns:", list(row.keys()))
for k, v in row.items():
    if hasattr(v, "shape"):
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")
    elif isinstance(v, list):
        print(f"  {k}: list len={len(v)}  first={v[0]}")
    else:
        print(f"  {k}: {type(v).__name__} = {v}")

# Find joint pos indices within observation.robot_state
print("\nFinding joint_pos slice in observation.robot_state...")
from datasets import load_dataset as _ld
_ds3 = _ld(repo, split="task_0", streaming=True)
_ds3 = _ds3.select_columns(["observation.robot_state", "observation.state.joint_pos"] if False else ["observation.robot_state"])
# Pull two consecutive rows to detect which dims change with joint movement
_rows = []
for _r in _ds3:
    _rows.append(_r["observation.robot_state"])
    if len(_rows) == 50:
        break
_obs_arr = np.array(_rows)   # (50, 34)
# joint pos should vary smoothly; find 7 consecutive dims with highest variance
_vars = np.var(_obs_arr, axis=0)
print("  Per-dim variance:", np.round(_vars, 6))
print("  Top 7 highest-variance dims:", np.argsort(_vars)[-7:][::-1].tolist())

# Collect actions for episode 0 only, stop as soon as episode changes
print("\nCollecting episode 0 actions...")
ds2 = load_dataset(repo, split="task_0", streaming=True)
ds2 = ds2.select_columns(["episode_index", "action"])
actions = []
for row in ds2:
    if int(row["episode_index"]) != 0:
        break
    actions.append(row["action"])

actions = np.array(actions)   # (T, 7)
print(f"  {len(actions)} timesteps, {actions.shape[1]} joints")

JOINT_NAMES = ["j0", "j1", "j2", "j3", "j4", "j5", "gripper"]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
DT = 1 / 30

fig, axs = plt.subplots(7, 1, figsize=(12, 10), sharex=True)
t = np.arange(len(actions)) * DT
for i, ax in enumerate(axs):
    ax.plot(t, actions[:, i], color=COLORS[i], linewidth=1.2)
    ax.set_ylabel(JOINT_NAMES[i], fontsize=8)
    ax.grid(True)
axs[-1].set_xlabel("Time (s)")
plt.suptitle(f"{repo}\nEpisode 0 — raw actions")
plt.tight_layout()
plt.show()