import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lerobot" / "src"))
import config

HF_USER       = "BrandonAL"
DATASET_REPO  = f"{HF_USER}/my_so101_stacking_1505"
TASK          = "Put the red cube on top of the blue cube."
EPISODES_PER_BATCH = 7

LEROBOT_ROOT  = Path(__file__).resolve().parents[2] / "lerobot"
RECORD_SCRIPT = "src/lerobot/scripts/lerobot_record.py"

LOCAL_DATASET = (
    Path.home() / ".cache" / "huggingface" / "lerobot" / DATASET_REPO
)


def record_batch():
    (follower_port, follower_name) = config.followerConfig()
    (leader_port, leader_name)     = config.leaderConfig()
    cameras_config                 = config.camConfig()

    if cameras_config.strip() in ("{}", "{  }", "{ }"):
        raise RuntimeError(
            "No cameras detected — check USB connections before recording.\n"
            f"cameras_config={cameras_config!r}"
        )

    missing = [name for name in ("camera1", "camera2") if name not in cameras_config]
    if missing:
        raise RuntimeError(
            f"Only partial camera set detected — missing: {missing}\n"
            "Both camera1 (overhead) and camera2 (wrist) must be connected.\n"
            f"cameras_config={cameras_config!r}"
        )

    resume = LOCAL_DATASET.exists()
    if resume:
        print(f"Resuming existing dataset ({LOCAL_DATASET})")
    else:
        print("Starting new dataset")

    cmd = [
        sys.executable,
        RECORD_SCRIPT,
        "--robot.type=so101_follower",
        f"--robot.port={follower_port}",
        f"--robot.id={follower_name}",
        f"--robot.cameras={cameras_config}",
        "--teleop.type=so101_leader",
        f"--teleop.port={leader_port}",
        f"--teleop.id={leader_name}",
        "--display_data=true",
        f"--dataset.repo_id={DATASET_REPO}",
        f"--dataset.num_episodes={EPISODES_PER_BATCH}",
        f'--dataset.single_task={TASK}',
        "--dataset.push_to_hub=true",
    ]
    
    if True:
        cmd.append("--resume=true")
    
    subprocess.run(cmd, check=True, cwd=LEROBOT_ROOT)
    print(f"Batch done — pushed to https://huggingface.co/datasets/{DATASET_REPO}")


if __name__ == "__main__":
    record_batch()
