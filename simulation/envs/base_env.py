"""Base MuJoCo Gymnasium environment for the SO-101 arm."""

from pathlib import Path

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

MJCF_DIR = Path(__file__).parent.parent / "mjcf"

SIM_HZ  = 500
CTRL_HZ = 30
N_SIM_STEPS = SIM_HZ // CTRL_HZ  # physics substeps per control step

# ── SO-101 joint names ────────────────────────────────────────────────────────
# Update these to match the joint names in your URDF/MJCF exactly.
SO101_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Camera names defined in mjcf/world_base.xml
CAMERA_NAMES = ["camera_overhead", "camera_wrist"]
IMAGE_H, IMAGE_W = 224, 224

# Arm home position — mean episode-start state measured from 51 training episodes.
# In degrees: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
HOME_QPOS_DEG = np.array([0.7, -93.1, 81.5, 77.8, 87.7, 12.2], dtype=np.float32)
HOME_QPOS_RAD = np.deg2rad(HOME_QPOS_DEG)


class SO101BaseEnv(gym.Env):
    """
    Base environment for the SO-101 arm in MuJoCo.

    Observation space (matches LeRobot/SmolVLA expected keys):
        observation.images.top    : (224, 224, 3) uint8
        observation.images.wrist  : (224, 224, 3) uint8
        observation.state         : (6,) float32  — joint positions in radians

    Action space:
        (6,) float32 in [-1, 1] — normalised target joint positions,
        mapped to each joint's range via _apply_action().
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, scene_xml: str, render_mode: str = "rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        xml_path = MJCF_DIR / scene_xml
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self._renderer = mujoco.Renderer(self.model, height=IMAGE_H, width=IMAGE_W)

        # Resolve joint ids (raises KeyError if name not found — fix JOINT_NAMES)
        self._joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in SO101_JOINT_NAMES
        ]
        self._joint_limits = np.array(
            [self.model.jnt_range[jid] for jid in self._joint_ids], dtype=np.float32
        )  # shape (6, 2)

        n_joints = len(SO101_JOINT_NAMES)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_joints,), dtype=np.float32
        )
        self.observation_space = spaces.Dict({
            "observation.images.top": spaces.Box(
                0, 255, shape=(IMAGE_H, IMAGE_W, 3), dtype=np.uint8
            ),
            "observation.images.wrist": spaces.Box(
                0, 255, shape=(IMAGE_H, IMAGE_W, 3), dtype=np.uint8
            ),
            "observation.state": spaces.Box(
                low=self._joint_limits[:, 0],
                high=self._joint_limits[:, 1],
                dtype=np.float32,
            ),
        })

    # ── Core gym interface ────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # Place arm at home position (matches training episode starts)
        for i, jid in enumerate(self._joint_ids):
            self.data.qpos[self.model.jnt_qposadr[jid]] = HOME_QPOS_RAD[i]
        self.data.ctrl[:len(SO101_JOINT_NAMES)] = HOME_QPOS_RAD
        self._reset_scene()
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._apply_action(action)
        for _ in range(N_SIM_STEPS):
            mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward, terminated, info = self._compute_reward()
        return obs, reward, terminated, False, info

    def render(self):
        self._renderer.update_scene(self.data, camera="camera_overhead")
        return self._renderer.render()

    def close(self):
        self._renderer.close()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_obs(self) -> dict:
        images = {}
        for cam in CAMERA_NAMES:
            self._renderer.update_scene(self.data, camera=cam)
            images[cam] = self._renderer.render().copy()

        qpos = np.array(
            [self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self._joint_ids], dtype=np.float32
        )
        return {
            "observation.images.top":   images["camera_overhead"],
            "observation.images.wrist": images["camera_wrist"],
            "observation.state":        qpos,
        }

    def _apply_action(self, action: np.ndarray):
        """Denormalise action from [-1, 1] → joint limits and set ctrl."""
        lo, hi = self._joint_limits[:, 0], self._joint_limits[:, 1]
        target = lo + (np.clip(action, -1.0, 1.0) + 1.0) / 2.0 * (hi - lo)
        self.data.ctrl[:len(SO101_JOINT_NAMES)] = target

    def _get_body_pos(self, body_name: str) -> np.ndarray:
        """Convenience: return world-frame position of a named body."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xpos[body_id].copy()

    # ── Subclass interface ────────────────────────────────────────────────────

    def _reset_scene(self):
        """Override to randomise object starting positions."""
        pass

    def _compute_reward(self) -> tuple[float, bool, dict]:
        """Override to define task-specific reward and termination."""
        return 0.0, False, {}
