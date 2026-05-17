"""Pick-and-place task: grasp red cube, place in box."""

import numpy as np
from .base_env import SO101BaseEnv

# Success thresholds (metres)
GRASP_HEIGHT_THRESH = 0.05   # cube z must be this far above table to count as grasped
PLACE_XY_THRESH = 0.045      # cube must be within this XY radius of box centre
PLACE_Z_THRESH = 0.03        # cube z must be within box height range

# Approximate box centre in world frame (matches pick_place.xml)
BOX_CENTRE = np.array([0.20, -0.15, 0.815])

CUBE_SPAWN_Z = 0.829  # mat top (0.814) + cube half-size (0.015)

# 5 canonical cube spawn positions (world-frame XY), matching the 5 eval configs
SPAWN_POSITIONS = [
    ( 0.00, -0.125),  # centre
    (-0.19, -0.070),  # top-left
    (-0.19, -0.180),  # bottom-left
    ( 0.10, -0.070),  # top-right
    ( 0.10, -0.180),  # bottom-right
]
CUBE_JITTER_M = 0.015  # ±1.5 cm position jitter applied to each canonical position

# Bimodal rotation: "aligned" (~0°) or "askew" (~45°), each with ±10° noise
ROT_ALIGNED = (0.0,   0.175)   # centre_rad, noise_rad  → roughly -10° to +10°
ROT_ASKEW   = (0.785, 0.175)   # centre_rad, noise_rad  → roughly 35° to 55°

class PickPlaceEnv(SO101BaseEnv):
    """
    Success scoring (mirrors SmolVLA paper):
        +0.5  cube grasped (lifted above table)
        +0.5  cube inside box

    Optional spawn_config overrides the module-level defaults for eval:
        spawn_config = {"xy_range": [(x0,x1),(y0,y1)], "rot_range": (r0,r1)}
    """

    def __init__(self, render_mode="rgb_array", spawn_config: dict | None = None):
        super().__init__("pick_place.xml", render_mode=render_mode)
        self._cube_grasped = False
        self._spawn_config = spawn_config  # None → use module-level defaults

    def _reset_scene(self):
        self._cube_grasped = False
        rng = self.np_random
        cfg = self._spawn_config

        if cfg and "rot_range" in cfg:
            # Eval mode: exact fixed ranges for both position and rotation
            x = rng.uniform(*cfg["xy_range"][0])
            y = rng.uniform(*cfg["xy_range"][1])
            rot_y = float(rng.uniform(*cfg["rot_range"]))
        else:
            # Recording mode: bimodal rotation (half aligned, half askew)
            centre, noise = ROT_ALIGNED if rng.random() < 0.5 else ROT_ASKEW
            rot_y = float(centre + rng.uniform(-noise, noise))
            if cfg:
                # Named config: fixed XY position + ±1 cm jitter
                cx, cy = cfg["xy_range"][0][0], cfg["xy_range"][1][0]
                x = cx + rng.uniform(-CUBE_JITTER_M, CUBE_JITTER_M)
                y = cy + rng.uniform(-CUBE_JITTER_M, CUBE_JITTER_M)
            else:
                # Random config: pick from all 5 canonical positions + jitter
                cx, cy = SPAWN_POSITIONS[rng.integers(len(SPAWN_POSITIONS))]
                x = cx + rng.uniform(-CUBE_JITTER_M, CUBE_JITTER_M)
                y = cy + rng.uniform(-CUBE_JITTER_M, CUBE_JITTER_M)

        a = rot_y / 2

        # Set freejoint qpos: [x, y, z, qw, qx, qy, qz]
        jid = self.model.joint("cube_red_joint").id
        qpos_addr = self.model.jnt_qposadr[jid]
        self.data.qpos[qpos_addr:qpos_addr + 3] = [x, y, CUBE_SPAWN_Z]
        self.data.qpos[qpos_addr + 3:qpos_addr + 7] = [np.cos(a), 0.0, 0.0, np.sin(a)]

    def _compute_reward(self) -> tuple[float, bool, dict]:
        cube_pos = self._get_body_pos("cube_red")
        reward = 0.0

        # 0.5 for grasping: cube lifted above table
        grasped = cube_pos[2] > (CUBE_SPAWN_Z + GRASP_HEIGHT_THRESH)
        if grasped and not self._cube_grasped:
            self._cube_grasped = True
        if self._cube_grasped:
            reward += 0.5

        # 0.5 for placing: cube inside box footprint and at table level
        xy_dist = np.linalg.norm(cube_pos[:2] - BOX_CENTRE[:2])
        in_box = (
            xy_dist < PLACE_XY_THRESH
            and abs(cube_pos[2] - BOX_CENTRE[2]) < PLACE_Z_THRESH
        )
        if in_box:
            reward += 0.5

        terminated = reward >= 1.0
        info = {
            "grasped": self._cube_grasped,
            "in_box": in_box,
            "success": terminated,
        }
        return reward, terminated, info
