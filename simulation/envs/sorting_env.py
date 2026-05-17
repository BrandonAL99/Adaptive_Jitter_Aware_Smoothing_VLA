"""Sorting task: red cube → right box, blue cube → left box."""

import numpy as np
from .base_env import SO101BaseEnv

GRASP_HEIGHT_THRESH = 0.05
PLACE_XY_THRESH = 0.045
PLACE_Z_THRESH  = 0.03

# Box centres in world frame (match sorting.xml)
BOX_RIGHT_CENTRE = np.array([ 0.22, -0.15, 0.815])  # red goes here
BOX_LEFT_CENTRE  = np.array([-0.22, -0.15, 0.815])  # blue goes here

CUBE_Z = 0.829  # mat top (0.814) + cube half-size (0.015)
CUBE_JITTER_M = 0.015  # ±1.5 cm position jitter per pair centre

# Each cube is offset ±CUBE_SEP in X from the pair's centre point.
# 0.025 m gives ~2 cm gap between cube faces (cube half-size = 0.015).
CUBE_SEP = 0.025

# 5 canonical pair-centre positions — same grid as pick_place / stacking.
SPAWN_POSITIONS = [
    ( 0.00, -0.125),  # centre
    (-0.19, -0.070),  # top-left
    (-0.19, -0.180),  # bottom-left
    ( 0.10, -0.070),  # top-right
    ( 0.10, -0.180),  # bottom-right
]

# Aligned rotation only (±10° noise)
ROT_ALIGNED = (0.0, 0.175)


class SortingEnv(SO101BaseEnv):
    """
    Success scoring:
        +0.25  red cube grasped
        +0.25  blue cube grasped
        +0.25  red cube in right box
        +0.25  blue cube in left box

    spawn_config keys (eval mode):
        xy_range  : [(x0,x1),(y0,y1)]  pair-centre XY range
        rot_range : (r0, r1)           rotation range (radians)
        red_side  : "left" | "right"   which side red starts on
    """

    def __init__(self, render_mode="rgb_array", spawn_config: dict | None = None):
        super().__init__("sorting.xml", render_mode=render_mode)
        self._red_grasped  = False
        self._blue_grasped = False
        self._spawn_config = spawn_config

    def _reset_scene(self):
        self._red_grasped = False
        self._blue_grasped = False
        rng = self.np_random
        cfg = self._spawn_config

        if cfg is not None and "rot_range" in cfg:
            # ── Eval mode: exact ranges from config ───────────────────
            cx = rng.uniform(*cfg["xy_range"][0])
            cy = rng.uniform(*cfg["xy_range"][1])
            rot_y = float(rng.uniform(*cfg["rot_range"]))
            red_right = cfg["red_side"] == "right"
        else:
            # ── Recording mode: random position, side; aligned rotation ─
            if cfg and "xy_range" in cfg:
                cx = cfg["xy_range"][0][0] + rng.uniform(-CUBE_JITTER_M, CUBE_JITTER_M)
                cy = cfg["xy_range"][1][0] + rng.uniform(-CUBE_JITTER_M, CUBE_JITTER_M)
            else:
                cx, cy = SPAWN_POSITIONS[rng.integers(len(SPAWN_POSITIONS))]
                cx += rng.uniform(-CUBE_JITTER_M, CUBE_JITTER_M)
                cy += rng.uniform(-CUBE_JITTER_M, CUBE_JITTER_M)
            centre, noise = ROT_ALIGNED
            rot_y = float(centre + rng.uniform(-noise, noise))
            if cfg and "red_side" in cfg:
                red_right = cfg["red_side"] == "right"
            else:
                red_right = rng.random() < 0.5

        red_x  = cx + (CUBE_SEP if red_right else -CUBE_SEP)
        blue_x = cx + (-CUBE_SEP if red_right else CUBE_SEP)

        a = rot_y / 2
        quat = [np.cos(a), 0.0, 0.0, np.sin(a)]

        for joint_name, x in [("cube_red_joint", red_x), ("cube_blue_joint", blue_x)]:
            jid  = self.model.joint(joint_name).id
            addr = self.model.jnt_qposadr[jid]
            self.data.qpos[addr:addr + 3]     = [x, cy, CUBE_Z]
            self.data.qpos[addr + 3:addr + 7] = quat

    def _compute_reward(self) -> tuple[float, bool, dict]:
        red_pos  = self._get_body_pos("cube_red")
        blue_pos = self._get_body_pos("cube_blue")
        reward = 0.0

        # Grasp checks
        if red_pos[2]  > CUBE_Z + GRASP_HEIGHT_THRESH:
            self._red_grasped  = True
        if blue_pos[2] > CUBE_Z + GRASP_HEIGHT_THRESH:
            self._blue_grasped = True
        if self._red_grasped:  reward += 0.25
        if self._blue_grasped: reward += 0.25

        # Placement checks
        def _in_box(cube_pos, box_centre):
            return (
                np.linalg.norm(cube_pos[:2] - box_centre[:2]) < PLACE_XY_THRESH
                and abs(cube_pos[2] - box_centre[2]) < PLACE_Z_THRESH
            )

        red_sorted  = _in_box(red_pos,  BOX_RIGHT_CENTRE)
        blue_sorted = _in_box(blue_pos, BOX_LEFT_CENTRE)
        if red_sorted:  reward += 0.25
        if blue_sorted: reward += 0.25

        terminated = reward >= 1.0
        info = {
            "red_grasped":  self._red_grasped,
            "blue_grasped": self._blue_grasped,
            "red_sorted":   red_sorted,
            "blue_sorted":  blue_sorted,
            "success":      terminated,
        }
        return reward, terminated, info
