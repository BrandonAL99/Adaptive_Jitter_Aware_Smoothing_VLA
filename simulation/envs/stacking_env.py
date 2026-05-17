"""Stacking task: grasp red cube, place it on top of blue cube."""

import numpy as np
from .base_env import SO101BaseEnv

GRASP_HEIGHT_THRESH = 0.05   # metres above starting z to count as grasped
STACK_XY_THRESH     = 0.025  # red cube centre must be within this XY radius of blue
STACK_Z_MIN         = 0.025  # red cube must be at least this far above blue centre z

CUBE_Z = 0.829               # mat top (0.814) + cube half-size (0.015)
CUBE_JITTER_M = 0.015        # ±1.5 cm position jitter per pair centre

# Separation: each cube is offset ±CUBE_SEP in X from the pair's centre point.
# 0.025 m gives ~2 cm gap between cube faces (cube half-size = 0.015).
CUBE_SEP = 0.025

# 5 canonical pair-centre positions (world-frame XY) — same grid as pick_place.
SPAWN_POSITIONS = [
    ( 0.00, -0.125),  # centre
    (-0.19, -0.070),  # top-left
    (-0.19, -0.180),  # bottom-left
    ( 0.10, -0.070),  # top-right
    ( 0.10, -0.180),  # bottom-right
]

# Recording mode uses aligned rotation only (±10° noise)
ROT_ALIGNED = (0.0, 0.175)   # ±10°


class StackingEnv(SO101BaseEnv):
    """
    Two cubes placed side by side (red left or right of blue).
    Task: pick up red, stack it on top of blue.

    Success scoring:
        +0.5  red cube grasped (lifted off table)
        +0.5  red cube stacked on blue

    spawn_config keys (eval mode when any key is present):
        xy_range  : [(x0,x1),(y0,y1)]  pair-centre XY range
        rot_range : (r0, r1)           rotation range (radians)
        red_side  : "left" | "right"   which side red cube starts on
    """

    def __init__(self, render_mode="rgb_array", spawn_config: dict | None = None):
        super().__init__("stacking.xml", render_mode=render_mode)
        self._red_grasped = False
        self._spawn_config = spawn_config

    def _reset_scene(self):
        self._red_grasped = False
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

        # Small independent jitter in recording mode
        if cfg is None:
            red_x  += rng.uniform(-CUBE_JITTER_M, CUBE_JITTER_M)
            blue_x += rng.uniform(-CUBE_JITTER_M, CUBE_JITTER_M)

        a = rot_y / 2  # axis-angle → quaternion half-angle
        quat = [np.cos(a), 0.0, 0.0, np.sin(a)]

        for joint_name, x, y in [("cube_red_joint", red_x, cy),
                                  ("cube_blue_joint", blue_x, cy)]:
            jid  = self.model.joint(joint_name).id
            addr = self.model.jnt_qposadr[jid]
            self.data.qpos[addr:addr + 3]     = [x, y, CUBE_Z]
            self.data.qpos[addr + 3:addr + 7] = quat

    def _compute_reward(self) -> tuple[float, bool, dict]:
        red_pos  = self._get_body_pos("cube_red")
        blue_pos = self._get_body_pos("cube_blue")
        reward   = 0.0

        grasped = red_pos[2] > (CUBE_Z + GRASP_HEIGHT_THRESH)
        if grasped:
            self._red_grasped = True
        if self._red_grasped:
            reward += 0.5

        xy_dist = np.linalg.norm(red_pos[:2] - blue_pos[:2])
        stacked = (
            xy_dist < STACK_XY_THRESH
            and red_pos[2] > blue_pos[2] + STACK_Z_MIN
        )
        if stacked:
            reward += 0.5

        terminated = reward >= 1.0
        info = {
            "red_grasped": self._red_grasped,
            "stacked":     stacked,
            "success":     terminated,
        }
        return reward, terminated, info
