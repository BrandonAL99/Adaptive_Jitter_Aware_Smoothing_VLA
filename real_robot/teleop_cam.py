import cv2
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.teleoperators.so_leader import SOLeader as SO101Leader, SOLeaderTeleopConfig as SO101LeaderConfig
from lerobot.robots.so_follower import SOFollower as SO101Follower, SOFollowerRobotConfig as SO101FollowerConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation
import rerun as rr
import config

rr.init("teleop_with_camera", spawn=True)

#Cameras
"""
camera_configs = {
    "overhead": OpenCVCameraConfig(
        index_or_path=4,
        fps=30,
        width=640,
        height=480,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION
    ),
    "wrist": OpenCVCameraConfig(
        index_or_path=2,
        fps=30,
        width=640,
        height=480,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION
    )
}

robot_config = SO101FollowerConfig(
    port="/dev/ttyACM1",
    id="your_follower_arm",
    cameras=camera_configs
)

teleop_config = SO101LeaderConfig(
    port="/dev/ttyACM0",
    id="your_leader_arm",
)
"""

camera_configs, teleop_config, robot_config = config.autoConfig()
print("config complete")
print("Camera config is")
print(camera_configs)

robot = SO101Follower(robot_config)
teleop = SO101Leader(teleop_config)
print("try to connect")
robot.connect()
print("Connected robot")
teleop.connect()
print("Teleop + rerun GUI running :)")

"""
try:
    while True:
        obs = robot.get_observation()

        # Camera frame is directly under the camera name
        frame = obs["front"]   # shape: (H, W, 3), RGB

        if frame is not None:
            frame_bgr = frame[:, :, ::-1]  # RGB → BGR
            cv2.imshow("Front Camera", frame_bgr)

        action = teleop.get_action()
        robot.send_action(action)

        if cv2.waitKey(1) & 0xFF == 27:
            break
"""

try:
    while True:
        obs = robot.get_observation()
        #print("Obs has keys", obs.keys())
        # ---- Camera ----
        frame1 = obs["camera1"]  # RGB uint8 HWC
        if frame1 is not None:
            rr.log("camera/camera1", rr.Image(frame1))

        frame2 = obs["camera2"]  # RGB uint8 HWC
        if frame2 is not None:
            rr.log("camera/camera2", rr.Image(frame2))

        for key, value in obs.items():
            if key.endswith(".pos"):
                joint_name = key.replace(".pos", "")
                rr.log(f"observation/joints/{joint_name}", rr.Scalars(value))

        # ---- Teleop ----
        action = teleop.get_action()
        # Offset wrist_roll 90 deg to align with gripper.
        # Units are RANGE_M100_100: full 360 deg maps to -100..+100, so 90 deg = 25.
        # Flip sign if the physical offset goes the wrong way.
        WRIST_ROLL_OFFSET = -50.0
        action["wrist_roll.pos"] = action["wrist_roll.pos"] + WRIST_ROLL_OFFSET
        robot.send_action(action)

finally:
    robot.disconnect()
    teleop.disconnect()
    cv2.destroyAllWindows()
