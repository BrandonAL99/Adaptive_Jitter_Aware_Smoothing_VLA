import cv2
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors
from pathlib import Path
from lerobot.cameras.configs import ColorMode, Cv2Rotation
import rerun as rr

NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 40
RESET_TIME_SEC = 15
TASK_DESCRIPTION = "Place blue block in green bowl"

# Robot & teleop config
# Camera config MUST be a dict
camera_config = {
    "front": OpenCVCameraConfig(
        index_or_path=2,
        fps=30,
        width=640,
        height=480,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION
    )
}
robot_config = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="your_follower_arm",
    cameras=camera_config
)
teleop_config = SO101LeaderConfig(
    port="/dev/ttyACM1",
    id="your_leader_arm",
)

robot = SO101Follower(robot_config)
teleop = SO101Leader(teleop_config)

# Dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}


dataset = LeRobotDataset.create(
    repo_id="BrandonAL/dataset_test3",
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4
)


# Keyboard listener & visualization
_, events = init_keyboard_listener()
init_rerun(session_name="recording")

robot.connect()
teleop.connect()

teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()


episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        teleop=teleop,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    # Reset the environment if not stopping or re-recording
    if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
        log_say("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=teleop,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    import time
    time.sleep(1)  # give file handles time to close
    #dataset.push_to_hub(revision="_version_")  
    episode_idx += 1

# Clean up
log_say("Stop recording")
robot.disconnect()
teleop.disconnect()
dataset.push_to_hub()
