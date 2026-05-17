from serial.tools import list_ports
from pathlib import Path
import cv2
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.teleoperators.so_leader import SOLeader as SO101Leader, SOLeaderTeleopConfig as SO101LeaderConfig
from lerobot.robots.so_follower import SOFollower as SO101Follower, SOFollowerRobotConfig as SO101FollowerConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation
import argparse
from dataclasses import dataclass

parser = argparse.ArgumentParser(description="Camera configuration script")
parser.add_argument(
    "--v", "--verbose",
    action="store_true",
    help="Enable verbose output"
)
args = parser.parse_args()

VERBOSE = args.v

@dataclass
class CameraSpec:
    serial: str          # USB ID or serial string
    name: str            # Friendly name
    fps: int = 30
    width: int = 640
    height: int = 480
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION

# Replace these with your own arm serial numbers.
# To find them: python -c "from serial.tools import list_ports; [print(p.device, p.serial_number) for p in list_ports.comports()]"
ROBOT_SERIAL_NUMS = {
    "YOUR_LEADER_SERIAL"   : "your_leader_arm",
    "YOUR_FOLLOWER_SERIAL" : "your_follower_arm",
}


CAMERAS = [
    CameraSpec(
        serial="usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0",
        name="camera1", #overhead cam from ali-express
        fps=30,
        width=640,
        height=480,
        color_mode=ColorMode.BGR,
        rotation=Cv2Rotation.NO_ROTATION
    ),
    CameraSpec(
        serial="usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0",
        name="camera2",#wrist cam from pre-assembled arm
        fps=30,
        width=640,
        height=480,
        color_mode=ColorMode.BGR,
        rotation=Cv2Rotation.NO_ROTATION
    ),
    CameraSpec(
        serial="pci-0000:00:14.0-usb-0:2.3:1.0-video-index0",#"usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index1",
        name="camera2",#wrist cam from ali-express
        fps=30,
        width=640,
        height=480,
        color_mode=ColorMode.BGR,
        rotation=Cv2Rotation.NO_ROTATION
    )
]



def autoConfig():
    if VERBOSE: print("Searching for cams...")
    v4l_dir = Path("/dev/v4l/by-id")
    camera_configs = {}

    if v4l_dir.exists():
        for cam_path in v4l_dir.iterdir():
            for cam in CAMERAS:
                if cam.serial == cam_path.name:
                    if VERBOSE: print(f"Found {cam.name} at {cam_path.resolve()}")
                    camera_configs[cam.name] = OpenCVCameraConfig(
                        index_or_path=str(cam_path.resolve()),
                        fps=cam.fps,
                        width=cam.width,
                        height=cam.height,
                        #color_mode=cam.color_mode,
                        rotation=cam.rotation
                    )


    # List all serial devices
    ports = list_ports.comports()

    if VERBOSE: print("Detected serial devices:")
    for p in ports:
        if (p.serial_number in ROBOT_SERIAL_NUMS):
            if "leader" in ROBOT_SERIAL_NUMS[p.serial_number]:
                if VERBOSE: print(ROBOT_SERIAL_NUMS[p.serial_number], " at ", p.device)
                teleop_config = SO101LeaderConfig(
                    port=p.device,
                    id=ROBOT_SERIAL_NUMS[p.serial_number]
                )  
            elif "follower" in ROBOT_SERIAL_NUMS[p.serial_number]:
                if VERBOSE: print(ROBOT_SERIAL_NUMS[p.serial_number], " at ", p.device)
                robot_config = SO101FollowerConfig(
                    port=p.device,
                    id=ROBOT_SERIAL_NUMS[p.serial_number],
                    cameras=camera_configs
                )   
            else:
                print("WARNING: Unknown serial device \"", p.serial_number, "\" at port \"", p.device)
            


    return(camera_configs, teleop_config, robot_config)

def followerConfig():
    ports = list_ports.comports()
    for p in ports:
        if (p.serial_number in ROBOT_SERIAL_NUMS):
            if "follower" in ROBOT_SERIAL_NUMS[p.serial_number]:
                return((p.device, ROBOT_SERIAL_NUMS[p.serial_number]))
    raise RuntimeError("Follower robot not found")
            

def leaderConfig():
    ports = list_ports.comports()
    for p in ports:
        if (p.serial_number in ROBOT_SERIAL_NUMS):
            if "leader" in ROBOT_SERIAL_NUMS[p.serial_number]:
                return((p.device, ROBOT_SERIAL_NUMS[p.serial_number]))

    raise RuntimeError("leader robot not found")

def camConfig(verbose=False):
    if verbose: print("Searching for cams...")
    v4l_dir = Path("/dev/v4l/by-id")
    cameras_cli = "{ "

    first = True
    if v4l_dir.exists():
        for cam_path in v4l_dir.iterdir():
            for cam in CAMERAS:
                if cam.serial == cam_path.name:
                    if verbose: print(f"Found {cam.name} at {cam_path.resolve()}")
                    if (cam.serial =="usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0"):
                        print("overhead camera at ", cam_path.resolve())
                    elif (cam.serial == "usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0"):
                        print("wrist camera at ", cam_path.resolve())
                    else:
                        print ("UNKNOWN CAM")
                    if not first:
                        cameras_cli += ", "
                    else:
                        first = False

                    cameras_cli += (
                        f'{cam.name}: {{type: opencv, index_or_path: "{str(cam_path.resolve())}", '
                        f'width: {cam.width}, height: {cam.height}, fps: {cam.fps}}}'
                    )

    cameras_cli += " }"
    #print(cameras_cli)
    return cameras_cli

