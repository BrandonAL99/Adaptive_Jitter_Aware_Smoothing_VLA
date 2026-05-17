import cv2
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
import config 
import time

# Load cameras from your config

camera_configs, teleop_config, robot_config = config.autoConfig()
print("config is")
print(camera_configs)
# Pick the camera you want (by name)
# Replace "camera1" with whatever name you set in CAMERAS
cam_config = camera_configs["camera1"]

# Create camera object
camera = OpenCVCamera(cam_config)
camera.connect()

try:
    while True:
        # Grab a single frame
        frame = camera.async_read(timeout_ms=500)
        if frame is not None:
            # Show the frame
            cv2.imshow("Camera Capture", frame)
            cv2.waitKey(60) #wait 500ms
            #cv2.destroyAllWindows()

            # Save frame
            #cv2.imwrite("captured_frame.png", frame)
            #print("Saved frame as captured_frame.png")
            #time.delay()
        else:
            print("Failed to read frame from camera.")
finally:
    camera.disconnect()
