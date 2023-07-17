import cv2
from utils.utils import postprocess

from tqdm import tqdm
from typing import Dict

def camera(conf: Dict) -> None:

    # Open the default camera for reading
    camera = cv2.VideoCapture(0)

    # Retrieve camera properties
    frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = camera.get(cv2.CAP_PROP_FPS)

    # Create a VideoWriter object to save the captured video
    output_video = cv2.VideoWriter(conf["save_path"],
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                fps,
                                (frame_width, frame_height))

    # Process each frame from the camera
    while camera.isOpened():
        # Read the current frame from the camera
        ret, frame = camera.read()

        if not ret:
            break

        objs = conf["detector"].detect_emotions(frame)
        modified_frame = postprocess(frame, objs)

        # Write the modified frame to the output video
        output_video.write(modified_frame)

        # Display the modified frame (optional)
        cv2.imshow('Modified Video', modified_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and video writer objects
    camera.release()
    output_video.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

def video(conf: Dict) -> None:

    # Open the video file for reading
    video_path = conf["input_path"]
    video_capture = cv2.VideoCapture(video_path)

    # Retrieve video properties
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Create a VideoWriter object to save the modified video
    output_video = cv2.VideoWriter(conf["save_path"],
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                fps,
                                (frame_width, frame_height))

    # Process each frame of the video
    for idx in tqdm(range(int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))):
        # Read the current frame
        ret, frame = video_capture.read()

        if not ret:
            break

        objs = conf["detector"].detect_emotions(frame)
        modified_frame = postprocess(frame, objs)

        # Write the modified frame to the output video
        output_video.write(modified_frame)

        # Display the modified frame (optional)
        cv2.imshow('Modified Video', modified_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and writer objects
    video_capture.release()
    output_video.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()
