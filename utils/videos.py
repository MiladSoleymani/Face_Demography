import cv2
import numpy as np
import os
import json
from utils.utils import postprocess

from tqdm import tqdm
from typing import Dict
from collections import defaultdict

import supervision as sv
from yolox.tracker.byte_tracker import BYTETracker
from dataclasses import dataclass

from utils.utils import (
    postprocess,
    convert_xywh_to_xyxy,
    detections2boxes,
    match_detections_with_tracks,
    filter,
    extract_emotion_from_dict,
)


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


def camera(conf: Dict) -> None:
    # Open the default camera for reading
    camera = cv2.VideoCapture(0)

    # Retrieve camera properties
    frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = camera.get(cv2.CAP_PROP_FPS)

    # Create a VideoWriter object to save the captured video
    output_video = cv2.VideoWriter(
        conf["save_path"],
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    byte_tracker = BYTETracker(BYTETrackerArgs())
    box_annotator = sv.BoxAnnotator(color=sv.Color.white(), thickness=2, text_scale=1)
    person_details = defaultdict(dict)
    person_emotion = defaultdict(str)

    # Process each frame from the camera
    while camera.isOpened():
        # Read the current frame from the camera
        ret, frame = camera.read()

        if not ret:
            break

        boxes, scores, classids, kpts = conf["face_model"].detect(frame)

        # Ignore frame with no detection
        if len(boxes) != 0:
            boxes = np.asarray(list(map(convert_xywh_to_xyxy, boxes)))
            detections = sv.Detections(
                xyxy=boxes,
                confidence=scores,
                class_id=classids,
            )

            tracks = byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape,
            )

            tracker_id = match_detections_with_tracks(
                detections=detections, tracks=tracks
            )

            detections.tracker_id = np.array(tracker_id)

            mask = np.array(
                [tracker_id is not None for tracker_id in detections.tracker_id],
                dtype=bool,
            )

            detections = filter(detections, mask)

            for xyxy, tracker_id in zip(detections.xyxy, detections.tracker_id):
                cropped_image = sv.crop(frame, xyxy=xyxy)
                objs = conf["detector"].detect_emotions(cropped_image)

                if len(objs) == 0:
                    continue
                person_details[int(tracker_id)] = objs[0]["emotions"]
                person_emotion[tracker_id] = extract_emotion_from_dict(
                    objs[0]["emotions"]
                )

            labels = [
                f"#{tracker_id} {conf*100:0.2f} {person_emotion[tracker_id]}"
                for _, _, conf, class_id, tracker_id in detections
            ]

            annotated_frame = box_annotator.annotate(
                frame, labels=labels, detections=detections
            )
        else:
            annotated_frame = frame

        # objs = conf["detector"].detect_emotions(frame)
        # modified_frame = postprocess(frame, objs)

        # Write the modified frame to the output video
        # output_video.write(annotated_frame)

        # Display the modified frame (optional)
        cv2.imshow("Modified Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the camera and video writer objects
    camera.release()
    output_video.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    with open(os.path.join(conf["save_path"], "person_details.json"), "w") as fout:
        json.dump(person_details, fout)


def video(conf: Dict) -> None:
    # Open the video file for reading
    video_path = conf["input_path"]
    video_capture = cv2.VideoCapture(video_path)

    # Retrieve video properties
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Create a VideoWriter object to save the modified video
    output_video = cv2.VideoWriter(
        conf["save_path"],
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

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
        cv2.imshow("Modified Video", modified_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture and writer objects
    video_capture.release()
    output_video.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()
