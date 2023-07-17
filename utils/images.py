import sys

sys.path.append(f"ByteTrack")

import cv2

import os
import re
import glob
from tqdm import tqdm
import numpy as np
import json
from typing import Dict
from collections import defaultdict

from utils.utils import (
    postprocess,
    convert_xywh_to_xyxy,
    detections2boxes,
    match_detections_with_tracks,
    filter,
    extract_emotion_from_dict,
)
import supervision as sv


def emotion_images(conf: Dict) -> None:
    # Define the regex pattern for matching file extensions
    pattern = re.compile(r"\.(jpg|png|jpeg|bmp)$", re.IGNORECASE)

    # Get a list of all files in the folder
    all_files = glob.glob(os.path.join(conf["input_path"], "*"))

    # Filter the files based on the regex pattern
    img_files = [file for file in all_files if re.search(pattern, file)]
    box_annotator = sv.BoxAnnotator(color=sv.Color.white(), thickness=2, text_scale=1)
    person_emotion = defaultdict(str)

    for img_path in tqdm(img_files):
        # Load the image
        image_name = os.path.split(img_path)[1]
        image = cv2.imread(img_path)

        # face model prediction on single frame
        boxes, scores, classids, kpts = conf["face_model"].detect(image)
        boxes = np.asarray(list(map(convert_xywh_to_xyxy, boxes)))
        detections = sv.Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=classids,
        )

        # tracks = byte_tracker.update(
        #     output_results=detections2boxes(detections=detections),
        #     img_info=image.shape,
        #     img_size=image.shape,
        # )

        # tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)

        detections.tracker_id = np.arange(
            0, len(detections)
        )  # Manual tracker for image

        # mask = np.array(
        #     [tracker_id is not None for tracker_id in detections.tracker_id],
        #     dtype=bool,
        # )

        # detections = filter(detections, mask)

        for xyxy, tracker_id in zip(detections.xyxy, detections.tracker_id):
            cropped_image = sv.crop(image, xyxy=xyxy)
            objs = conf["detector"].detect_emotions(cropped_image)[0]
            person_emotion[tracker_id] = extract_emotion_from_dict(objs["emotions"])

        labels = [
            f"#{tracker_id} {conf*100:0.2f} {person_emotion[tracker_id]}"
            for _, _, conf, class_id, tracker_id in detections
        ]

        with sv.ImageSink(
            target_dir_path=os.path.join(os.getcwd(), conf["save_path"]),
            overwrite=False,
        ) as sink:
            annotated_frame = box_annotator.annotate(
                image, labels=labels, detections=detections
            )

            sink.save_image(
                image=annotated_frame,
                image_name=f"{image_name}",
            )


def emotion_image(conf: Dict) -> None:
    image_name = os.path.split(conf["input_path"])[1]
    image = cv2.imread(conf["input_path"])
    objs = conf["detector"].detect_emotions(image)
    modified_frame = postprocess(image, objs)
    cv2.imwrite(os.path.join(conf["save_path"], f"{image_name}"), modified_frame)
