import cv2

import os
import re
import glob
from tqdm import tqdm
import numpy as np
from typing import Dict

from utils.utils import postprocess, convert_xywh_to_xyxy
import supervision as sv


def emotion_images(conf: Dict) -> None:
    # Define the regex pattern for matching file extensions
    pattern = re.compile(r"\.(jpg|png|jpeg|bmp)$", re.IGNORECASE)

    # Get a list of all files in the folder
    all_files = glob.glob(os.path.join(conf["input_path"], "*"))

    # Filter the files based on the regex pattern
    img_files = [file for file in all_files if re.search(pattern, file)]
    box_annotator = sv.BoxAnnotator()

    for img_path in tqdm(img_files):
        # Load the image
        image_name = os.path.split(img_path)[1]
        image = cv2.imread(img_path)
        # objs = conf["detector"].detect_emotions(image)
        # modified_frame = postprocess(image, objs)
        # cv2.imwrite(os.path.join(conf["save_path"], f"{image_name}"), modified_frame)

        # face model prediction on single frame
        boxes, scores, classids, kpts = conf["face_model"].detect(image)
        boxes = np.asarray(list(map(convert_xywh_to_xyxy, boxes)))
        detections = sv.Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=classids,
        )

        for xyxy in detections.xyxy:
            cropped_image = sv.crop(image, xyxy=xyxy)
            objs = conf["detector"].detect_emotions(cropped_image)[0]
            print(f"{objs['emotions'] = }")

        labels = [
            f"#{class_id} {conf*100:0.2f}" for _, _, conf, class_id, _ in detections
        ]
        # detections = detections.with_nms(threshold=0.70)

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
