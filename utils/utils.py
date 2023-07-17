import math

import cv2
import numpy as np

import supervision as sv
from yolox.tracker.byte_tracker import STrack
from onemetric.cv.utils.iou import box_iou_batch

from typing import List


def postprocess(image: np.array, objs: List) -> np.array:
    # Define the font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    for face in objs:
        # Sort dictionary based on values and retrieve sorted keys and values as tuples
        sorted_items = sorted(
            face["emotions"].items(), key=lambda x: x[1], reverse=True
        )
        # Retrieve sorted keys and values separately
        sorted_keys = [item[0] for item in sorted_items]
        sorted_values = [item[1] for item in sorted_items]
        # Iterate over each detected face
        dominant_emotion = sorted_keys[0] + " " + str(sorted_values[0])
        region = face["box"]
        cv2.rectangle(
            image,
            (region[0], region[1]),
            (region[0] + region[2], region[1] + region[3]),
            (0, 255, 0),
            2,
        )

        # Get the size of the label text
        (label_width, _), _ = cv2.getTextSize(
            dominant_emotion, font, font_scale, thickness
        )
        # Calculate the position for placing the label
        label_x = region[0] + int((region[2] - label_width) / 2)
        label_y = region[1] - 10  # Adjust the offset as needed
        # Write the label text on the image
        cv2.putText(
            image,
            dominant_emotion,
            (label_x, label_y),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA,
        )

    return image


def convert_xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    expansion_factor = 0.14
    expansion_amount_h = int(h * expansion_factor)
    expansion_amount_w = int(w * expansion_factor)

    y1 -= expansion_amount_h
    y2 += expansion_amount_h

    x1 -= expansion_amount_w
    x2 += expansion_amount_w

    return (x1, y1, x2, y2)


def detections2boxes(detections: sv.Detections) -> np.ndarray:
    return np.hstack((detections.xyxy, detections.confidence[:, np.newaxis]))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([track.tlbr for track in tracks], dtype=float)


def match_detections_with_tracks(
    detections: sv.Detections, tracks: List[STrack]
) -> sv.Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)
    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids


def filter(detection, mask: np.ndarray):
    """
    Filter the detections by applying a mask

    :param mask: np.ndarray : A mask of shape (n,) containing a boolean value for each detection indicating if it should be included in the filtered detections
    :param inplace: bool : If True, the original data will be modified and self will be returned.
    :return: Optional[np.ndarray] : A new instance of Detections with the filtered detections, if inplace is set to False. None otherwise.
    """
    detection.xyxy = detection.xyxy[mask]
    detection.confidence = detection.confidence[mask]
    detection.class_id = detection.class_id[mask]
    detection.tracker_id = (
        detection.tracker_id[mask] if detection.tracker_id is not None else None
    )

    return detection


def extract_emotion_from_dict(emotions):
    return list(sorted(emotions.items(), key=lambda x: x[1], reverse=True))[0][0]
