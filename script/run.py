from typing import Dict
import argparse
import sys
import os
import cv2

sys.path.append(os.getcwd())

from utils.images import emotion_images, emotion_image
from utils.videos import video, camera
from models.face_detection import YOLOv8_face

from ultralytics import YOLO

import shutil
from fer import FER


def run(conf: Dict) -> None:
    # save_path
    if os.path.exists(os.path.join(os.getcwd(), conf["save_path"])):
        shutil.rmtree(os.path.join(os.getcwd(), conf["save_path"]))

    os.makedirs(os.path.join(os.getcwd(), conf["save_path"]), exist_ok=True)
    conf["detector"] = FER()
    conf["face_model"] = YOLOv8_face()

    if conf["source"] == 0:
        emotion_image(conf)
    elif conf["source"] == 1:
        emotion_images(conf)
    elif conf["source"] == 2:
        video(conf)
    elif conf["source"] == 3:
        camera(conf)
    else:
        raise Exception(f"unknown data_type {conf['data_type']}")


def parse_args() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help="0 : image, 1 : a folder of images, 2 : video, 3 : live camera",
    )

    parser.add_argument(
        "--input_path",
        type=str,
        default="dataset/",
    )

    parser.add_argument(
        "--yolo_face",
        type=str,
        default="models/yolov8n-face.onnx",
        help="setting yolo type that want to use",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="results",
        help="the path that you want to save results there",
    )

    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parse_args()
    conf = vars(opts)
    run(conf)
