from typing import Dict
import argparse
import sys
import os

sys.path.append(os.getcwd())

from utils.images import emotion_images, emotion_image
from utils.videos import video, camera


from fer import FER


def run(conf: Dict) -> None:

    conf["detector"] = FER()

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
        default=0,
        choices=[0,1,2,3],
        help="0 : image, 1 : a folder of images, 2 : video, 3 : live camera",
    )

    parser.add_argument(
        "--input_path",
        type=str,
        default="dataset/",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="results/",
        help="the path that you want to save results there",
    )

    opts = parser.parse_args()
    return opts


if __name__ == "__main__":

    opts = parse_args()
    conf = vars(opts)
    run(conf)