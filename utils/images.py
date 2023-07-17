import cv2

import os
import re
import glob
from tqdm import tqdm
from typing import Dict

from utils.utils import postprocess

def emotion_images(conf: Dict) -> None:
    # Define the regex pattern for matching file extensions
    pattern = re.compile(r'\.(jpg|png|jpeg|bmp)$', re.IGNORECASE)

    # Get a list of all files in the folder
    all_files = glob.glob(os.path.join(conf["input_path"], '*'))

    # Filter the files based on the regex pattern
    img_files = [file for file in all_files if re.search(pattern, file)]

    for img_path in tqdm(img_files):
        # Load the image
        image_name = os.path.split(img_path)[1]
        image = cv2.imread(img_path)
        objs = conf["detector"].detect_emotions(image)
        modified_frame = postprocess(image, objs)
        cv2.imwrite(os.path.join(conf["save_path"], f"{image_name}"), modified_frame)

def emotion_image(conf: Dict) -> None:
    image_name = os.path.split(conf["input_path"])[1]
    image = cv2.imread(conf["input_path"])
    objs = conf["detector"].detect_emotions(image)
    modified_frame = postprocess(image, objs)
    cv2.imwrite(os.path.join(conf["save_path"], f"{image_name}"), modified_frame)
