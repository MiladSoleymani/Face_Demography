import cv2
import numpy as np

from typing import List

def postprocess(image:np.array, objs:List) -> np.array:
    # Define the font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    for face in objs:
        # Sort dictionary based on values and retrieve sorted keys and values as tuples
        sorted_items = sorted(face['emotions'].items(), key=lambda x: x[1], reverse=True)
        # Retrieve sorted keys and values separately
        sorted_keys = [item[0] for item in sorted_items]
        sorted_values = [item[1] for item in sorted_items]
        # Iterate over each detected face
        dominant_emotion = sorted_keys[0] + " " + str(sorted_values[0])
        region = face['box']
        cv2.rectangle(image, (region[0], region[1]), (region[0] + region[2], region[1] + region[3]), (0, 255, 0), 2)

        # Get the size of the label text
        (label_width, _), _ = cv2.getTextSize(dominant_emotion, font, font_scale, thickness)
        # Calculate the position for placing the label
        label_x = region[0] + int((region[2] - label_width) / 2)
        label_y = region[1] - 10  # Adjust the offset as needed
        # Write the label text on the image
        cv2.putText(image, dominant_emotion, (label_x, label_y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    return image


