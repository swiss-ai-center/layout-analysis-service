import os
import numpy as np
import cv2
from json import JSONEncoder
import json
from paddleocr.ppstructure.utility import parse_args


def save_image(data, output_dir="img_dir"):
    """
    Saves a single image from the data dictionary to the specified directory.

    Args:
    - data: dict, the data dictionary containing the image bytes.
    - output_dir: str, the directory where the image will be saved.

    Returns:
    - The image and the type.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract the image bytes from data
    image_bytes = data["images"].data  # Extract the raw bytes of the image
    input_type = data["images"].type

    # Decode the image from bytes
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), 1)

    # Define the path where the image will be saved
    image_path = os.path.join(output_dir, "image.png")

    # Save the image to the specified path
    cv2.imwrite(image_path, img)

    return img, input_type


def custom_parse_args(**kwargs):
    # Temporarily override `sys.argv`
    import sys
    original_argv = sys.argv
    sys.argv = ["main.py"] + [f"--{k}={v}" for k, v in kwargs.items()]

    args = parse_args()

    # Restore original argv
    sys.argv = original_argv
    return args


class CustomEncoder(JSONEncoder):
    def default(self, o):
        return json.dumps(
            o,
            default=lambda x: x.__dict__,
            sort_keys=True,
            indent=4)