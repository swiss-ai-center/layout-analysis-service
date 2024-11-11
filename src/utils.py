import os
import zipfile
import numpy as np
import cv2
import shutil
from json import JSONEncoder
import json
import io
from paddleocr.ppstructure.utility import parse_args



def extract_images_from_zip(data, output_dir="img_dir"):
    """
    Unzips the 'images' entry from the data dictionary and saves each image to the specified directory.

    Args:
    - data: dict, the data dictionary containing the zipped images in bytes.
    - output_dir: str, the directory where extracted images will be saved.

    Returns:
    - List of file paths to the extracted images.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract the zip file bytes from data
    zip_bytes = data["images"].data  # Extract raw bytes of the zip file
    image_paths = []  # List to hold paths of extracted images

    # Read zip file from bytes
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zip_file:
        # Extract each file in the zip archive
        for file_info in zip_file.infolist():
            if file_info.filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
                # Read file contents as bytes
                with zip_file.open(file_info) as img_file:
                    img_data = img_file.read()

                    # Decode image from bytes
                    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), 1)

                    # Save the image to the output directory
                    image_path = os.path.join(output_dir, file_info.filename)
                    cv2.imwrite(image_path, img)


def save_image(data, output_dir="img_dir"):
    """
    Saves a single image from the data dictionary to the specified directory.

    Args:
    - data: dict, the data dictionary containing the image bytes.
    - output_dir: str, the directory where the image will be saved.

    Returns:
    - The file path to the saved image.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract the image bytes from data
    image_bytes = data["images"].data  # Extract the raw bytes of the image
    input_type = data["images"].type

    # Decode the image from bytes
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), 1)

    # Define the path where the image will be saved
    image_path = os.path.join(output_dir, "fuck.png")

    # Save the image to the specified path
    cv2.imwrite(image_path, img)

    return img, input_type


def custom_parse_args(**kwargs):
    # Temporarily override `sys.argv` if necessary
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