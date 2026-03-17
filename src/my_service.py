from common_code.config import get_settings
from common_code.logger.logger import get_logger, Logger
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from common_code.tasks.models import TaskData
# Imports required by the service's model
from common_code.tasks.service import get_extension
import numpy as np
import cv2
from model.main_ import main as main_model
from json import JSONEncoder
import json
from paddleocr.ppstructure.utility import parse_args

api_description = """
Inputs:
- Document Image: A single image-based document (JPEG, PNG).

Outputs:
- JSON File: A structured JSON file containing detected parts, including their bounding boxes (bboxes), types,
and confidence scores. Example:
```json
    [
      {"type": "text", "bbox": [12, 730, 410, 848], "score": 0.7757388353347778},
      {"type": "table", "bbox": [15, 360, 405, 711], "score": 0.9503183960914612}
    ]
```
- Annotated Image: The original document image with bounding boxes drawn around detected regions,
labeled with their corresponding types.

Model Specifications:
- Model: PP-PicoDet
- Pretraining Dataset: PubTabNet
- Model Size: 9.7 MB
- Reference : [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
"""
api_summary = """
This service provides advanced layout analysis for image based document,
designed to detect and classify various content regions such as text, titles, tables, and figures.
The service leverages PP-PicoDet, a state-of-the-art real-time object detection model optimized for mobile
and lightweight deployments.
"""

api_title = "My Service API"
version = "0.0.1"

settings = get_settings()


class MyService(Service):
    """
    My layout analysis service model
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="Layout Analysis",
            slug="layout-analysis",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(
                    name="image",
                    type=[
                        FieldDescriptionType.IMAGE_JPEG, FieldDescriptionType.IMAGE_PNG
                    ],
                ),
            ],
            data_out_fields=[
                FieldDescription(
                    name="result_text", type=[FieldDescriptionType.APPLICATION_JSON]
                ),
                FieldDescription(
                    name="result_img", type=[FieldDescriptionType.IMAGE_PNG, FieldDescriptionType.IMAGE_JPEG]
                ),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.IMAGE_PROCESSING,
                    acronym=ExecutionUnitTagAcronym.IMAGE_PROCESSING,
                ),
            ],
            has_ai=True,
            # OPTIONAL: CHANGE THE DOCS URL TO YOUR SERVICE'S DOCS
            docs_url="https://docs.swiss-ai-center.ch/reference/core-concepts/service/",
        )
        self._logger = get_logger(settings)

    def process(self, data):
        # NOTE that the data is a dictionary with the keys being the field names set in the data_in_fields
        # The objects in the data variable are always bytes. It is necessary to convert them to the desired type
        # before using them.

        # Pass specific arguments directly
        args = custom_parse_args(
            vis_font_path="Fonts/arial.ttf",
            use_gpu=False,
            image_dir="img_dir",
            layout_model_dir="model/inference/picodet_lcnet_x1_0_layout_infer",
            layout_dict_path="model/dict/layout_publaynet_dict.txt",
            output="../output",
            table=False,
            ocr=False,
        )

        # Extract the image bytes from data
        image_bytes = data["image"].data  # Extract the raw bytes of the image
        input_type = data["image"].type

        # Decode the image from bytes
        img_ = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), 1)

        res, img = main_model(args, img_)
        guessed_extension = get_extension(input_type)
        is_success, out_buff = cv2.imencode(guessed_extension, img)
        res = CustomEncoder().encode(res)

        # NOTE that the result must be a dictionary with the keys being the field names set in the data_out_fields
        return {
            "result_text": TaskData(data=res, type=FieldDescriptionType.APPLICATION_JSON),

            "result_img": TaskData(
                data=out_buff.tobytes(),
                type=input_type,
            )
        }


def custom_parse_args(**kwargs):
    # Temporarily override `sys.argv`
    import sys  # noqa: E402
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
