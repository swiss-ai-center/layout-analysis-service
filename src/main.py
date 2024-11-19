import asyncio
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from common_code.config import get_settings
from common_code.http_client import HttpClient
from common_code.logger.logger import get_logger, Logger
from common_code.service.controller import router as service_router
from common_code.service.service import ServiceService
from common_code.storage.service import StorageService
from common_code.tasks.controller import router as tasks_router
from common_code.tasks.service import TasksService
from common_code.tasks.models import TaskData
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from contextlib import asynccontextmanager

# Imports required by the service's model
from utils import custom_parse_args, save_image, CustomEncoder
from common_code.tasks.service import get_extension
import cv2
from model.main_ import main as main_model
import shutil

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
            name="Layout Analysis service",
            slug="layout-analysis-service",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(
                    name="images",
                    type=[
                        FieldDescriptionType.IMAGE_JPEG
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
            vis_font_path="Fonts/Arial.ttf",
            use_gpu=False,
            image_dir="img_dir",
            layout_model_dir="model/inference/picodet_lcnet_x1_0_layout_infer",
            layout_dict_path="model/dict/layout_publaynet_dict.txt",
            output="../output",
            table=False,
            ocr=False,
        )

        # Execute main_model
        _, input_type = save_image(data)
        res, img = main_model(args)
        guessed_extension = get_extension(input_type)
        is_success, out_buff = cv2.imencode(guessed_extension, img)
        res = CustomEncoder().encode(res)

        shutil.rmtree("img_dir")
        shutil.rmtree("../output")

        # NOTE that the result must be a dictionary with the keys being the field names set in the data_out_fields
        return {
            "result_text": TaskData(data=res, type=FieldDescriptionType.APPLICATION_JSON),

            "result_img": TaskData(
                data=out_buff.tobytes(),
                type=input_type,
            )
        }


service_service: ServiceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Manual instances because startup events doesn't support Dependency Injection
    # https://github.com/tiangolo/fastapi/issues/2057
    # https://github.com/tiangolo/fastapi/issues/425

    # Global variable
    global service_service

    # Startup
    logger = get_logger(settings)
    http_client = HttpClient()
    storage_service = StorageService(logger)
    my_service = MyService()
    tasks_service = TasksService(logger, settings, http_client, storage_service)
    service_service = ServiceService(logger, settings, http_client, tasks_service)

    tasks_service.set_service(my_service)

    # Start the tasks service
    tasks_service.start()

    async def announce():
        retries = settings.engine_announce_retries
        for engine_url in settings.engine_urls:
            announced = False
            while not announced and retries > 0:
                announced = await service_service.announce_service(
                    my_service, engine_url
                )
                retries -= 1
                if not announced:
                    time.sleep(settings.engine_announce_retry_delay)
                    if retries == 0:
                        logger.warning(
                            f"Aborting service announcement after "
                            f"{settings.engine_announce_retries} retries"
                        )

    # Announce the service to its engine
    asyncio.ensure_future(announce())

    yield

    # Shutdown
    for engine_url in settings.engine_urls:
        await service_service.graceful_shutdown(my_service, engine_url)


api_description = """Layout Analysis service takes an input image and processes it to extract detailed structural information. 
The service generates output images annotated with bounding boxes (bboxes) that highlight detected elements, making it easy to visualize the layout. 
Additionally, it provides a JSON-formatted result that describes these detected regions, including their positions and attributes, 
facilitating seamless integration with data workflows and further analysis. 
Ideal for document processing, digitization, and automated layout understanding, 
this service ensures accuracy and clarity in presenting image-based layout insights.
"""
api_summary = """Layout Analysis detect part of an image-based document using PP-PicoDet 
"""

# Define the FastAPI application with information

app = FastAPI(
    lifespan=lifespan,
    title="My layout analysis service API.",
    description=api_description,
    version="0.0.1",
    contact={
        "name": "Swiss AI Center",
        "url": "https://swiss-ai-center.ch/",
        "email": "info@swiss-ai-center.ch",
    },
    swagger_ui_parameters={
        "tagsSorter": "alpha",
        "operationsSorter": "method",
    },
    license_info={
        "name": "GNU Affero General Public License v3.0 (GNU AGPLv3)",
        "url": "https://choosealicense.com/licenses/agpl-3.0/",
    },
)

# Include routers from other files
app.include_router(service_router, tags=["Service"])
app.include_router(tasks_router, tags=["Tasks"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs", status_code=301)
