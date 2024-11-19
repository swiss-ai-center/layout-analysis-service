# Layout Analysis Service

## Overview

The Layout Analysis Service is a high-performance system built on the Swiss AI Center Core Engine, designed for analyzing the layout of image-based documents. It takes an image-based document as input and provides:

    A JSON file containing:
        The identified components/parts of the document.
        Their bounding boxes (bbox).
    An annotated image:
        Visualizing the bounding boxes.
        Displaying the names of the identified parts.

This service is inspired by state-of-the-art research and optimized for real-time performance on mobile and CPU devices, leveraging PaddlePaddle's innovative detection architecture.

## References


[PP-PicoDet: A better real-time object detector on mobile devices](https://github.com/PaddlePaddle/PaddleDetection)

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

_Check the [related documentation](https://docs.swiss-ai-center.ch/reference/core-concepts/service/) for more information._
