import os
import sys
import cv2
import json
import numpy as np
import time
import logging
from copy import deepcopy


from paddleocr.ppocr.utils.logging import get_logger
from paddleocr.tools.infer.predict_system import TextSystem
from paddleocr.ppstructure.layout.predict_layout import LayoutPredictor
from paddleocr.ppstructure.utility import draw_structure_result, cal_ocr_word_box
import tempfile

from PIL import Image

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

logger = get_logger()


class StructureSystem(object):
    def __init__(self, args):
        self.mode = args.mode
        self.recovery = args.recovery

        if self.mode == "structure":
            if not args.show_log:
                logger.setLevel(logging.INFO)
            if not args.layout and args.ocr:
                args.ocr = False
                logger.warning(
                    "When args.layout is false, args.ocr is automatically set to false"
                )
            # init model
            self.layout_predictor = None
            self.text_system = None
            self.formula_system = None
            if args.layout:
                self.layout_predictor = LayoutPredictor(args)
                if args.ocr:
                    self.text_system = TextSystem(args)
        self.return_word_box = args.return_word_box

    def __call__(self, img, return_ocr_result_in_table=False, img_idx=0):
        time_dict = {
            "layout": 0,
            "table_match": 0,
            "det": 0,
            "rec": 0,
            "all": 0,
        }
        start = time.time()

        if self.mode == "structure":
            ori_im = img.copy()
            if self.layout_predictor is not None:
                layout_res, elapse = self.layout_predictor(img)
                time_dict["layout"] += elapse
            else:
                h, w = ori_im.shape[:2]
                layout_res = [dict(bbox=None, label="table", score=0.0)]

            text_res = None
            if self.text_system is not None:
                text_res, ocr_time_dict = self._predict_text(img)
                time_dict["det"] += ocr_time_dict["det"]
                time_dict["rec"] += ocr_time_dict["rec"]

            res_list = []
            for region in layout_res:
                res = ""
                if region["bbox"] is not None:
                    x1, y1, x2, y2 = region["bbox"]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    roi_img = ori_im[y1:y2, x1:x2, :]
                else:
                    x1, y1, x2, y2 = 0, 0, w, h
                    roi_img = ori_im
                bbox = [x1, y1, x2, y2]

                res_list.append(
                    {
                        "type": region["label"].lower(),
                        "bbox": bbox,
                        "img": roi_img,
                        "res": res,
                        "img_idx": img_idx,
                        "score": region["score"],
                    }
                )

            end = time.time()
            time_dict["all"] = end - start
            return res_list, time_dict

        return None, None

    def _predict_text(self, img):
        filter_boxes, filter_rec_res, ocr_time_dict = self.text_system(img)

        # remove style char,
        # when using the recognition model trained on the PubtabNet dataset,
        # it will recognize the text format in the table, such as <b>
        style_token = [
            "<strike>",
            "<strike>",
            "<sup>",
            "</sub>",
            "<b>",
            "</b>",
            "<sub>",
            "</sup>",
            "<overline>",
            "</overline>",
            "<underline>",
            "</underline>",
            "<i>",
            "</i>",
        ]
        res = []
        for box, rec_res in zip(filter_boxes, filter_rec_res):
            rec_str, rec_conf = rec_res[0], rec_res[1]
            for token in style_token:
                if token in rec_str:
                    rec_str = rec_str.replace(token, "")
            if self.return_word_box:
                word_box_content_list, word_box_list = cal_ocr_word_box(
                    rec_str, box, rec_res[2]
                )
                res.append(
                    {
                        "text": rec_str,
                        "confidence": float(rec_conf),
                        "text_region": box.tolist(),
                        "text_word": word_box_content_list,
                        "text_word_region": word_box_list,
                    }
                )
            else:
                res.append(
                    {
                        "text": rec_str,
                        "confidence": float(rec_conf),
                        "text_region": box.tolist(),
                    }
                )
        return res, ocr_time_dict

    def _filter_text_res(self, text_res, bbox):
        res = []
        for r in text_res:
            box = r["text_region"]
            rect = box[0][0], box[0][1], box[2][0], box[2][1]
            if self._has_intersection(bbox, rect):
                res.append(r)
        return res

    def _has_intersection(self, rect1, rect2):
        x_min1, y_min1, x_max1, y_max1 = rect1
        x_min2, y_min2, x_max2, y_max2 = rect2
        if x_min1 > x_max2 or x_max1 < x_min2:
            return False
        if y_min1 > y_max2 or y_max1 < y_min2:
            return False
        return True


def save_structure_res(res, save_folder, img_name, img_idx=0):
    excel_save_folder = os.path.join(save_folder, img_name)
    os.makedirs(excel_save_folder, exist_ok=True)
    res_cp = deepcopy(res)
    # save res
    with open(
            os.path.join(excel_save_folder, "res_{}.txt".format(img_idx)),
            "w",
            encoding="utf8",
    ) as f:
        for region in res_cp:
            region.pop("img")
            region.pop("res")
            region.pop("img_idx")
            f.write("{}\n".format(json.dumps(region)))


def load_structure_res(output_folder, img_name, img_idx=0):
    save_folder = os.path.join(output_folder, "structure")
    # Construct the path to the .txt file
    res_file_path = os.path.join(save_folder, img_name, f"res_{img_idx}.txt")

    if not os.path.exists(res_file_path):
        raise FileNotFoundError(f"The file {res_file_path} does not exist.")

    # Read and load the content
    results = []
    with open(res_file_path, 'r', encoding='utf8') as f:
        for line in f:
            region = json.loads(line.strip())
            results.append(region)

    img_save_path = os.path.join(
        save_folder, img_name, "show_{}.jpg".format(img_idx)
    )

    img = cv2.imread(img_save_path)

    return results, img


def main(args, img):
    if not args.use_pdf2docx_api:
        structure_sys = StructureSystem(args)
        temp_dir = tempfile.TemporaryDirectory()
        save_folder = os.path.join(temp_dir.name, structure_sys.mode)
        os.makedirs(save_folder, exist_ok=True)

        img_name = "image"
        index = 0

        res, time_dict = structure_sys(img, img_idx=index)
        img_save_path = os.path.join(
            save_folder, img_name, "show_{}.jpg".format(index)
        )
        os.makedirs(os.path.join(save_folder, img_name), exist_ok=True)
        if structure_sys.mode == "structure" and res != []:
            draw_img = draw_structure_result(img, res, font_path=args.vis_font_path)

            # Convert the NumPy array to a PIL image
            if isinstance(draw_img, np.ndarray):
                draw_img = Image.fromarray(draw_img)

            # Get the dimensions of the composite image
            width, height = draw_img.size

            # Calculate the midpoint (to divide into two parts)
            midpoint = width // 2

            # Crop the left part (the annotated original image)
            left_part = draw_img.crop((0, 0, midpoint, height))

            # Convert the cropped image back to a NumPy array for further processing
            left_part_np = np.array(left_part)

            save_structure_res(res, save_folder, img_name, index)
        if res != []:
            cv2.imwrite(img_save_path, left_part_np)
            logger.info("result save to {}".format(img_save_path))
    logger.info("Predict time : {:.3f}s".format(time_dict["all"]))

    res_, img_ = load_structure_res(temp_dir.name, img_name)

    temp_dir.cleanup()

    return res_, img_
