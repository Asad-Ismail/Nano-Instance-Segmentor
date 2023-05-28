# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import copy
import io
import itertools
import json
import logging
import os
import warnings

import numpy as np
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate
import torch.nn.functional as F
import torch
import pycocotools.mask as mask_util
import matplotlib.pyplot as plt

logger = logging.getLogger("NanoDet")


def xyxy2xywh(bbox):
    """
    change bbox to coco format
    :param bbox: [x1, y1, x2, y2]
    :return: [x, y, w, h]
    """
    return [
        bbox[0],
        bbox[1],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
    ]


class CocoDetectionEvaluator:
    def __init__(self, dataset):
        assert hasattr(dataset, "coco_api")
        self.class_names = dataset.class_names
        self.coco_api = dataset.coco_api
        self.cat_ids = dataset.cat_ids
        self.metric_names = ["mAP", "AP_50", "AP_75", "AP_small", "AP_m", "AP_l"]

    def results2json(self, results):
        """
        results: {image_id: {label: [bboxes...] } }
        :return coco json format: {image_id:
                                   category_id:
                                   bbox:
                                   score: }
        """
        json_results = []
        for image_id, dets in results.items():
            for label, bboxes in dets.items():
                category_id = self.cat_ids[label]
                for bbox in bboxes:
                    score = float(bbox[4])
                    detection = dict(
                        image_id=int(image_id),
                        category_id=int(category_id),
                        bbox=xyxy2xywh(bbox),
                        score=score,
                    )
                    json_results.append(detection)
        return json_results

    def evaluate(self, results, save_dir, rank=-1):
        results_json = self.results2json(results)
        if len(results_json) == 0:
            warnings.warn(
                "Detection result is empty! Please check whether "
                "training set is too small (need to increase val_interval "
                "in config and train more epochs). Or check annotation "
                "correctness."
            )
            empty_eval_results = {}
            for key in self.metric_names:
                empty_eval_results[key] = 0
            return empty_eval_results
        json_path = os.path.join(save_dir, "results{}.json".format(rank))
        json.dump(results_json, open(json_path, "w"))
        coco_dets = self.coco_api.loadRes(json_path)
        coco_eval = COCOeval(
            copy.deepcopy(self.coco_api), copy.deepcopy(coco_dets), "bbox"
        )
        coco_eval.evaluate()
        coco_eval.accumulate()

        # use logger to log coco eval results
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            coco_eval.summarize()
        logger.info("\n" + redirect_string.getvalue())

        # print per class AP
        headers = ["class", "AP50", "mAP"]
        colums = 6
        per_class_ap50s = []
        per_class_maps = []
        precisions = coco_eval.eval["precision"]
        # dimension of precisions: [TxRxKxAxM]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(self.class_names) == precisions.shape[2]

        for idx, name in enumerate(self.class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision_50 = precisions[0, :, idx, 0, -1]
            precision_50 = precision_50[precision_50 > -1]
            ap50 = np.mean(precision_50) if precision_50.size else float("nan")
            per_class_ap50s.append(float(ap50 * 100))

            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            per_class_maps.append(float(ap * 100))

        num_cols = min(colums, len(self.class_names) * len(headers))
        flatten_results = []
        for name, ap50, mAP in zip(self.class_names, per_class_ap50s, per_class_maps):
            flatten_results += [name, ap50, mAP]

        row_pair = itertools.zip_longest(
            *[flatten_results[i::num_cols] for i in range(num_cols)]
        )
        table_headers = headers * (num_cols // len(headers))
        table = tabulate(
            row_pair,
            tablefmt="pipe",
            floatfmt=".1f",
            headers=table_headers,
            numalign="left",
        )
        logger.info("\n" + table)

        aps = coco_eval.stats[:6]
        eval_results = {}
        for k, v in zip(self.metric_names, aps):
            eval_results[k] = v
        return eval_results


class CocoSegmentationEvaluator:
    def __init__(self, dataset):
        assert hasattr(dataset, "coco_api")
        self.class_names = dataset.class_names
        self.coco_api = dataset.coco_api
        self.cat_ids = dataset.cat_ids
        self.metric_names = ["mAP", "AP_50", "AP_75", "AP_small", "AP_m", "AP_l"]

    def interpolate_masks(self,box,mask,height,width,mask_threshold=0.3):
        #Interpolate and paste masks at right place
        img = np.zeros((height,width), dtype=np.bool)
        x_min, y_min, x_max, y_max = box
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

        x_min, x_max = max(0, x_min), min(x_max+1, width)
        y_min, y_max = max(0, y_min), min(y_max+1, height)

        width, height = x_max - x_min, y_max - y_min

        mask = torch.from_numpy(mask).unsqueeze(0)
        mask = F.interpolate(mask, size=(height, width), mode='bicubic', align_corners=True)
        mask[mask < mask_threshold] = 0
        binary_mask = mask > 0
        binary_mask_np=binary_mask.squeeze(0).squeeze(0).cpu().numpy()
        binary_mask_np = binary_mask_np.astype(np.bool)
        ## set binary mask in original image
        img[y_min:y_max, x_min:x_max][binary_mask_np] = True

        return img
    

    def results2json(self, results):
        """
        results: {image_id: {label: [detections...] } }
        :return coco json format: {image_id:
                                   category_id:
                                   bbox:
                                   score:
                                   mask: 
                                    }
        """
        json_results = []
        mask_threshold=0.3
        #print(f"Original results keys(image ids) are {results.keys()}")
        for image_id, dets in results.items():
            #print(f"Dets keys(cat ids) are {dets.keys()}")
            for label, rslts in dets.items():
                category_id = self.cat_ids[label]
                #print(f"Length for results list are {len(rslts)}")
                for i,rslt in enumerate(rslts):
                    score = float(rslt["score"])
                    mask= np.array(rslt["mask"])
                    height, width = rslt["height"],rslt["width"]
                    bbox=rslt["bbox"]
                    interp_mask=self.interpolate_masks(bbox,mask,height,width)
                    # Convert mask to COCO RLE format
                    rle = mask_util.encode(np.asfortranarray(interp_mask))
                    rle['counts'] = rle['counts'].decode('utf-8')

                    detection = dict(
                        image_id=int(image_id),
                        category_id=int(category_id),
                        bbox=xyxy2xywh(bbox),  # Make sure this is in xywh format
                        score=score,
                        segmentation=rle
                    )
                    json_results.append(detection)
        return json_results

    def evaluate(self, results, save_dir, rank=-1):
        results_json = self.results2json(results)
        if len(results_json) == 0:
            warnings.warn(
                "Detection result is empty! Please check whether "
                "training set is too small (need to increase val_interval "
                "in config and train more epochs). Or check annotation "
                "correctness."
            )
            empty_eval_results = {}
            for key in self.metric_names:
                empty_eval_results[f'{key}_bbox'] = 0
                empty_eval_results[f'{key}_segm'] = 0
            return empty_eval_results
        json_path = os.path.join(save_dir, "results{}.json".format(rank))
        json.dump(results_json, open(json_path, "w"))
        coco_dets = self.coco_api.loadRes(json_path)

        # Evaluating bbox metrics
        coco_eval = COCOeval(
            copy.deepcopy(self.coco_api), copy.deepcopy(coco_dets), "bbox"
        )
        coco_eval.evaluate()
        coco_eval.accumulate()

        # Evaluating segm metrics
        coco_eval_segm = COCOeval(
            copy.deepcopy(self.coco_api), copy.deepcopy(coco_dets), "segm"
        )
        coco_eval_segm.evaluate()
        coco_eval_segm.accumulate()

        # use logger to log coco eval results
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            coco_eval.summarize()
            coco_eval_segm.summarize()
        logger.info("\n" + redirect_string.getvalue())

        # print per class AP
        headers = ["class", "bbox_AP50", "bbox_mAP", "segm_AP50", "segm_mAP"]
        columns = 6

        per_class_bbox_ap50s = []
        per_class_bbox_maps = []
        per_class_segm_ap50s = []
        per_class_segm_maps = []

        precisions_bbox = coco_eval.eval["precision"]
        precisions_segm = coco_eval_segm.eval["precision"]

        # dimension of precisions: [TxRxKxAxM]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(self.class_names) == precisions_bbox.shape[2] == precisions_segm.shape[2]

        for idx, name in enumerate(self.class_names):
            # BBOX metrics
            precision_50_bbox = precisions_bbox[0, :, idx, 0, -1]
            precision_50_bbox = precision_50_bbox[precision_50_bbox > -1]
            ap50_bbox = np.mean(precision_50_bbox) if precision_50_bbox.size else float("nan")
            per_class_bbox_ap50s.append(float(ap50_bbox * 100))

            precision_bbox = precisions_bbox[:, :, idx, 0, -1]
            precision_bbox = precision_bbox[precision_bbox > -1]
            ap_bbox = np.mean(precision_bbox) if precision_bbox.size else float("nan")
            per_class_bbox_maps.append(float(ap_bbox * 100))

            # SEGM metrics
            precision_50_segm = precisions_segm[0, :, idx, 0, -1]
            precision_50_segm = precision_50_segm[precision_50_segm > -1]
            ap50_segm = np.mean(precision_50_segm) if precision_50_segm.size else float("nan")
            per_class_segm_ap50s.append(float(ap50_segm * 100))

            precision_segm = precisions_segm[:, :, idx, 0, -1]
            precision_segm = precision_segm[precision_segm > -1]
            ap_segm = np.mean(precision_segm) if precision_segm.size else float("nan")
            per_class_segm_maps.append(float(ap_segm * 100))

        num_cols = min(columns, len(self.class_names) * len(headers))
        flatten_results = []
        for name, bbox_ap50, bbox_mAP, segm_ap50, segm_mAP in zip(
            self.class_names, per_class_bbox_ap50s, per_class_bbox_maps, per_class_segm_ap50s, per_class_segm_maps):
            flatten_results += [name, bbox_ap50, bbox_mAP, segm_ap50, segm_mAP]

        row_pair = itertools.zip_longest(*[flatten_results[i::num_cols] for i in range(num_cols)])
        table_headers = headers * (num_cols // len(headers))
        table = tabulate(
            row_pair,
            tablefmt="pipe",
            floatfmt=".1f",
            headers=table_headers,
            numalign="left",
        )
        logger.info("\n" + table)

        # Return eval results for both bbox and segm
        bbox_stats = coco_eval.stats[:6]
        segm_stats = coco_eval_segm.stats[:6]

        eval_results = {}
        for metric, bbox_stat, segm_stat in zip(self.metric_names, bbox_stats, segm_stats):
            eval_results[f'{metric}_bbox'] = bbox_stat
            eval_results[f'{metric}_segm'] = segm_stat

        return eval_results

