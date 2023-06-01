import cv2
import torch
import onnxsim
import onnx
import numpy as np
import onnxruntime
from nanodet.data.transform import PipelineInference
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from nanodet.data.transform.color import _normalize
from torch.nn import functional as F
import warnings

image_path = "data/cucumbers/113.png"
org_img = cv2.imread(image_path)
img=org_img.copy()
img = org_img.astype(np.float32) / 255
img=_normalize(img, mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395])
img=img.transpose(2, 0, 1)
img=img[np.newaxis,...]
print(img.shape)

# Load ONNX model
print(f"Loading ONNX model!!")
ort_session = onnxruntime.InferenceSession("segmentor.onnx")
print(f"Loaded ONNX model!!")

# Run inference on ONNX model
ort_inputs = {ort_session.get_inputs()[0].name: img}

print(f"Running Inference!!")
ort_outs = ort_session.run(None, ort_inputs)
print(f"Finished running Inference!!")

onnx_bbox, onnx_msks, onnx_labels, onnx_scores = ort_outs

def generate_random_color():
    """Generate a random RGB color."""
    return [np.random.randint(0, 255) for _ in range(3)]

def vis_masks(img, masks, boxes,scores,mask_threshold=0.2, box_threshold=0.5):
    img_height, img_width, _ = img.shape

    for mask, box, score in zip(masks, boxes,scores):
        x_min, y_min, x_max, y_max = box

        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

        if score < box_threshold:
            print("Filtering using box threshold")
            continue
        
        x_min, x_max = max(0, x_min), min(x_max+1, img_width)
        y_min, y_max = max(0, y_min), min(y_max+1, img_height)

        width, height = x_max - x_min, y_max - y_min
        mask = torch.tensor(mask).unsqueeze(0)
        mask = F.interpolate(mask, size=(height, width), mode='bicubic', align_corners=True)
        mask[mask < mask_threshold] = 0
        binary_mask = mask > 0

        color = generate_random_color()
        img[y_min:y_max, x_min:x_max][binary_mask.squeeze()] = color
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

    return img


visimg=vis_masks(org_img.copy(), onnx_msks, onnx_bbox, onnx_scores)
cv2.imwrite(f"onnx.png",visimg)