
import cv2
import numpy as np
import sys
import torch 
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from nanodet.data.transform import PipelineInference
from nanodet.data.transform.color import _normalize


def load_tensor_image(image_path, cfg):
    img = cv2.imread(image_path)
    img_info = {"height": img.shape[0], "width": img.shape[1]}
    meta = dict(img_info=img_info, raw_img=img, img=img)

    pipeline = PipelineInference(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
    meta = pipeline(meta, cfg.data.val.input_size)
    img_tensor = torch.from_numpy(meta["img"].transpose(2, 0, 1)).unsqueeze(0)
    return img_tensor

def load_np_image(image_path):
    img = cv2.imread(image_path)
    img = img.astype(np.float32) / 255
    img = _normalize(img, mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395])
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis,...]
    return img

def generate_random_color():
    """Generate a random RGB color."""
    return [np.random.randint(0, 255) for _ in range(3)]


def vis_results(img, masks, bboxs, scores, mask_threshold=0.2, box_threshold=0.5):
    img_height, img_width, _ = img.shape

    for mask, bbox, score in zip(masks, bboxs,scores):
        x_min, y_min, x_max, y_max = map(int, bbox)

        if score < box_threshold:
            print("Filtering using box threshold")
            return img

        x_min, x_max = max(0, x_min), min(x_max+1, img_width)
        y_min, y_max = max(0, y_min), min(y_max+1, img_height)

        width, height = x_max - x_min, y_max - y_min

        mask = cv2.resize(mask[0,...], (width, height), interpolation = cv2.INTER_CUBIC)
        mask[mask < mask_threshold] = 0
        binary_mask = mask > 0

        color = generate_random_color()
        img[y_min:y_max, x_min:x_max][binary_mask.squeeze()] = color
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

    return img

def save_image(img, path):
    cv2.imwrite(path, img[...,::-1])


def unnormalize(img, mean, std):
    img = img.detach().squeeze(0).numpy()
    img = img.astype(np.float32)
    mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
    std = np.array(std, dtype=np.float32).reshape(-1, 1, 1)
    img = img * std + mean
    img = np.clip(img, 0, 255)  # Clip values to the range [0, 255]
    img = img.transpose(1,2,0).astype(np.uint8)
    return img