import cv2
import torch
import numpy as np
import onnxruntime
from nanodet.data.transform import PipelineInference
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from nanodet.data.transform.color import _normalize
import onnxsim
import onnx
import warnings

image_path = "data/cucumbers/113.png"
img = cv2.imread(image_path)
img=_normalize(img, mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395])

torch_img=torch.tensor(img.transpose(2, 0, 1))

print(f"Torch Image min and max are {torch_img.min()},{torch_img.max()}")

print(f"Image min and max are {img.min()}, {img.max()}")