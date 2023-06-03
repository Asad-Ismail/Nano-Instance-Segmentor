import cv2
import numpy as np
import onnxruntime
import warnings

def load_model(model_path):
    print(f"Loading ONNX model from {model_path}")
    ort_session = onnxruntime.InferenceSession(model_path)
    print(f"Loaded ONNX model!!")
    return ort_session

def run_inference(ort_session, img):
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    print(f"Running Inference!!")
    ort_outs = ort_session.run(None, ort_inputs)
    print(f"Finished running Inference!!")
    return ort_outs

# Load image
image_path = "data/cucumbers/113.png"
img = load_image(image_path)

# Load model and run inference
model_path = "segmentor.onnx"
ort_session = load_model(model_path)
bboxs, masks, labels, scores = run_inference(ort_session, img)
# Apply masks to image and save
img = cv2.imread(image_path)
img = vis_results(img, masks, bboxs, scores)
save_image(img, "vis_results/onnx.png")
