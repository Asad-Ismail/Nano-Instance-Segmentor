import cv2
import numpy as np
import onnxruntime
import warnings
from nanodet.data.transform.color import _normalize


def load_image(image_path):
    img = cv2.imread(image_path)
    img = img.astype(np.float32) / 255
    img = _normalize(img, mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395])
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis,...]
    return img

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

def visualize_masks(img, masks, boxes, scores):
    for mask, box, score in zip(masks, boxes, scores):
        img = apply_mask(img.copy(), mask, box, score)
    return img

def save_image(img, path):
    cv2.imwrite(path, img[...,::-1])

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
