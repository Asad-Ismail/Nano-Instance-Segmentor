# Construct the command for Model Optimizer.
from pathlib import Path
import subprocess
import cv2
import numpy as np
import warnings
from nanodet.data.transform.color import _normalize


def load_image(image_path):
    img = cv2.imread(image_path)
    img = img.astype(np.float32) / 255
    img = _normalize(img, mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395])
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis,...]
    return img

def generate_random_color():
    """Generate a random RGB color."""
    return [np.random.randint(0, 255) for _ in range(3)]

def save_image(img, path):
    cv2.imwrite(path, img[...,::-1])

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

onnx_path="segmentor.onnx"
model_path="irmodel"
model_path = Path(model_path)
ir_path = model_path.with_suffix(".xml")
IMAGE_HEIGHT=512
IMAGE_WIDTH=512
mo_command = f"""mo
                 --input_model "{onnx_path}"
                 --input_shape "[1,3, {IMAGE_HEIGHT}, {IMAGE_WIDTH}]"
                 --compress_to_fp16
                 --output_dir "{model_path}"
                 """
mo_command = " ".join(mo_command.split())
print("Model Optimizer command to convert the ONNX model to OpenVINO:")
print(mo_command)

print("Exporting ONNX model to IR... This may take a few minutes.")
mo_result = subprocess.run(mo_command, shell=True, capture_output=True, text=True)
print(mo_result.stdout)


ir_path=Path("irmodel/segmentor.xml")
from openvino.runtime import Core
ie = Core()
model_ir = ie.read_model(model=ir_path)
compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU")

# Get input and output layers.
output_layer_ir0 = compiled_model_ir.output(0)
output_layer_ir1 = compiled_model_ir.output(1)
output_layer_ir2 = compiled_model_ir.output(2)
output_layer_ir3 = compiled_model_ir.output(3)


image_path = "data/cucumbers/113.png"
img = load_image(image_path)
res = compiled_model_ir([img])

bboxs=res[output_layer_ir0]
masks=res[output_layer_ir1]
labels=res[output_layer_ir2]
scores=res[output_layer_ir3]

img = cv2.imread(image_path)
vis_img=vis_results(img, masks, bboxs, scores)
save_image(vis_img, "vis_results/onenvino.png")