# Construct the command for Model Optimizer.
from pathlib import Path
import subprocess
import cv2
import numpy as np
import warnings
from utils import load_np_image,generate_random_color,vis_results,save_image


onnx_path="segmentor.onnx"
model_path="irmodel"
model_path = Path(model_path)
ir_path = model_path.with_suffix(".xml")
IMAGE_HEIGHT=512
IMAGE_WIDTH=512
 #--compress_to_fp16
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
img = load_np_image(image_path)
res = compiled_model_ir([img])

bboxs=res[output_layer_ir0]
masks=res[output_layer_ir1]
labels=res[output_layer_ir2]
scores=res[output_layer_ir3]

img = cv2.imread(image_path)
vis_img=vis_results(img, masks, bboxs, scores)
save_image(vis_img, "vis_results/onenvino.png")