import sys
import time
import warnings
from pathlib import Path
import cv2
import numpy as np
import torch
from openvino.runtime import Core
import onnxruntime
from pathlib import Path
from utils import load_np_image

image_path = "data/cucumbers/113.png"
normalized_input_image = load_np_image(image_path)

# Onnx model
onnx_path = "segmentor.onnx"
#openvinomodel
ir_path=Path("irmodel/segmentor.xml")

ie = Core()

print(f"Available devices on this machine are ")
devices = ie.available_devices
for device in devices:
    device_name = ie.get_property(device, "FULL_DEVICE_NAME")
    print(f"{device}: {device_name}")


## ONNX model in ONNX runtime
print(f"Loading ONNX model in ONNX runtime!!")
ort_session = onnxruntime.InferenceSession(onnx_path)
ort_inputs = {ort_session.get_inputs()[0].name: normalized_input_image}
print(f"Success!!")


## ONNX model in openvino runtime
# Load the network to OpenVINO Runtime.
print(f"Loading ONNX model in openvino runtime!!")
ie = Core()
model_onnx = ie.read_model(model=onnx_path)
compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")
output_layer_onnx = compiled_model_onnx.output(0)
# Run inference on the input image.
res_onnx = compiled_model_onnx([normalized_input_image])[output_layer_onnx]
print(f"Success!!")

# Openvino model in openvino runtime
# Load the network in OpenVINO Runtime.
print(f"Loading Openvino model in openvino runtime!!")
ie = Core()
model_ir = ie.read_model(model=ir_path)
compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU")
# Get input and output layers.
output_layer_ir = compiled_model_ir.output(0)
# Run inference on the input image.
res_ir = compiled_model_ir([normalized_input_image])[output_layer_ir]
print(f"Success!!")

## Runtime comparison
num_images = 1000



start = time.perf_counter()
for _ in range(num_images):
    ort_outs = ort_session.run(None, ort_inputs)
end = time.perf_counter()
time_onnx = end - start
print(
    f"ONNX model in ONNX Runtime/CPU: {time_onnx/num_images:.3f} "
    f"seconds per image, FPS: {num_images/time_onnx:.2f}"
)


start = time.perf_counter()
for _ in range(num_images):
    compiled_model_onnx([normalized_input_image])
end = time.perf_counter()
time_onnx = end - start
print(
    f"ONNX model in OpenVINO Runtime/CPU: {time_onnx/num_images:.3f} "
    f"seconds per image, FPS: {num_images/time_onnx:.2f}"
)

start = time.perf_counter()
for _ in range(num_images):
    compiled_model_ir([normalized_input_image])
end = time.perf_counter()
time_ir = end - start
print(
    f"OpenVINO IR model in OpenVINO Runtime/CPU: {time_ir/num_images:.3f} "
    f"seconds per image, FPS: {num_images/time_ir:.2f}"
)


"""
GPU is not working now :(

if "GPU" in ie.available_devices:
    compiled_model_onnx_gpu = ie.compile_model(model=model_onnx, device_name="GPU")
    start = time.perf_counter()
    for _ in range(num_images):
        compiled_model_onnx_gpu([normalized_input_image])
    end = time.perf_counter()
    time_onnx_gpu = end - start
    print(
        f"ONNX model in OpenVINO/GPU: {time_onnx_gpu/num_images:.3f} "
        f"seconds per image, FPS: {num_images/time_onnx_gpu:.2f}"
    )

    print(f"normalized input shape is {normalized_input_image.shape}")
    compiled_model_ir_gpu = ie.compile_model(model=model_ir, device_name="GPU")
    start = time.perf_counter()
    for _ in range(num_images):
        compiled_model_ir_gpu([normalized_input_image])
    end = time.perf_counter()
    time_ir_gpu = end - start
    print(
        f"IR model in OpenVINO/GPU: {time_ir_gpu/num_images:.3f} "
        f"seconds per image, FPS: {num_images/time_ir_gpu:.2f}"
    )

"""