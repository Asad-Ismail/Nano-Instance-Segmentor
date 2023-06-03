import sys
import time
import warnings
from pathlib import Path
import cv2
import numpy as np
import torch
from openvino.runtime import Core


ie = Core()

print(f"Available devices on this machine are ")
devices = ie.available_devices
for device in devices:
    device_name = ie.get_property(device, "FULL_DEVICE_NAME")
    print(f"{device}: {device_name}")


## ONNX model in openvino runtime
# Load the network to OpenVINO Runtime.
ie = Core()
model_onnx = ie.read_model(model=onnx_path)
compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")
output_layer_onnx = compiled_model_onnx.output(0)
# Run inference on the input image.
res_onnx = compiled_model_onnx([normalized_input_image])[output_layer_onnx]

# Openvino model in openvino runtime
# Load the network in OpenVINO Runtime.
ie = Core()
model_ir = ie.read_model(model=ir_path)
compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU")

# Get input and output layers.
output_layer_ir = compiled_model_ir.output(0)

# Run inference on the input image.
res_ir = compiled_model_ir([normalized_input_image])[output_layer_ir]

## Runtime comparison
num_images = 100

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
    compiled_model_ir([input_image])
end = time.perf_counter()
time_ir = end - start
print(
    f"OpenVINO IR model in OpenVINO Runtime/CPU: {time_ir/num_images:.3f} "
    f"seconds per image, FPS: {num_images/time_ir:.2f}"
)

if "GPU" in ie.available_devices:
    compiled_model_onnx_gpu = ie.compile_model(model=model_onnx, device_name="GPU")
    start = time.perf_counter()
    for _ in range(num_images):
        compiled_model_onnx_gpu([input_image])
    end = time.perf_counter()
    time_onnx_gpu = end - start
    print(
        f"ONNX model in OpenVINO/GPU: {time_onnx_gpu/num_images:.3f} "
        f"seconds per image, FPS: {num_images/time_onnx_gpu:.2f}"
    )

    compiled_model_ir_gpu = ie.compile_model(model=model_ir, device_name="GPU")
    start = time.perf_counter()
    for _ in range(num_images):
        compiled_model_ir_gpu([input_image])
    end = time.perf_counter()
    time_ir_gpu = end - start
    print(
        f"IR model in OpenVINO/GPU: {time_ir_gpu/num_images:.3f} "
        f"seconds per image, FPS: {num_images/time_ir_gpu:.2f}"
    )