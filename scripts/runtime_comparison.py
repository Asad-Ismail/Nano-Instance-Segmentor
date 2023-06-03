import sys
import time
from pathlib import Path
import onnxruntime
from openvino.runtime import Core
from utils import load_np_image

def get_device_info(runtime_engine: Core):
    print("Available devices on this machine are:")
    devices = runtime_engine.available_devices
    for device in devices:
        device_name = runtime_engine.get_property(device, "FULL_DEVICE_NAME")
        print(f"{device}: {device_name}")


def load_model_in_openvino(runtime_engine: Core, model_path: Path, device: str):
    print(f"Loading model from {model_path} in openvino runtime!!")
    model = runtime_engine.read_model(model=model_path)
    compiled_model = runtime_engine.compile_model(model=model, device_name=device)
    output_layer = compiled_model.output(0)
    print("Success!!")
    return compiled_model, output_layer


def load_model_in_onnxruntime(model_path: str):
    print(f"Loading ONNX model in ONNX runtime!!")
    session = onnxruntime.InferenceSession(model_path)
    print("Success!!")
    return session


def run_inference(compiled_model, input_image, output_layer=None, num_images=1000):
    start = time.perf_counter()
    for _ in range(num_images):
        if output_layer:
            compiled_model([input_image])[output_layer]
        else:
            compiled_model.run(None, {"images": input_image})
    end = time.perf_counter()
    elapsed_time = end - start
    return elapsed_time


def print_inference_stats(model_name: str, device: str, elapsed_time: float, num_images: int):
    print(
        f"{model_name} in {device}: {elapsed_time/num_images:.3f} "
        f"seconds per image, FPS: {num_images/elapsed_time:.2f}"
    )

def main():
    # Set paths
    image_path = "data/cucumbers/113.png"
    onnx_path = "segmentor.onnx"
    ir_path = Path("irmodel/segmentor.xml")
    
    # Load image
    normalized_input_image = load_np_image(image_path)
    
    # Initialize runtime engine
    runtime_engine = Core()
    
    # Print device info
    get_device_info(runtime_engine)

    # Load ONNX model in ONNX Runtime
    ort_session = load_model_in_onnxruntime(onnx_path)

    # Load ONNX model in OpenVINO
    compiled_model_onnx, output_layer_onnx = load_model_in_openvino(runtime_engine, onnx_path, "CPU")

    # Load OpenVINO model in OpenVINO
    compiled_model_ir, output_layer_ir = load_model_in_openvino(runtime_engine, ir_path, "CPU")

    # Run inferences and print stats
    num_images = 1000
    elapsed_time = run_inference(ort_session, normalized_input_image, num_images=num_images)
    print_inference_stats("ONNX model", "ONNX Runtime/CPU", elapsed_time, num_images)

    elapsed_time = run_inference(compiled_model_onnx, normalized_input_image, output_layer_onnx, num_images=num_images)
    print_inference_stats("ONNX model", "OpenVINO Runtime/CPU", elapsed_time, num_images)

    elapsed_time = run_inference(compiled_model_ir, normalized_input_image, output_layer_ir, num_images=num_images)
    print_inference_stats("OpenVINO IR model", "OpenVINO Runtime/CPU", elapsed_time, num_images)

    """
    GPU is not working now :(
    
    if "GPU" in runtime_engine.available_devices:
        compiled_model_onnx_gpu, _ = load_model_in_openvino(runtime_engine, onnx_path, "GPU")
        elapsed_time = run_inference(compiled_model_onnx_gpu, normalized_input_image, output_layer_onnx, num_images=num_images)
        print_inference_stats("ONNX model", "OpenVINO/GPU", elapsed_time, num_images)

        compiled_model_ir_gpu, _ = load_model_in_openvino(runtime_engine, ir_path, "GPU")
        elapsed_time = run_inference(compiled_model_ir_gpu, normalized_input_image, output_layer_ir, num_images=num_images)
        print_inference_stats("IR model", "OpenVINO/GPU", elapsed_time, num_images)
    """

if __name__ == "__main__":
    main()