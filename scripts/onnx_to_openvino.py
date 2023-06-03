import cv2
import subprocess
import numpy as np
from pathlib import Path
from openvino.runtime import Core
from utils import load_np_image, generate_random_color, vis_results, save_image


def convert_model_to_ir(onnx_path, model_path, image_height, image_width):
    mo_command = f"""
        mo
        --input_model "{onnx_path}"
        --input_shape "[1,3, {image_height}, {image_width}]"
        --compress_to_fp16
        --output_dir "{model_path}"
    """
    mo_command = " ".join(mo_command.split())
    print("Model Optimizer command to convert the ONNX model to OpenVINO:")
    print(mo_command)
    print("Exporting ONNX model to IR... This may take a few minutes.")
    mo_result = subprocess.run(mo_command, shell=True, capture_output=True, text=True)
    print(mo_result.stdout)
    return mo_result


def load_model_into_openvino(ir_path, device="CPU"):
    ie = Core()
    model_ir = ie.read_model(ir_path)
    compiled_model_ir = ie.compile_model(model=model_ir, device_name=device)
    return compiled_model_ir, ie


def run_inference_on_image(compiled_model, image):
    output_layers = [compiled_model.output(i) for i in range(4)]
    result = compiled_model([image])
    results = dict(zip(['bboxs', 'masks', 'labels', 'scores'], output_layers))
    return {key: result[value] for key, value in results.items()}



def visualize_and_save_results(image_path, results, save_path):
    img = cv2.imread(image_path)
    vis_img = vis_results(img, results['masks'], results['bboxs'], results['scores'])
    save_image(vis_img, save_path)


def main():
    IMAGE_HEIGHT, IMAGE_WIDTH = 512, 512
    onnx_path = "segmentor.onnx"
    model_path = "irmodel"

    convert_model_to_ir(onnx_path, model_path, IMAGE_HEIGHT, IMAGE_WIDTH)

    ir_path = Path("irmodel/segmentor.xml")
    compiled_model_ir, ie = load_model_into_openvino(ir_path)

    image_path = "data/cucumbers/113.png"
    img = load_np_image(image_path)

    results = run_inference_on_image(compiled_model_ir, img)
    visualize_and_save_results(image_path, results, "vis_results/onenvino.png")


if __name__ == "__main__":
    main()
