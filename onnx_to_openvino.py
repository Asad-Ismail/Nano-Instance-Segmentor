# Construct the command for Model Optimizer.
from pathlib import Path
onnx_path="tinynet.onnx"
model_path="irmodel"
model_path = Path(model_path)
ir_path = model_path.with_suffix(".xml")
IMAGE_HEIGHT=512
IMAGE_WIDTH=512
mo_command = f"""mo
                 --input_model "{onnx_path}"
                 --input_shape "[1,3, {IMAGE_HEIGHT}, {IMAGE_WIDTH}]"
                 --data_type FP32
                 --output_dir "{model_path}"
                 """
mo_command = " ".join(mo_command.split())
print("Model Optimizer command to convert the ONNX model to OpenVINO:")
print(mo_command)

print("Exporting ONNX model to IR... This may take a few minutes.")
mo_result = subprocess.run(mo_command, shell=True, capture_output=True, text=True)
print(mo_result.stdout)