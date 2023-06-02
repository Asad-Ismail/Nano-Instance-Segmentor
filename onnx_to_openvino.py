# Construct the command for Model Optimizer.
from pathlib import Path
onnx_path="segmentor.onnx"
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


# Get input and output layers.
output_layer_ir0 = compiled_model_ir.output(0)
output_layer_ir1 = compiled_model_ir.output(1)
output_layer_ir2 = compiled_model_ir.output(2)
output_layer_ir3 = compiled_model_ir.output(3)

res = compiled_model_ir([input_image])


res_ir0 = compiled_model_ir([input_image])[output_layer_ir0]
res_ir1 = compiled_model_ir([input_image])[output_layer_ir1]
res_ir2 = compiled_model_ir([input_image])[output_layer_ir2]
res_ir3 = compiled_model_ir([input_image])[output_layer_ir3]


print(res_ir0.shape)