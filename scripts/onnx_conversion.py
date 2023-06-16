import cv2
import torch
import numpy as np
import onnxruntime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from utils import load_tensor_image
import onnxsim
import onnx
import warnings

warnings.filterwarnings('ignore') # Ignore warnings

# Configurations
cfg_path = "../config/nanoinstance-mask-512.yml"
model_path = "../workspace/nanodet-segmentor-pretrain_cucumber/model_last.ckpt"
out_path = "segmentor.onnx"
input_shape = (512, 512)
image_path = "../data/cucumbers/113.png"

assert os.path.exists(image_path), "Image does not exists!"


def load_model(cfg, model_path):
    logger = Logger(-1, cfg.save_dir, False)
    model = build_model(cfg.model)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    logger.log("Loading trained model weights!")
    load_model_weight(model, checkpoint, logger)
    model.eval()
    return model

def main():
    # Load configurations
    load_config(cfg, cfg_path)

    # Load and process image
    img_tensor = load_tensor_image(image_path, cfg)
    print(f"Loaded tensor shape is {img_tensor.shape}")

    # Load model
    model = load_model(cfg, model_path)
    # Run inference on PyTorch model
    torch_bbox, torch_msks, torch_labels, torch_scores = model.forward_onnx(img_tensor)

    print(f"Torch out shape is {torch_bbox.shape},{torch_msks.shape},{torch_labels.shape},{torch_scores.shape}")

    # Export model to ONNX
    torch.onnx.export(model,
                      img_tensor,
                      out_path,
                      keep_initializers_as_inputs=True,
                      do_constant_folding=True,
                      opset_version=17,
                      input_names=['images'],
                      output_names=["boxes", "masks", "labels", "scores"],
                      dynamic_axes={'images': {0: 'batch'}, 'scores': {0: 'batch'}, 
                                    'labels': {0: 'batch'}, 'boxes': {0: 'batch'}, 
                                    'masks': {0: 'batch'}})

    # Load ONNX model
    print(f"Loading ONNX model!!")
    ort_session = onnxruntime.InferenceSession("segmentor.onnx")
    print(f"Loaded ONNX model!!")

    # Run inference on ONNX model
    ort_inputs = {ort_session.get_inputs()[0].name: img_tensor.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    onnx_bbox, onnx_msks, onnx_labels, onnx_scores = ort_outs

    # Assert that all outputs have equal shape
    assert torch_bbox.shape == onnx_bbox.shape
    assert torch_msks.shape == onnx_msks.shape
    assert torch_labels.shape == onnx_labels.shape
    assert torch_scores.shape == onnx_scores.shape

    # Assert that all outputs are close
    assert np.allclose(torch_bbox.detach().numpy(), onnx_bbox, rtol=1e-03, atol=1e-05)
    assert np.allclose(torch_msks.detach().numpy(), onnx_msks, rtol=1e-03, atol=1e-05)
    assert np.allclose(torch_labels.detach().numpy(), onnx_labels, rtol=1e-03, atol=1e-05)
    assert np.allclose(torch_scores.detach().numpy(), onnx_scores, rtol=1e-03, atol=1e-05)

    # Print differences if shapes are not equal or outputs are not close
    if not np.allclose(torch_bbox.detach().numpy(), onnx_bbox, rtol=1e-03, atol=1e-05):
        print('Difference in bbox: ', np.abs(torch_bbox.detach().numpy() - onnx_bbox).max())

    if not np.allclose(torch_msks.detach().numpy(), onnx_msks, rtol=1e-03, atol=1e-05):
        print('Difference in masks: ', np.abs(torch_msks.detach().numpy() - onnx_msks).max())

    if not np.allclose(torch_labels.detach().numpy(), onnx_labels, rtol=1e-03, atol=1e-05):
        print('Difference in labels: ', np.abs(torch_labels.detach().numpy() - onnx_labels).max())

    if not np.allclose(torch_scores.detach().numpy(), onnx_scores, rtol=1e-03, atol=1e-05):
        print('Difference in scores: ', np.abs(torch_scores.detach().numpy() - onnx_scores).max())
    
    print(f"All tests succedded the conversion to ONNX was successful!!")
    
    print("start simplifying onnx ")
    input_data = {"images": img_tensor.detach().cpu().numpy()}
    model_sim, flag = onnxsim.simplify(out_path, input_data=input_data)
    if flag:
        onnx.save(model_sim, out_path)
        print("simplify onnx successfully")
    else:
        print("simplify onnx failed")


    # Load Simplified ONNX model
    print(f"Loading Simplified ONNX model!!")
    ort_session = onnxruntime.InferenceSession("segmentor.onnx")
    print(f"Loaded Simplified ONNX model!!")

    # Run inference on ONNX model
    ort_inputs = {ort_session.get_inputs()[0].name: img_tensor.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    onnx_bbox, onnx_msks, onnx_labels, onnx_scores = ort_outs

    # Assert that all outputs have equal shape
    assert torch_bbox.shape == onnx_bbox.shape
    assert torch_msks.shape == onnx_msks.shape
    assert torch_labels.shape == onnx_labels.shape
    assert torch_scores.shape == onnx_scores.shape

    # Assert that all outputs are close
    assert np.allclose(torch_bbox.detach().numpy(), onnx_bbox, rtol=1e-03, atol=1e-05)
    assert np.allclose(torch_msks.detach().numpy(), onnx_msks, rtol=1e-03, atol=1e-05)
    assert np.allclose(torch_labels.detach().numpy(), onnx_labels, rtol=1e-03, atol=1e-05)
    assert np.allclose(torch_scores.detach().numpy(), onnx_scores, rtol=1e-03, atol=1e-05)

    # Print differences if shapes are not equal or outputs are not close
    if not np.allclose(torch_bbox.detach().numpy(), onnx_bbox, rtol=1e-03, atol=1e-05):
        print('Difference in bbox: ', np.abs(torch_bbox.detach().numpy() - onnx_bbox).max())

    if not np.allclose(torch_msks.detach().numpy(), onnx_msks, rtol=1e-03, atol=1e-05):
        print('Difference in masks: ', np.abs(torch_msks.detach().numpy() - onnx_msks).max())

    if not np.allclose(torch_labels.detach().numpy(), onnx_labels, rtol=1e-03, atol=1e-05):
        print('Difference in labels: ', np.abs(torch_labels.detach().numpy() - onnx_labels).max())

    if not np.allclose(torch_scores.detach().numpy(), onnx_scores, rtol=1e-03, atol=1e-05):
        print('Difference in scores: ', np.abs(torch_scores.detach().numpy() - onnx_scores).max())
    
    print(f"All tests succedded the conversion to ONNX Simplify was successful!!")


if __name__ == "__main__":
    main()
