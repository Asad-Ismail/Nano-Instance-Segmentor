import argparse
import copy
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import rank_zero_only
import pytorch_lightning as pl
import torch
from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.model.arch import build_model
from nanodet.evaluator import build_evaluator
from nanodet.data.batch_process import stack_batch_img
from nanodet.util import (
    cfg,
    load_config,
    load_model_weight,
    mkdir,
    env_utils,
)
from typing import Dict, Any
from nanodet.model.weight_averager import build_weight_averager
import numpy as np
import cv2

# Configurations
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default="config/nanoinstance-512.yml", help="train config file path")
    parser.add_argument("--local_rank", default=-1, type=int, help="node rank for distributed training")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()
    return args

args = parse_args()

# Load configuration
load_config(cfg, args.config)

# Consistency check
if cfg.model.arch.head.num_classes != len(cfg.class_names):
    raise ValueError(
        f"cfg.model.arch.head.num_classes must equal len(cfg.class_names), "
        f"but got {cfg.model.arch.head.num_classes} and {len(cfg.class_names)}"
    )

# Prepare data
print("Setting up data...")
train_dataset = build_dataset(cfg.data.train, "train", class_names=cfg.class_names)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=cfg.device.workers_per_gpu,
    pin_memory=True,
    collate_fn=naive_collate,
    drop_last=True,
)

class TrainingTask(LightningModule):
    """
    Pytorch Lightning module of a general training task.
    Including training, evaluating and testing.
    Args:
        cfg: Training configurations
        evaluator: Evaluator for evaluating the model performance.
    """

    def __init__(self, cfg, evaluator=None):
        super(TrainingTask, self).__init__()
        self.cfg = cfg
        self.model = build_model(cfg.model)
        self.weight_averager = None
        if "weight_averager" in cfg.model:
            self.weight_averager = build_weight_averager(
                cfg.model.weight_averager, device=self.device
            )
            self.avg_model = copy.deepcopy(self.model)

    def _preprocess_batch_input(self, batch):
        batch_imgs = batch["img"]
        if isinstance(batch_imgs, list):
            batch_imgs = [img.to(self.device) for img in batch_imgs]
            batch_img_tensor = stack_batch_img(batch_imgs, divisible=32)
            batch["img"] = batch_img_tensor
        # Convert masks to torch tensors
        if "gt_masks" in batch:
            gt_masks= batch["gt_masks"]
            if isinstance(batch_imgs, list):
                batch_masks = [torch.from_numpy(mask).to(self.device) for mask in gt_masks]
                batch["gt_masks"]=batch_masks
        return batch

    def forward(self, x):
        x = self.model(x)
        return x

    @torch.no_grad()
    def predict(self, batch, batch_idx=None, dataloader_idx=None):
        batch = self._preprocess_batch_input(batch)
        results=self.model.inference(batch)
        return results

    def on_load_checkpoint(self, checkpointed_state: Dict[str, Any]) -> None:
        if self.weight_averager:
            avg_params = convert_avg_params(checkpointed_state)
            if len(avg_params) != len(self.model.state_dict()):
                self.logger.info(
                    "Weight averaging is enabled but average state does not"
                    "match the model"
                )
            else:
                self.weight_averager = build_weight_averager(
                    self.cfg.model.weight_averager, device=self.device
                )
                self.weight_averager.load_state_dict(avg_params)
                self.logger.info("Loaded average state from checkpoint.")


task = TrainingTask(cfg)

# Load model
model_resume_path = os.path.join(cfg.save_dir, "model_last.ckpt")
print(f"Loading model weights {model_resume_path}!!!")
task.load_state_dict(torch.load(model_resume_path)["state_dict"]) 
task.eval()

def generate_random_color():
    """Generate a random RGB color."""
    return [np.random.randint(0, 255) for _ in range(3)]

def vis_masks(img, masks, boxes,scores,mask_threshold=0.2, box_threshold=0.5):
    height, width, _ = img.shape

    for mask, box, score in zip(masks, boxes,scores):
        x_min, y_min, x_max, y_max = box

        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

        if score < box_threshold:
            print("Filtering using box threshold")
            continue

        x_min, x_max = max(0, x_min), min(x_max+1, width)
        y_min, y_max = max(0, y_min), min(y_max+1, height)

        width, height = x_max - x_min, y_max - y_min

        mask = mask.unsqueeze(0)
        mask = F.interpolate(mask, size=(height, width), mode='bicubic', align_corners=True)
        mask[mask < mask_threshold] = 0
        binary_mask = mask > 0

        color = generate_random_color()
        img[y_min:y_max, x_min:x_max][binary_mask.squeeze()] = color
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

    return img


def unnormalize(img, mean, std):
    img = img.detach().squeeze(0).numpy()
    img = img.astype(np.float32)
    mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
    std = np.array(std, dtype=np.float32).reshape(-1, 1, 1)
    img = img * std + mean
    img = np.clip(img, 0, 255)  # Clip values to the range [0, 255]
    img = img.transpose(1,2,0).astype(np.uint8)
    return img


evaluator = build_evaluator(cfg.evaluator,train_dataset)

for i,batch in enumerate(train_dataloader):
    with torch.no_grad():
        predictions = task.predict(batch)
        eval_results = evaluator.evaluate(predictions, cfg.save_dir)
        for k,v in predictions.items():
            for clas,preds in v.items():
                bboxes=[item["bbox"] for item in preds]
                masks=[torch.from_numpy(np.array(item["mask"])) for item in preds]
                scores=[item["score"] for item in preds]
                raw_img=unnormalize(batch["img"], *cfg["data"]["train"]["pipeline"]["normalize"])
                vis_img=vis_masks(raw_img.copy(),masks,bboxes,scores)
                #print(raw_img.shape)
                #cv2.imwrite("kk.png",raw_img)
                cv2.imwrite(f"vis_results/vis{i}.png",vis_img)
