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
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
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
from utils import vis_results,generate_random_color,unnormalize,unnormalize_simple,save_image
import deeplake as hub

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

def log_to_activeloop(img,gt_masks,gt_boxes,gt_labels,pred_masks,pred_boxes,pred_labels):

    ds = hub.empty('hub://aismail2/pepper_evaluate',overwrite=True)

    class_map={"pepper":0}
    class_names=["pepper"]

    with ds:

        ds.create_tensor('images', htype='image',sample_compression = 'jpeg')
        ds.create_tensor('boxes', htype='bbox')
        ds.boxes.info.update(coords = {'type': 'pixel', 'mode': 'LTRB'})
        ds.create_tensor('labels', htype='class_label', class_names = class_names)
        ds.create_tensor('masks', htype = 'binary_mask', sample_compression = 'lz4')

        ds.create_group('model_evaluation')
        ds.model_evaluation.create_tensor('labels', htype = 'class_label', class_names = ds.labels.info.class_names,exist_ok=True)
        ds.model_evaluation.create_tensor('boxes', htype = 'bbox', coords = {'type': 'pixel', 'mode': 'LTRB'},exist_ok=True)
        ds.model_evaluation.create_tensor('masks', htype = 'binary_mask', sample_compression = 'lz4',exist_ok=True)
        
        h,w,_=img.shape
        all_masks=[]
        all_boxes=[]

        for i,msk in enumerate(pred_masks):
            bx=pred_boxes[i]
            z_img=np.zeros((h,w),dtype=bool)
            z_img[bx[1]:bx[3],bx[0]:bx[2]]=msk==True
            all_masks.append(z_img)
            all_boxes.append(bx)
        
        f_bxs=np.array(all_boxes)
        f_msks=np.array(all_masks)
        f_msks=np.moveaxis(f_msks, 0, -1)
        pred_labels=np.array(pred_labels)


        gt_masks=np.moveaxis(gt_masks.detach().numpy(), 0, -1)
        ## Append ground truth     
        ds.images.append(img)
        ds.labels.append(gt_labels.astype(np.uint32))
        ds.boxes.append(gt_boxes.astype(np.float32))
        ds.masks.append(gt_masks.astype(bool))
            
        # Append predictions
        ds.model_evaluation.masks.append(f_msks.astype(bool))
        ds.model_evaluation.boxes.append(f_bxs.astype(np.float32))
        ds.model_evaluation.labels.append(pred_labels.astype(np.uint32))
    
    
    ds.commit('Added model predictions.')
    print(f"All done!!")
    #cv2.imwrite("ds_img.png",ds_img)
    #cv2.imwrite("im.png",img)

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
val_dataset = build_dataset(cfg.data.val, "val", class_names=cfg.class_names)

print(f"Length of datast is {len(val_dataset)}")

val_dataloader = DataLoader(
    val_dataset,
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
#model_resume_path = os.path.join(cfg.save_dir, "model_last.ckpt")
model_resume_path = os.path.join(cfg.save_dir,"model_best","model_best.ckpt")
print(f"Loading model weights {model_resume_path}!!!")
task.load_state_dict(torch.load(model_resume_path)["state_dict"]) 
task.eval()

evaluator = build_evaluator(cfg.evaluator,val_dataset)


for i,batch in enumerate(val_dataloader):
    all_masks=[]
    all_lbl=[]
    all_boxes=[]
    with torch.no_grad():
        predictions = task.predict(batch)
        eval_results = evaluator.evaluate(predictions, cfg.save_dir)
        for k,v in predictions.items():
            for clas,preds in v.items():
                bboxes=[item["bbox"] for item in preds]
                masks=[np.array(item["mask"]) for item in preds]
                scores=[item["score"] for item in preds]
                raw_img=unnormalize(batch["img"], *cfg["data"]["train"]["pipeline"]["normalize"])
                #raw_img=unnormalize_simple(batch["img"])
                vis_img,fil_msks,fil_boxes,fil_labels=vis_results(raw_img.copy(),masks,bboxes,scores)
                
                all_masks.extend(fil_msks)
                all_boxes.extend(fil_boxes)
                all_lbl.append(fil_labels)
                save_image(vis_img, f"vis_results/pepper/vis{i}.png")
    log_to_activeloop(raw_img,batch["gt_masks"][0],batch["gt_bboxes"][0],batch["gt_labels"][0],all_masks, all_boxes, all_lbl)
    break
    #print(eval_results)