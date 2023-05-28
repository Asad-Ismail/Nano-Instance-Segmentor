import torch
import torch.nn as nn
from torchvision.ops import roi_align
import numpy as np

from ..module.conv import ConvModule, DepthwiseConvModule
from ..module.init_weights import normal_init
from .gfl_head import GFLHead
from ..loss.mask_loss import CombinedMaskLoss,MaskLoss
from ...data.transform.warp import warp_boxes


class NanoDetSegmHead(GFLHead):
    """
    Modified from GFL, use same loss functions but much lightweight convolution heads
    """

    def __init__(
        self,
        num_classes,
        loss,
        input_channel,
        stacked_convs=2,
        octave_base_scale=5,
        conv_type="DWConv",
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        reg_max=16,
        share_cls_reg=False,
        activation="LeakyReLU",
        feat_channels=256,
        strides=[8, 16, 32],
        mask_sz=48,
        **kwargs
    ):
        self.share_cls_reg = share_cls_reg
        self.mask_sz=mask_sz
        self.activation = activation
        self.ConvModule = ConvModule if conv_type == "Conv" else DepthwiseConvModule
        super(NanoDetSegmHead, self).__init__(
            num_classes,
            loss,
            input_channel,
            feat_channels,
            stacked_convs,
            octave_base_scale,
            strides,
            conv_cfg,
            norm_cfg,
            reg_max,
            **kwargs
        )

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        for _ in self.strides:
            cls_convs, reg_convs = self._buid_not_shared_head()
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

        # Segmentation head is shared 
        self.seg_convs=self._build_seg_head()

        self.gfl_cls = nn.ModuleList(
            [
                nn.Conv2d(
                    self.feat_channels,
                    self.cls_out_channels + 4 * (self.reg_max + 1)
                    if self.share_cls_reg
                    else self.cls_out_channels,
                    1,
                    padding=0,
                )
                for _ in self.strides
            ]
        )
        # TODO: if
        self.gfl_reg = nn.ModuleList(
            [
                nn.Conv2d(self.feat_channels, 4 * (self.reg_max + 1), 1, padding=0)
                for _ in self.strides
            ]
        )
        # Segmentation Head
        self.segm = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1, padding=0)
         # Segmentation loss
        self.calculate_mask_loss = MaskLoss()


    def _build_seg_head(self):
        modules = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            modules.append(
                self.ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None,
                    activation=self.activation,
                )
            )
        return nn.Sequential(*modules)

    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                self.ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None,
                    activation=self.activation,
                )
            )
            if not self.share_cls_reg:
                reg_convs.append(
                    self.ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None,
                        activation=self.activation,
                    )
                )

        return cls_convs, reg_convs

    def masks_to_image(self,masks):
        """
        masks: A tensor of shape (N, H, W) representing N binary masks.
        Returns: A tensor of shape (H, W) where each pixel is the sum of all masks at that location.
        """
        combined_mask = masks.sum(dim=0)
        combined_mask = torch.where(combined_mask > 0, 1.0, 0.0)  
        return combined_mask


    def crop_gt_masks(self, gt_mask, boxes,size):
        """
        gt_mask: A tensor of shape (H, W) representing a binary ground truth mask.
        boxes: A tensor of shape (N, 4) representing predicted bounding boxes, where N is the number of boxes.
        """
        cropped_masks = []
        gt_mask=self.masks_to_image(gt_mask)
        h, w = gt_mask.shape
        for box in boxes:
            x1, y1, x2, y2 = box.int()
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)
            cropped_mask = gt_mask[y1:y2, x1:x2]
            cropped_mask = cropped_mask[None, None, ...] 
            resized_masks = torch.nn.functional.interpolate(cropped_mask, size=size, mode='nearest')
            cropped_masks.append(resized_masks)  # Add a dimension for stacking
        cropped_masks = torch.cat(cropped_masks, dim=0)  # Stacks masks along a new dimension
        return cropped_masks

    def crop_and_resize_masks(self, gt_mask, boxes, size):
        cropped_masks = self.crop_gt_masks(gt_mask, boxes,size)  # Use the crop_gt_masks function defined earlier
        return cropped_masks

    def masks_process(self, preds, features, meta):
        cls_scores, bbox_preds = preds.split([self.num_classes, 4 * (self.reg_max + 1)], dim=-1)
        result_list = self.get_bboxes(cls_scores, bbox_preds, meta)
        input_height, input_width = meta["img"].shape[2:]
        feature_idx = 0
        _, _, fh, fw = features[feature_idx].shape
        spatial_scale = fh/input_height
        output_size = (self.mask_sz, self.mask_sz)
        all_pred_masks = []
        all_boxes = []
        for i, result in enumerate(result_list):
            image_boxes = result[0]
            if image_boxes.numel() > 0:
                boxes = image_boxes[:,:4]
                aligned_features = roi_align(features[feature_idx][i].unsqueeze(0), [boxes], output_size, spatial_scale=spatial_scale, sampling_ratio=-1)
                pred_masks = self.seg_convs(aligned_features)
                pred_masks = self.segm(pred_masks).sigmoid()
            else:
                pred_masks = torch.tensor([])
                boxes = torch.tensor([])

            all_pred_masks.append(pred_masks)
            all_boxes.append(boxes)
        return all_pred_masks

    

    def process_mask_train(self, preds, features, meta):
        cls_scores, bbox_preds = preds.split([self.num_classes, 4 * (self.reg_max + 1)], dim=-1)
        result_list = self.get_bboxes(cls_scores, bbox_preds, meta)
        gt_masks = meta["gt_masks"]

        input_height, input_width = meta["img"].shape[2:]
        feature_idx = 0
        _, _, fh, fw = features[feature_idx].shape
        spatial_scale = fh/input_height
        output_size = (self.mask_sz, self.mask_sz)  # The size of the predicted masks

        all_pred_masks = []
        all_boxes = []
        mask_losses = []

        for i, result in enumerate(result_list):
            image_boxes = result[0]
            if image_boxes.numel() > 0:
                boxes = image_boxes[:,:4]
                aligned_features = roi_align(features[feature_idx][i].unsqueeze(0), [boxes], output_size, spatial_scale=spatial_scale, sampling_ratio=-1)
                pred_masks = self.seg_convs(aligned_features)
                pred_masks = self.segm(pred_masks)
                # Crop and resize the ground truth masks based on the predicted boxes
                with torch.no_grad():
                    gt_resized_masks = self.crop_and_resize_masks(gt_masks[i], boxes, pred_masks.shape[-2:])

                # Calculate the mask loss
                mask_loss = self.calculate_mask_loss(pred_masks, gt_resized_masks)
                mask_losses.append(mask_loss)
            else:
                pred_masks = torch.tensor([], device=features[0].device)
                boxes = torch.tensor([], device=features[0].device)
                
            all_pred_masks.append(pred_masks)
            all_boxes.append(boxes)

        if mask_losses:
            mean_mask_loss = torch.stack(mask_losses).mean()  # Return mean mask loss across all images
        else:
            mean_mask_loss = torch.tensor(0., device=features[0].device)
        return all_pred_masks, mean_mask_loss  # Return mean mask loss across all images

    ## Overwrite post processing
    def post_process(self, preds, pred_masks, meta):
        cls_scores, bbox_preds = preds.split([self.num_classes, 4 * (self.reg_max + 1)], dim=-1)
        result_list = self.get_bboxes(cls_scores, bbox_preds, meta)
        det_results = {}
        warp_matrixes = (
            meta["warp_matrix"]
            if isinstance(meta["warp_matrix"], list)
            else meta["warp_matrix"]
        )
        img_heights = (
            meta["img_info"]["height"].cpu().numpy()
            if isinstance(meta["img_info"]["height"], torch.Tensor)
            else meta["img_info"]["height"]
        )
        img_widths = (
            meta["img_info"]["width"].cpu().numpy()
            if isinstance(meta["img_info"]["width"], torch.Tensor)
            else meta["img_info"]["width"]
        )
        img_ids = (
            meta["img_info"]["id"].cpu().numpy()
            if isinstance(meta["img_info"]["id"], torch.Tensor)
            else meta["img_info"]["id"]
        )

        for result, img_width, img_height, img_id, warp_matrix,masks in zip(result_list, img_widths, img_heights, img_ids, warp_matrixes,pred_masks):
            det_result = {}
            det_bboxes, det_labels = result
            det_bboxes = det_bboxes.detach().cpu().numpy()
            det_bboxes[:, :4] = warp_boxes(det_bboxes[:, :4], np.linalg.inv(warp_matrix), img_width, img_height)
            classes = det_labels.detach().cpu().numpy()
            for i in range(self.num_classes):
                inds = classes == i
                det_lst=[]
                for ind, bbox,mask in zip(inds, det_bboxes,masks):
                    if ind:
                        det_lst.append({
                        'bbox': bbox[:4].astype(np.float32).tolist(), 
                        'score': float(bbox[4]), 
                        'mask': mask.tolist(),
                        'height': img_height,
                        'width': img_width})
                det_result[i]=det_lst
                #det_result[i] = [{
                #    'bbox': bbox[:4].astype(np.float32).tolist(), 
                #    'score': float(bbox[4]), 
                #    'mask': mask.tolist(),
                #    'height': img_height,
                #    'width': img_width
                #} for ind, bbox,mask in zip(inds, det_bboxes,masks) if ind]
            det_results[img_id] = det_result
        return det_results

    def init_weights(self):
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.reg_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.seg_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        # init cls head with confidence = 0.01
        bias_cls = -4.595
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
            normal_init(self.gfl_reg[i], std=0.01)

    def forward(self, feats):
        outputs = []
        for x, cls_convs, reg_convs, gfl_cls, gfl_reg in zip(feats, self.cls_convs, self.reg_convs, self.gfl_cls, self.gfl_reg):
            cls_feat = x
            reg_feat = x
            for cls_conv in cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in reg_convs:
                reg_feat = reg_conv(reg_feat)
            if self.share_cls_reg:
                output = gfl_cls(cls_feat)
            else:
                cls_score = gfl_cls(cls_feat)
                bbox_pred = gfl_reg(reg_feat)
                output = torch.cat([cls_score, bbox_pred], dim=1)
            outputs.append(output.flatten(start_dim=2))
        outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)
        return outputs

    def _forward_onnx(self, preds, features):
        """only used for onnx export"""
        cls_scores, bbox_preds = preds.split([self.num_classes, 4 * (self.reg_max + 1)], dim=-1)
        result = self.onnx_get_bboxes(cls_scores, bbox_preds)
        input_height, input_width = (512,512)
        feature_idx = 0
        _, _, fh, fw = features[feature_idx].shape
        spatial_scale = fh/input_height
        output_size = (self.mask_sz, self.mask_sz)
        image_boxes = result[0]
        pred_classes = result[1]
        if image_boxes.numel() > 0:
            pred_boxes = image_boxes[:,:4]
            pred_scores = image_boxes[:,4]
            aligned_features = roi_align(features[feature_idx][0].unsqueeze(0), [pred_boxes], output_size, spatial_scale=spatial_scale, sampling_ratio=-1)
            pred_masks = self.seg_convs(aligned_features)
            pred_masks = self.segm(pred_masks).sigmoid()
        else:
            pred_masks = torch.tensor([])
            pred_boxes = torch.tensor([])
            pred_scores = torch.tensor([])
            pred_classes = torch.tensor([])

        return pred_boxes,pred_masks,pred_classes,pred_scores
