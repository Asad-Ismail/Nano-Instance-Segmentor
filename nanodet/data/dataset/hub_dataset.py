# Copyright 2023 Asad Ismail.

import logging
import os
import time
import xml.etree.ElementTree as ET
from collections import defaultdict

from pycocotools.coco import COCO

import deeplake as hub
import cv2
import numpy as np
from tqdm import tqdm
import pycocotools.mask as mask_util

from .coco import CocoDataset


class CocoHub(COCO):
    def __init__(self, annotation):
        """
        Constructor of Microsoft COCO helper class for
        reading and visualizing annotations.
        :param annotation: annotation dict
        :return:
        """
        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        dataset = annotation
        assert type(dataset) == dict, "annotation file format {} not supported".format(
            type(dataset)
        )
        self.dataset = dataset
        self.createIndex()


class HubDataset(CocoDataset):
    def __init__(self, class_names, **kwargs):
        self.class_names = class_names
        src="hub://aismail2/cucumber_OD"
        self.ds = hub.load(src)
        self.img_sz=kwargs["input_size"][0]
        self.seg_sz=64
        self.save_imgs=True
        self.img_path=kwargs["img_path"]
        super(HubDataset, self).__init__(**kwargs)
        self.use_instance_mask= True

    def hub_to_coco(self, ann_path):
        """
        convert xml annotations to coco_api
        :param ann_path:
        :return:
        """
        logging.info("loading annotations into memory...")
        tic = time.time()
        image_info = []
        categories = []
        annotations = []
        
        for idx, supercat in enumerate(self.class_names):
            categories.append({"supercategory": supercat, "id": idx + 1, "name": supercat})
            
        ann_id = 1
        
        for i,d in tqdm(enumerate(self.ds)):
            if (i>315):
                break
            image=d.images.numpy()
            image=cv2.resize(image,(self.img_sz,self.img_sz))
            if self.save_imgs:
                cv2.imwrite(os.path.join(self.img_path,f"{i}.png"),image)
            masks=d.masks.numpy().astype(np.uint8)*255

            file_name = f"{i}.png"
            height,width,_=image.shape
            info = {
                "file_name": file_name,
                "height": height,
                "width": width,
                "id": i + 1,
            }
            image_info.append(info)
            
            for j in range(masks.shape[-1]):
                mask=masks[...,j]
                mask=cv2.resize(mask,(self.img_sz,self.img_sz),cv2.INTER_NEAREST)//255
                #print(f"Mask min and max is {mask.min()},{mask.max()}")
                nzeros=np.nonzero(mask)
                ys=nzeros[0]
                xs=nzeros[1]
                ymin=min(ys)
                ymax=max(ys)
                xmin=min(xs)
                xmax=max(xs)          
                w = xmax - xmin
                h = ymax - ymin
                if w < 0 or h < 0:
                    logging.warning(
                        "WARNING! Find error data in file {}! Box w and "
                        "h should > 0. Pass this box annotation.".format(xml_name)
                    )
                    continue
                coco_box = [max(xmin, 0), max(ymin, 0), min(w, width), min(h, height)]
                ann = {
                    "image_id": i + 1,
                    "bbox": coco_box,
                    "category_id": 1,
                    "iscrowd": 0,
                    "id": ann_id,
                    "area": coco_box[2] * coco_box[3],
                    "masks":mask,
                    # required for coco eval :(
                    "segmentation": mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
                }
                annotations.append(ann)
                ann_id += 1

        coco_dict = {
            "images": image_info,
            "categories": categories,
            "annotations": annotations,
        }
        logging.info("Load {} xml files and {} boxes".format(len(image_info), len(annotations)))
        logging.info("Done (t={:0.2f}s)".format(time.time() - tic))
            
        return coco_dict

    def get_data_info(self, ann_path):
        """
        Load basic information of dataset such as image path, label and so on.
        :param ann_path: coco json file path
        :return: image info:
        [{'file_name': '000000000139.jpg',
          'height': 426,
          'width': 640,
          'id': 139},
         ...
        ]
        """
        coco_dict = self.hub_to_coco(ann_path)
        self.coco_api = CocoHub(coco_dict)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.img_ids = sorted(self.coco_api.imgs.keys())
        img_info = self.coco_api.loadImgs(self.img_ids)
        return img_info

if __name__=="__main__":
    pass