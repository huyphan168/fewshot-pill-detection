import numpy as np
from fvcore.common.file_io import PathManager
import json
import cv2
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


__all__ = ["register_meta_coco"]


def load_emed_json(imgdir, jsondir, metadata, dataset_name):
    """
    Load a json file with COCO's instances annotation format.
    """
    is_shots = "shot" in dataset_name
    if is_shots:
        img_ids = {}
        # split_dir = os.path.join("datasets", "cocosplit")
        split_dir = "/home/vishc1/datasets/emed/emedsplit"
        if "seed" in dataset_name:
            shot = dataset_name.split("_")[-2].split("shot")[0]
            seed = int(dataset_name.split("_seed")[-1])
            split_dir = os.path.join(split_dir, "seed{}".format(seed))
        for idx, cls in enumerate(metadata["classes"]):
            json_file = os.path.join(
                split_dir, "full_box_{}shot_{}_train.json".format(shot, metadata["name2id"][cls])
            )
            with open(json_file, "r") as f:
                data = json.load(f)
                annos = data["annotations"]
                for anno in annos:
                    if anno["image_id"] in img_ids:
                        img_ids[anno["image_id"]].append(anno)
                    else:
                        img_ids[anno["image_id"]] = [anno]
    name2id = metadata["name2id"]
    id_map = metadata["all_classes_id_to_contiguous_id"]
    dataset_dicts = []
    if is_shots:
        for img_id, annos in img_ids.items():
            record = {}
            filename = os.path.join(imgdir, img_id)
            height, width = cv2.imread(filename).shape[:2]
            
            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
            objs = []
            for box in annos:
                label_id = id_map[name2id[box["label"]]]
                obj = {
                    "bbox": [box["x"], box["y"], box["w"], box["h"]],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": label_id,
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    elif "base" in dataset_name:
        img_ids = os.listdir(imgdir)
        name2id = metadata["name2id"]
        id_map = metadata["base_classes_id_to_contiguous_id"]

        dataset_dicts = []
        for idx, ids in enumerate(img_ids):
            with open(os.path.join(jsondir, ids[:-4] + ".json")) as f:
                json_file = json.load(f)
            record = {}
            filename = os.path.join(imgdir, json_file["path"])
            height, width = cv2.imread(filename).shape[:2]
            
            
            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
        
            annos = json_file["boxes"]
            objs = []
            for box in annos:
                label_id = id_map[name2id[box["label"]]]
                obj = {
                    "bbox": [box["x"], box["y"], box["w"], box["h"]],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": label_id,
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    elif "all" in dataset_name:
        img_ids = os.listdir(imgdir)
        name2id = metadata["name2id"]
        id_map = metadata["all_classes_id_to_contiguous_id"]

        dataset_dicts = []
        for idx, ids in enumerate(img_ids):
            with open(os.path.join(jsondir, ids[:-4] + ".json")) as f:
                json_file = json.load(f)
            record = {}
            filename = os.path.join(imgdir, json_file["path"])
            height, width = cv2.imread(filename).shape[:2]
            
            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width

            annos = json_file["boxes"]
            objs = []
            for box in annos:
                label_id = id_map[name2id[box["label"]]]
                obj = {
                    "bbox": [box["x"], box["y"], box["w"], box["h"]],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": label_id,
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts


def register_meta_emed(name, metadata, imgdir, jsondir):
    DatasetCatalog.register(
        name,
        lambda: load_emed_json(imgdir, jsondir, metadata, name),
    )
    if "base" in name:
        metadata["thing_dataset_id_to_contiguous_id"] = metadata["base_classes_id_to_contiguous_id"]
        MetadataCatalog.get(name).set(thing_classes=metadata["base_classes"])
    elif "all" in name:
        metadata["thing_dataset_id_to_contiguous_id"] = metadata["all_classes_id_to_contiguous_id"]
        MetadataCatalog.get(name).set(thing_classes=metadata["classes"])
    MetadataCatalog.get(name).set(
        image_root=imgdir,
        evaluator_type="emed",
        dirname="/home/vishc1/datasets/emed/",
        **metadata,
    )

