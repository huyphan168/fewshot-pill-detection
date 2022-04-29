import numpy as np
from fvcore.common.file_io import PathManager
import json
import cv2
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from pycocotools.coco import COCO

import contextlib
import io

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


__all__ = ["register_meta_coco"]


# def load_emed_json(imgdir, jsondir, metadata, dataset_name):
#     """
#     Load a json file with COCO's instances annotation format.
#     """
#     is_shots = "shot" in dataset_name
#     if is_shots:
#         img_ids = {}
#         # split_dir = os.path.join("datasets", "cocosplit")
#         split_dir = "/home/vishc1/datasets/vaipe/vaipesplit"
#         if "seed" in dataset_name:
#             shot = dataset_name.split("_")[-2].split("shot")[0]
#             seed = int(dataset_name.split("_seed")[-1])
#             split_dir = os.path.join(split_dir, "seed{}".format(seed))
#         for idx, cls in enumerate(metadata["classes"]):
#             json_file = os.path.join(
#                 split_dir, "full_box_{}shot_{}_train.json".format(shot, metadata["name2id"][cls])
#             )
#             with open(json_file, "r") as f:
#                 data = json.load(f)
#                 annos = data["annotations"]
#                 for anno in annos:
#                     if anno["image_id"] in img_ids:
#                         img_ids[anno["image_id"]].append(anno)
#                     else:
#                         img_ids[anno["image_id"]] = [anno]
#     name2id = metadata["name2id"]
#     id_map = metadata["all_classes_id_to_contiguous_id"]
#     dataset_dicts = []
#     if is_shots:
#         for img_id, annos in img_ids.items():
#             record = {}
#             filename = os.path.join(imgdir, img_id)
#             height, width = cv2.imread(filename).shape[:2]
            
#             record["file_name"] = filename
#             record["image_id"] = idx
#             record["height"] = height
#             record["width"] = width
#             objs = []
#             for box in annos:
#                 label_id = id_map[name2id[box["label"]]]
#                 obj = {
#                     "bbox": [box["x"], box["y"], box["w"], box["h"]],
#                     "bbox_mode": BoxMode.XYWH_ABS,
#                     "category_id": label_id,
#                 }
#                 objs.append(obj)
#             record["annotations"] = objs
#             dataset_dicts.append(record)
#     elif "base" in dataset_name:
#         img_ids = os.listdir(imgdir)
#         name2id = metadata["name2id"]
#         id_map = metadata["base_classes_id_to_contiguous_id"]

#         dataset_dicts = []
#         for idx, ids in enumerate(img_ids):
#             with open(os.path.join(jsondir, ids[:-4] + ".json")) as f:
#                 json_file = json.load(f)
#             record = {}
#             filename = os.path.join(imgdir, json_file["path"])
#             height, width = cv2.imread(filename).shape[:2]
            
            
#             record["file_name"] = filename
#             record["image_id"] = idx
#             record["height"] = height
#             record["width"] = width
        
#             annos = json_file["boxes"]
#             objs = []
#             for box in annos:
#                 label_id = id_map[name2id[box["label"]]]
#                 obj = {
#                     "bbox": [box["x"], box["y"], box["w"], box["h"]],
#                     "bbox_mode": BoxMode.XYWH_ABS,
#                     "category_id": label_id,
#                 }
#                 objs.append(obj)
#             record["annotations"] = objs
#             dataset_dicts.append(record)
#     elif "all" in dataset_name:
#         img_ids = os.listdir(imgdir)
#         name2id = metadata["name2id"]
#         id_map = metadata["all_classes_id_to_contiguous_id"]

#         dataset_dicts = []
#         for idx, ids in enumerate(img_ids):
#             with open(os.path.join(jsondir, ids[:-4] + ".json")) as f:
#                 json_file = json.load(f)
#             record = {}
#             filename = os.path.join(imgdir, json_file["path"])
#             height, width = cv2.imread(filename).shape[:2]
            
#             record["file_name"] = filename
#             record["image_id"] = idx
#             record["height"] = height
#             record["width"] = width

#             annos = json_file["boxes"]
#             objs = []
#             for box in annos:
#                 label_id = id_map[name2id[box["label"]]]
#                 obj = {
#                     "bbox": [box["x"], box["y"], box["w"], box["h"]],
#                     "bbox_mode": BoxMode.XYWH_ABS,
#                     "category_id": label_id,
#                 }
#                 objs.append(obj)
#             record["annotations"] = objs
#             dataset_dicts.append(record)
#     return dataset_dicts

def load_emed_json(json_file, image_root, metadata, dataset_name):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection.
    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str): the directory where the images in this json file exists.
        metadata: meta data associated with dataset_name
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    is_shots = "shot" in dataset_name
    is_joint = "few" in dataset_name
    if is_shots:
        fileids = {}
        split_dir = "/home/vishc1/datasets/vaipe/vaipesplit"
        if "seed" in dataset_name:
            shot = dataset_name.split("_")[-2].split("shot")[0]
            seed = int(dataset_name.split("_seed")[-1])
            split_dir = os.path.join(split_dir, "seed{}".format(seed))
        else:
            shot = dataset_name.split("_")[-1].split("shot")[0]
        for idx, cls in enumerate([metadata["name2id"][cls] for cls in metadata["thing_classes"]]):
            json_file = os.path.join(
                split_dir, "full_box_{}shot_{}_train.json".format(shot, cls)
            )
            json_file = PathManager.get_local_path(json_file)
            with contextlib.redirect_stdout(io.StringIO()):
                coco_api = COCO(json_file)
            img_ids = sorted(list(coco_api.imgs.keys()))
            imgs = coco_api.loadImgs(img_ids)
            anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
            fileids[idx] = list(zip(imgs, anns))
    elif is_joint:
        fileids = {}
        split_dir = "/home/vishc1/datasets/vaipe/vaipesplit"
        if "seed" in dataset_name:
            shot = dataset_name.split("_")[-1]
            seed = int(dataset_name.split("_seed")[-1])
            split_dir = os.path.join(split_dir, "seed{}".format(seed))
        else:
            shot = dataset_name.split("_")[-1]
        for idx, cls in enumerate([metadata["name2id"][cls] for cls in metadata["novel_classes"]]):
            json_file_novel = os.path.join(
                split_dir, "full_box_{}shot_{}_train.json".format(shot, cls)
            )
            json_file_novel = PathManager.get_local_path(json_file_novel)
            with contextlib.redirect_stdout(io.StringIO()):
                coco_api = COCO(json_file_novel)
            img_ids = sorted(list(coco_api.imgs.keys()))
            imgs = coco_api.loadImgs(img_ids)
            anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
            fileids[idx] = list(zip(imgs, anns))
        json_file = PathManager.get_local_path(json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(json_file)
        # sort indices for reproducible results
        img_ids = sorted(list(coco_api.imgs.keys()))
        imgs = coco_api.loadImgs(img_ids)
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
        imgs_anns = list(zip(imgs, anns))
    else:
        json_file = PathManager.get_local_path(json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(json_file)
        # sort indices for reproducible results
        img_ids = sorted(list(coco_api.imgs.keys()))
        imgs = coco_api.loadImgs(img_ids)
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
        imgs_anns = list(zip(imgs, anns))
    id_map = metadata["thing_dataset_id_to_contiguous_id"]

    dataset_dicts = []
    ann_keys = ["iscrowd", "bbox", "category_id", "edge_file", "texture_file"]

    if is_shots:
        for _, fileids_ in fileids.items():
            dicts = []
            for (img_dict, anno_dict_list) in fileids_:
                for anno in anno_dict_list:
                    record = {}
                    record["file_name"] = os.path.join(
                        image_root, img_dict["file_name"]
                    )
                    record["height"] = img_dict["height"]
                    record["width"] = img_dict["width"]
                    image_id = record["image_id"] = img_dict["id"]

                    assert anno["image_id"] == image_id
                    assert anno.get("ignore", 0) == 0

                    obj = {key: anno[key] for key in ann_keys if key in anno}

                    obj["bbox_mode"] = BoxMode.XYWH_ABS
                    obj["category_id"] = id_map[obj["category_id"]]
                    record["annotations"] = [obj]
                    dicts.append(record)
            if len(dicts) > int(shot):
                dicts = np.random.choice(dicts, int(shot), replace=False)
            dataset_dicts.extend(dicts)
    elif is_joint:
        for _, fileids_ in fileids.items():
            dicts = []
            for (img_dict, anno_dict_list) in fileids_:
                for anno in anno_dict_list:
                    record = {}
                    record["file_name"] = os.path.join(
                        image_root, img_dict["file_name"]
                    )
                    record["height"] = img_dict["height"]
                    record["width"] = img_dict["width"]
                    image_id = record["image_id"] = img_dict["id"]

                    assert anno["image_id"] == image_id
                    assert anno.get("ignore", 0) == 0

                    obj = {key: anno[key] for key in ann_keys if key in anno}

                    obj["bbox_mode"] = BoxMode.XYWH_ABS
                    obj["category_id"] = id_map[obj["category_id"]]
                    record["annotations"] = [obj]
                    dicts.append(record)
                    
            if len(dicts) > int(shot):
                dicts = np.random.choice(dicts, int(shot), replace=False)
            dataset_dicts.extend(dicts)
        for (img_dict, anno_dict_list) in imgs_anns:
            record = {}
            record["file_name"] = os.path.join(
                image_root, img_dict["file_name"]
            )
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            image_id = record["image_id"] = img_dict["id"]

            objs = []
            for anno in anno_dict_list:
                assert anno["image_id"] == image_id
                assert anno.get("ignore", 0) == 0

                obj = {key: anno[key] for key in ann_keys if key in anno}
                obj["bbox_mode"] = BoxMode.XYWH_ABS
                if obj["category_id"] in id_map and obj["category_id"] in metadata["base_dataset_id_to_contiguous_id"]:
                    obj["category_id"] = id_map[obj["category_id"]]
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    else:
        for (img_dict, anno_dict_list) in imgs_anns:
            record = {}
            record["file_name"] = os.path.join(
                image_root, img_dict["file_name"]
            )
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            image_id = record["image_id"] = img_dict["id"]

            objs = []
            for anno in anno_dict_list:
                assert anno["image_id"] == image_id
                assert anno.get("ignore", 0) == 0

                obj = {key: anno[key] for key in ann_keys if key in anno}

                obj["bbox_mode"] = BoxMode.XYWH_ABS
                if obj["category_id"] in id_map:
                    obj["category_id"] = id_map[obj["category_id"]]
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)

    return dataset_dicts

# def register_meta_emed(name, metadata, imgdir, jsondir):
#     DatasetCatalog.register(
#         name,
#         lambda: load_emed_json(imgdir, jsondir, metadata, name),
#     )
#     if "base" in name:
#         metadata["thing_dataset_id_to_contiguous_id"] = metadata["base_classes_id_to_contiguous_id"]
#         MetadataCatalog.get(name).set(thing_classes=metadata["base_classes"])
#     elif "all" in name:
#         metadata["thing_dataset_id_to_contiguous_id"] = metadata["all_classes_id_to_contiguous_id"]
#         MetadataCatalog.get(name).set(thing_classes=metadata["classes"])
#     MetadataCatalog.get(name).set(
#         image_root=imgdir,
#         evaluator_type="emed",
#         dirname="/home/vishc1/datasets/emed/",
#         **metadata,
#     )

def register_meta_emed(name, metadata, imgdir, annofile):
    DatasetCatalog.register(
        name,
        lambda: load_emed_json(annofile, imgdir, metadata, name),
    )

    if "_base" in name or "_novel" in name:
        split = "base" if "_base" in name else "novel"
        metadata["thing_dataset_id_to_contiguous_id"] = metadata[
            "{}_dataset_id_to_contiguous_id".format(split)
        ]
        metadata["thing_classes"] = metadata["{}_classes".format(split)]

    MetadataCatalog.get(name).set(
        json_file=annofile,
        image_root=imgdir,
        evaluator_type="emed",
        dirname="/home/vishc1/datasets/vaipe",
        **metadata,
    )