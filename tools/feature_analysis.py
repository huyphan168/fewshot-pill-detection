import sys
sys.path.insert(0, "/home/aiotlab/projects/huyvinuni/RFC/few-shot-object-detection")
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from fsdet.config import get_cfg, set_global_cfg
from fsdet.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
from detectron2.data import MetadataCatalog
import pickle as pkl
import argparse
import yaml
import os
import random
import json

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='config file')
    parser.add_argument('--model-name', type=str, default='', help='model name')
    parser.add_argument('--test-folder', type=str, default='test', help='test folder')
    return parser.parse_args()

def activation_hook_fpn3(inst, inp, out):
    out = out.squeeze()
    out = out.sum(axis=0)
    out_flat = out.flatten()
    out_norm = F.normalize(out_flat.unsqueeze(0), p=2, dim=1)
    out = out_norm.view_as(out).cpu().numpy()
    out = np.swapaxes(out, 0,1)
    # fig, ax = plt.subplots(out.shape[0])
    # for idx in range(out.shape[0]):
    #     ax[idx].imshow(out[idx])
    fig = plt.figure()
    plt.imshow(out)
    writer.add_figure("fpn feature output layer 3", fig, 0)

def activation_hook_fpn2(inst, inp, out):
    out = out.squeeze()
    out = out.sum(axis=0)
    out_flat = out.flatten()
    out_norm = F.normalize(out_flat.unsqueeze(0), p=2, dim=1)
    out = out_norm.view_as(out).cpu().numpy()
    out = np.swapaxes(out, 0,1)
    # fig, ax = plt.subplots(out.shape[0])
    # for idx in range(out.shape[0]):
    #     ax[idx].imshow(out[idx])
    fig = plt.figure()
    plt.imshow(out)
    writer.add_figure("fpn feature output layer 2", fig, 0)

def activation_hook_fpn4(inst, inp, out):
    out = out.squeeze()
    out = out.sum(axis=0)
    out_flat = out.flatten()
    out_norm = F.normalize(out_flat.unsqueeze(0), p=2, dim=1)
    out = out_norm.view_as(out).cpu().numpy()
    out = np.swapaxes(out, 0,1)
    # fig, ax = plt.subplots(out.shape[0])
    # for idx in range(out.shape[0]):
    #     ax[idx].imshow(out[idx])
    fig = plt.figure()
    plt.imshow(out)
    writer.add_figure("fpn feature output layer 4", fig, 0)

def activation_hook_fpn5(inst, inp, out):
    out = out.squeeze()
    out = out.sum(axis=0)
    out_flat = out.flatten()
    out_norm = F.normalize(out_flat.unsqueeze(0), p=2, dim=1)
    out = out_norm.view_as(out).cpu().numpy()
    out = np.swapaxes(out, 0,1)
    # fig, ax = plt.subplots(out.shape[0])
    # for idx in range(out.shape[0]):
    #     ax[idx].imshow(out[idx])
    fig = plt.figure()
    plt.imshow(out)
    writer.add_figure("fpn feature output layer 5", fig, 0)

if __name__ == "__main__":
    args = arg_parse()
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    if not os.path.exists(os.path.join("few-shot-object-detection/analysis_results", args.model_name)):
        os.makedirs(os.path.join("few-shot-object-detection/analysis_results", args.model_name))
    writer = SummaryWriter(os.path.join("few-shot-object-detection/analysis_results", args.model_name))
    predictor = DefaultPredictor(cfg)
    predictor.model.backbone.fpn_output3.register_forward_hook(activation_hook_fpn3)
    predictor.model.backbone.fpn_output2.register_forward_hook(activation_hook_fpn2)
    predictor.model.backbone.fpn_output4.register_forward_hook(activation_hook_fpn4)
    predictor.model.backbone.fpn_output5.register_forward_hook(activation_hook_fpn5)

    with open("/home/vishc1/datasets/vaipe/few_shot_names.pkl", "rb") as f:
        few_shot_names = pkl.load(f)
    with open("/home/vishc1/datasets/vaipe/base_names.pkl", "rb") as f:
        base_shot_names = pkl.load(f)
    with open("/home/vishc1/datasets/vaipe/name2id.pkl", "rb") as f:
        name2id = pkl.load(f)
    few_shot_ids = [name2id[name] for name in few_shot_names]


    few_objects = []
    for i in os.listdir("/home/vishc1/datasets/vaipe/test/labels"):
        json_path = os.path.join("/home/vishc1/datasets/vaipe/test/labels", i)
        with open(json_path, "r") as f:
            data = json.load(f)
        boxes = data["boxes"] 
        for idx, box in enumerate(boxes):
            if box["label"] in few_shot_names:
                img_id = data["path"]
                box_few = box
                few_objects.append((img_id, box))
                break
    
    few_selected = random.sample(few_objects, 1)
    img_id = few_selected[0][0]
    box_few = few_selected[0][1]

    img_path = os.path.join(args.test_folder, img_id)

    im = cv2.imread(img_path)
    im_cvt = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_drawed = cv2.rectangle(im_cvt, (box_few["x"], box_few["y"]), (box_few["x"] + box_few["w"], box_few["y"] + box_few["h"]), (0, 255, 0), 10)
    im = cv2.resize(im, (640, 640))

    outputs = predictor(im)
    # print(outputs)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    predicted_img = v.get_image()
    writer.add_image("predicted image", np.swapaxes(predicted_img, 2,0), 0)
    writer.add_image("source image", np.swapaxes(im_drawed,2,0), 0)
    writer.close()