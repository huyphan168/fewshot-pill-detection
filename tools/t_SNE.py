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
import tqdm
import seaborn as sns
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE
import time
from sklearn import preprocessing
from torch.nn import functional as F
RS=123

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='config file')
    parser.add_argument('--model-name', type=str, default='', help='model name')
    return parser.parse_args()

def fashion_scatter(x, colors, path):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.xlabel("tSNE_1")
    plt.ylabel("tSNE_2")
    plt.title("tSNE Visualization of RoI features")
    plt.grid(visible=True)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    # for i in range(num_classes):

    #     # Position of each label at median of data points.
    #     print(np.median(x[colors == i, :], axis=0))
    #     xtext, ytext = np.median(x[colors == i, :], axis=0)
    #     txt = ax.text(xtext, ytext, str(i), fontsize=24)
    #     txt.set_path_effects([
    #         PathEffects.Stroke(linewidth=5, foreground="w"),
    #         PathEffects.Normal()])
    #     txts.append(txt)
    plt.savefig(path, dpi=300)
    return f, ax, sc, txts

if __name__ == "__main__":

    args = arg_parse()
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    if not os.path.exists(os.path.join("analysis_results", args.model_name)):
        os.makedirs(os.path.join("analysis_results", args.model_name))
    predictor = DefaultPredictor(cfg)

    with open("/home/vishc1/datasets/vaipe/few_shot_names.pkl", "rb") as f:
        few_shot_names = pkl.load(f)
    with open("/home/vishc1/datasets/vaipe/base_names.pkl", "rb") as f:
        base_shot_names = pkl.load(f)
    with open("/home/vishc1/datasets/vaipe/name2id.pkl", "rb") as f:
        name2id = pkl.load(f)
    cls_feats = {name:[] for name in name2id.values()}
    few_objects = []
    # for i in tqdm.tqdm(os.listdir("/home/vishc1/datasets/vaipe/test/labels")):
    #     json_path = os.path.join("/home/vishc1/datasets/vaipe/test/labels", i)
    #     with open(json_path, "r") as f:
    #         data = json.load(f)
    #     boxes = data["boxes"] 
    #     img_id = data["path"]
    #     img_path = os.path.join("/home/vishc1/datasets/vaipe/test/images", img_id)

    #     im = cv2.imread(img_path)
    #     im = cv2.resize(im, (640, 640))

    #     outputs, features = predictor(im)
    #     outputs = outputs[0]
    #     classes = outputs["instances"].pred_classes
    #     for index, cls in enumerate(classes):
    #         cls_feats[cls.cpu().numpy().tolist()].append(features[index])
    few_ids = [name2id[name] for name in few_shot_names]
    
    # with open(os.path.join("analysis_results", args.model_name, "cls_feats.pkl"), "wb") as f:
    #     pkl.dump(cls_feats, f)
    with open(os.path.join("analysis_results", args.model_name, "cls_feats.pkl"), "rb") as f:
        cls_feats = pkl.load(f)
    cls_feats_few = {name:[feat.cpu().numpy() for feat in cls_feats[name]] for name in few_ids}
    x,y=[],[]
    remap = {x:i for i,x in enumerate(few_ids)}
    for cls in cls_feats_few:
        for feat in cls_feats_few[cls]:
            x.append(feat)
            y.append(remap[cls])
    print(x[0])
    fashion_tsne = TSNE(random_state=RS).fit_transform(x)
    print(fashion_tsne)
    fashion_scatter(fashion_tsne, y, os.path.join("analysis_results", args.model_name, "tsne.png"))