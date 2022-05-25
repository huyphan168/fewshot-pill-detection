import sys
sys.path.insert(0, "/home/vishc1/projects/huy/fewshot_pill_detection")
import pickle as pkl
import torchvision.models as models
import torch
from fsdet.checkpoint import DetectionCheckpointer
from fsdet.modeling import build_model
from fsdet.config import get_cfg, set_global_cfg, add_custom_config
import matplotlib.pyplot as plt
config = "configs/VAIPE-detection/faster_rcnn_R_50_FPN_ftroi.yaml"
# config = "configs/analysis/faster_rcnn_R_50_ft_fc_13shot.yaml"

cfg = get_cfg()
cfg.merge_from_file(config)

model = build_model(cfg)
model.eval()
model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS)["model"])
have_bias = False
for p in model.roi_heads.box_head.parameters():
    print(p.size())
    if p.size()[-1] == 7*7*256:
        w1 = p.detach().cpu().numpy()
        have_bias = True
        continue
    if have_bias == True:
        b1 = p.detach().cpu().numpy()
        have_bias = False
with open("checkpoints/vaipe/TFA_fastercnn_base_r50/fc1.pkl", "wb") as f:
    print(b1.shape)
    pkl.dump({"weight":w1, "bias": b1}, f)
