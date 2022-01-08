import torch
model = torch.load("checkpoints/emed/sparse_rcnn/sparse_rcnn_R_50_base/model_final.pth")
print(model["model"].keys())
for name in model["model"].keys():
    if "head.head_series.5" in name:
        print(name, model["model"][name].size())
print(model["model"]["head.head_series.5.bboxes_delta.bias"].size())