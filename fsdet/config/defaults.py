from detectron2.config import CfgNode as CN
from detectron2.config.defaults import _C
from typing import List
# adding additional default values built on top of the default values in detectron2

_CC = _C
_CC.TRAINER = "Trainer"
# FREEZE Parameters
_CC.MODEL.ROI_HEADS.CONTRASTIVE_BRANCH = CN()
_CC.MODEL.BACKBONE.FREEZE = False
_CC.MODEL.PROPOSAL_GENERATOR.FREEZE = False
_CC.MODEL.ROI_HEADS.FREEZE_FEAT = False
_CC.MODEL.BACKBONE.UNFREEZE_CONV5 = False
# choose from "FastRCNNOutputLayers" and "CosineSimOutputLayers"
_CC.MODEL.ROI_HEADS.OUTPUT_LAYER = "FastRCNNOutputLayers"
# scale of cosine similarity (set to -1 for learnable scale)
_CC.MODEL.ROI_HEADS.COSINE_SCALE = 20.0
# Backward Compatible options.
_CC.MUTE_HEADER = True
