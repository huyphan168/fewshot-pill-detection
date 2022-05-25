import torch
import torch.nn as nn
import numpy as np
from numpy.linalg import lstsq
from scipy.linalg import orth
from fsdet.config import get_cfg, add_custom_config
from fsdet.modeling.meta_arch import build_model
import scipy
import pickle as pkl

def lowrank_init(cfg, RT_torch, L_torch, b_torch, res_layers):
    weight_dict = pkl.load(open(cfg.MODEL.ROI_BOX_HEAD.WEIGHT_PATH, "rb"))
    W = weight_dict["weight"]
    b = weight_dict["bias"]
    L,RT = lowrank_decompose(cfg, W)
    print(L.shape)
    print(RT.shape)
    print(RT_torch.weight.data.size())
    print(L_torch.weight.data.size())
    # assert L.shape == L_torch.weight.data.numpy().shape
    # assert RT.shape == RT_torch.weight.data.numpy().shape
    RT_torch.weight.data = torch.tensor(RT)
    L_torch.weight.data = torch.tensor(L)
    b_torch.data = torch.tensor(b).cuda()
    for i, _ in enumerate(res_layers):
        u,v = orth_init(i, res_layers, L)
        res_layers[i][0].data = torch.tensor(u).float()
        res_layers[i][1].data = torch.tensor(v).float()
    return RT_torch, L_torch, b_torch, [(u.cuda(), v.cuda()) for (u,v) in res_layers]
def lowrank_decompose(cfg, W):
    method = cfg.MODEL.ROI_BOX_HEAD.LOWRANK_DECOMPOSE_METHOD 
    if method == "SparseSVD":
        u, s, vT = scipy.sparse.linalg.svds(W, k=cfg.MODEL.ROI_BOX_HEAD.RANK)
        return u, np.diag(s)@vT
    else:
        pass

def orth_init(i, res_layers, W):
    U = [np.expand_dims(res_layers[j][0].data.numpy(),1) for j in range(i+1)]
    U.append(W)
    for vec in U:
        print(vec.shape)
    orth_W = np.concatenate(U, axis=1)
    O = orth(orth_W)
    res = find_orth(O)
    if all(np.abs(np.dot(res, col)) < 10e-9 for col in O.T):
        print("Success")
    else:
        print("Failure")
    return torch.tensor(res), res_layers[i][1].data

def find_orth(O):
        rand_vec = np.random.rand(O.shape[0], 1)
        A = np.hstack((O, rand_vec))
        b = np.zeros(O.shape[1] + 1)
        b[-1] = 1
        return lstsq(A.T, b)[0]