from .build import META_ARCH_REGISTRY, build_model  # isort:skip

# import all the meta_arch, so they will be registered
from .rcnn import GeneralizedRCNN, ProposalNetwork
from .rcnn_grad import GeneralizedDecoupledRCNN
from .rcnn_heur import GeneralizedPriorRCNN
from .sparse_rcnn import SparseRCNN
