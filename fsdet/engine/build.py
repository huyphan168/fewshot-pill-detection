from detectron2.utils.registry import Registry

TRAINER_REGISTRY = Registry("TRAINER")  # noqa F401 isort:skip
TRAINER_REGISTRY.__doc__ = """
Registry for trainers"""


def build_trainer(cfg):
    """
    Built the trainer
    """
    trainer = cfg.TRAINER
    return TRAINER_REGISTRY.get(trainer)(cfg)