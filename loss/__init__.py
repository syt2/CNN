import logging
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from loss.loss_fn import (
    LabelSmoothLoss,
)

key2loss = {
    "cross_entropy": CrossEntropyLoss,
    "label_smooth": LabelSmoothLoss,
}

logger = logging.getLogger("CNN")


def get_loss_fn(cfg):
    if cfg["training"]["loss"] is None:
        logger.info("Using default cross entropy loss")
        return nn.CrossEntropyLoss

    loss_dict = cfg["training"]["loss"]
    loss_name = loss_dict["name"]
    loss_params = {k: v for k, v in loss_dict.items() if k != "name"}

    if loss_name not in key2loss:
        raise NotImplementedError("Loss {} not implemented".format(loss_name))

    logger.info("Using {} with {} params".format(loss_name, loss_params))

    return key2loss[loss_name](**loss_params)

