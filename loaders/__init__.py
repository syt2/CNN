import logging
from loaders.cifar10_loader import cifar10_loader
from loaders.cifar100_loader import cifar100_loader
from loaders.imagenet_loader import imagenet_loader

logger = logging.getLogger("CNN")

key2loader = {
    "cifar10": cifar10_loader,
    "cifar100": cifar100_loader,
    "imagenet": imagenet_loader,
}


def get_loader(cfg):
    assert cfg["data"] is not None, "dataset is unspecified"
    assert cfg["training"] is not None and \
           cfg["training"]["batch_size"] is not None, "batch_size is unspecified"

    loader_dict = cfg["data"]
    loader_dataset = loader_dict["dataset"]
    loader_params = {k: v for k, v in loader_dict.items() if k != "dataset"}
    loader_batch_size = cfg["training"]["batch_size"]

    if loader_dataset not in key2loader:
        raise NotImplementedError("Dataset {} not implemented".format(loader_dataset))

    logger.info("Using {} with {} params".format(loader_dataset, loader_params))
    return key2loader[loader_dataset](**loader_params, batch_size=loader_batch_size)
