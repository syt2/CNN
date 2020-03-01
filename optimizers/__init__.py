import logging

from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop

logger = logging.getLogger("CNN")

key2opt = {
    "sgd": SGD,
    "adam": Adam,
    "asgd": ASGD,
    "adamax": Adamax,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
}


def get_optimizer(params, cfg):
    if cfg["training"]["optimizer"] is None:
        logger.info("Using SGD optimizer")
        return SGD

    opt_dict = cfg["training"]["optimizer"]
    opt_name = opt_dict["name"]
    opt_params = {k: v for k, v in opt_dict.items() if k != "name"}

    if opt_name not in key2opt:
        raise NotImplementedError("Optimizer {} not implemented".format(opt_name))
    logger.info("Using {} optimizer with {} params".format(opt_name, opt_params))
    return key2opt[opt_name](params, **opt_params)
