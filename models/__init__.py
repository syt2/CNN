from models.ResNeXt import ResNeXt
from models.CifarResNet import ResNet as CifarResNet
from models.ResNet import ResNet
from models.SEResNet import ResNet as SEResNet
from models.BasicResNet import ResNet as BasicResNet
from models.MobileNetV2 import mobilenet_v2 as MobileNetV2
from models.CBAM import ResNet as CBAM

def get_model(cfg):
    assert cfg["model"] is not None, "model is unspecified"
    assert cfg["data"]["dataset"] is not None, "dataset is unspecified"

    model_dict = cfg["model"]
    model_name = model_dict["arch"]
    model_params = {k: v for k, v in model_dict.items() if k != "arch"}
    model = _get_model_instance(model_name)

    dataset_name = cfg["data"]["dataset"]
    num_classes = _get_num_classes(dataset_name)

    model = model(num_classes=num_classes, **model_params)
    return model


def _get_num_classes(name):
    try:
        return {
            "imagenet": 1000,
            "cifar10": 10,
            "cifar100": 100,
        }[name]
    except:
        raise ("Dataset {} not available".format(name))


def _get_model_instance(name):
    try:
        return {
            "resnext": ResNeXt,
            "cifarresnet": CifarResNet,
            "resnet": ResNet,
            "seresnet": SEResNet,
            "basicresnet": BasicResNet,
            "mobilenetv2": MobileNetV2,
            "cbam": CBAM,
        }[name]
    except:
        raise ("Model {} not available".format(name))
