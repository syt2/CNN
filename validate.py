import yaml
import torch
import argparse
import os

from tqdm import tqdm
from model import get_model
from loader import get_loader
from loss import get_loss_fn
from metrics import AverageMeter
from utils import convert_state_dict, accuracy


def validate(cfg):
    use_cuda = False
    if cfg.get("cuda", None) is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.get("cuda", None)
        use_cuda = torch.cuda.is_available()

    # Setup Dataloader
    train_loader, val_loader = get_loader(cfg)

    # Setup Model
    model = get_model(cfg)
    if use_cuda and torch.cuda.device_count() > 0:
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    best_path = os.path.join(cfg["val"]["best_model"])

    if os.path.isfile(best_path):
        checkpoint = torch.load(best_path)
        # state = convert_state_dict(checkpoint["state_dict"])
        model.load_state_dict(checkpoint["state_dict"])

        loss_fn = get_loss_fn(cfg)
        validate_epoch(val_loader, model, loss_fn, use_cuda)


def validate_epoch(val_loader, model, loss_fn, use_cuda, logger):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    for i, (input, label) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            if use_cuda:
                label = label.cuda()
                input = input.cuda()
            input_var = torch.autograd.Variable(input)
            label_var = torch.autograd.Variable(label)
            output = model(input_var)
            loss = loss_fn(output, label_var)
        prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))
    logger.info('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg, losses.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/resnet_imagenet.yml",
        help="Configuration file to use",
    )
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp)

    validate(cfg)
