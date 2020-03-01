import os
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np

from tqdm import tqdm
from model import get_model
from loss import get_loss_fn
from loader import get_loader
from utils import get_logger, convert_secs2time, time_string, accuracy, save_checkpoint
from metrics import RecorderMeter, AverageMeter
from schedulers import get_scheduler
from optimizers import get_optimizer

from tensorboardX import SummaryWriter

torch.backends.cudnn.benchmark = True


def train(cfg, writer, logger):
    # This statement must be declared before using torch
    use_cuda = False
    if cfg.get("cuda", None) is not None:
        if cfg.get("cuda", None) != "all":
            os.environ["CUDA_VISIBLE_DEVICES"] = cfg.get("cuda", None)
        use_cuda = torch.cuda.is_available()

    # Setup seed
    seed = cfg["training"].get("seed", random.randint(1, 10000))
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Setup Dataloader
    train_loader, val_loader = get_loader(cfg)

    # Setup Model
    model = get_model(cfg)
    if use_cuda and torch.cuda.device_count() > 0:
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    # writer.add_graph(model, torch.rand((1, 3, 224, 224)))

    # Setup optimizer, lr_scheduler and loss function
    optimizer = get_optimizer(model.parameters(), cfg)
    scheduler = get_scheduler(optimizer, cfg)
    loss_fn = get_loss_fn(cfg)

    # Setup Metrics
    epochs = cfg["training"]["epochs"]
    recorder = RecorderMeter(epochs)
    start_epoch = 0

    # save model parameters every <n> epochs
    save_interval = cfg["training"]["save_interval"]

    if use_cuda:
        model.cuda()
        loss_fn.cuda()

    # Resume Trained Model
    resume_path = os.path.join(writer.file_writer.get_logdir(), cfg["training"]["resume"])
    best_path = os.path.join(writer.file_writer.get_logdir(), cfg["training"]["best_model"])

    if cfg["training"]["resume"] is not None:
        if os.path.isfile(resume_path):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(resume_path)
            )
            checkpoint = torch.load(resume_path)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"]
            recorder = checkpoint['recorder']
            logger.info("Loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint["epoch"]))
        else:
            logger.info("No checkpoint found at '{}'".format(resume_path))

    epoch_time = AverageMeter()
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        scheduler.step(epoch)
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        logger.info('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(
            time_string(), epoch, epochs, need_time, scheduler.get_lr()[0]) +
                    ' [Best : Accuracy={:.2f}]'.format(recorder.max_accuracy(False))
                    )
        train_acc, train_los = train_epoch(train_loader, model, loss_fn, optimizer, use_cuda, logger)
        val_acc, val_los = validate_epoch(val_loader, model, loss_fn, use_cuda, logger)
        is_best = recorder.update(epoch, train_los, train_acc, val_los, val_acc)

        if is_best or epoch % save_interval == 0 or epoch == epochs - 1:  # save model (resume model and best model)
            save_checkpoint({
                'epoch': epoch + 1,
                'recorder': recorder,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, best_path, resume_path)

            for name, param in model.named_parameters():  # save histogram
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

        writer.add_scalar('Train/loss', train_los, epoch)  # save curves
        writer.add_scalar('Train/acc', train_acc, epoch)
        writer.add_scalar('Val/loss', val_los, epoch)
        writer.add_scalar('Val/acc', val_acc, epoch)

        epoch_time.update(time.time() - start_time)


def train_epoch(train_loader, model, loss_fn, optimizer, use_cuda, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    end_time = time.time()
    for i, (input, label) in enumerate(tqdm(train_loader)):
        data_time.update(time.time() - end_time)
        if use_cuda:
            label = label.cuda()
            input = input.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            label_var = torch.autograd.Variable(label)
        output = model(input_var)
        loss = loss_fn(output, label_var)
        prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

    logger.info('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg, losses.avg


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

    run_id = cfg["training"].get("runid", random.randint(1, 100000))
    logdir = os.path.join("runs", os.path.basename(args.config)[:-4], str(run_id))
    writer = SummaryWriter(logdir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Train begin")

    train(cfg, writer, logger)
