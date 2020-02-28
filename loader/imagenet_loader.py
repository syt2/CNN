import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image


def imagenet_train_dataloader(train_dir, batch_size=128, workers=8):
    trans = [transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]
    trans = transforms.Compose(trans)
    train_set = datasets.ImageFolder(train_dir, transform=trans)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=True, sampler=None)
    return train_loader


def imagenet_val_dataloader(val_dir, batch_size=128, workers=8):
    trans = [transforms.Resize(256, interpolation=Image.BICUBIC),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]
    trans = transforms.Compose(trans)
    val_set = datasets.ImageFolder(val_dir, transform=trans)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=batch_size, shuffle=False, num_workers=workers,
                                             pin_memory=True)
    return val_loader


def imagenet_loader(train_dir, val_dir, batch_size=128, workers=8):
    return imagenet_train_dataloader(train_dir=train_dir, batch_size=batch_size, workers=workers), \
           imagenet_val_dataloader(val_dir=val_dir, batch_size=batch_size, workers=workers)
