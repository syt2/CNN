import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def cifar10_train_dataloader(train_dir, batch_size=128, workers=8):
    trans = [transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(32, padding=4),
             transforms.ToTensor(),
             transforms.Normalize(mean=[x / 255 for x in [125.3, 123.0, 113.9]],
                                  std=[x / 255 for x in [63.0, 62.1, 66.7]])]
    trans = transforms.Compose(trans)
    train_set = datasets.CIFAR10(train_dir, train=True, transform=trans, download=True)
    loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    return loader


def cifar10_val_dataloader(val_dir, batch_size=128, workers=8):
    trans = [transforms.ToTensor(),
             transforms.Normalize(mean=[x / 255 for x in [125.3, 123.0, 113.9]],
                                  std=[x / 255 for x in [63.0, 62.1, 66.7]])]
    trans = transforms.Compose(trans)
    test_set = datasets.CIFAR10(val_dir, train=False, transform=trans)
    loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return loader


def cifar10_loader(train_dir, val_dir, batch_size=128, workers=8):
    return cifar10_train_dataloader(train_dir=train_dir, batch_size=batch_size, workers=workers), \
           cifar10_val_dataloader(val_dir=val_dir, batch_size=batch_size, workers=workers)
