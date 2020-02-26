import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def cifar100_train_dataloader(train_dir, batch_size=128, workers=8):
    trans = [transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(32, padding=4),
             transforms.ToTensor(),
             transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],
                                  std=[n / 255. for n in [68.2, 65.4, 70.4]])]
    trans = transforms.Compose(trans)
    train_set = datasets.CIFAR100(train_dir, train=True, transform=trans, download=True)
    loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    return loader


def cifar100_val_dataloader(val_dir, batch_size=128, workers=8):
    trans = [transforms.ToTensor(),
             transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],
                                  std=[n / 255. for n in [68.2, 65.4, 70.4]])]
    trans = transforms.Compose(trans)
    test_set = datasets.CIFAR100(val_dir, train=False, transform=trans)
    loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return loader


def cifar100_loader(train_dir, val_dir, batch_size=128, workers=8):
    return cifar100_train_dataloader(train_dir=train_dir, batch_size=batch_size, workers=workers), \
           cifar100_val_dataloader(val_dir=val_dir, batch_size=batch_size, workers=workers)
