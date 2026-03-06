from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ----- Custom CNN -----


def build_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return train_tf, eval_tf


def build_transforms_imagenet(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def get_imagefolder_loaders(
    data_processed: Path,
    img_size: int,
    batch_size: int,
    num_workers: int,
    normalize: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    normalize=False  -> basic transforms (for custom CNN, no pretrained weights)
    normalize=True   -> ImageNet normalization + augmentation (for pretrained models)
    """
    if normalize:
        train_tf, eval_tf = build_transforms_imagenet(img_size)
    else:
        train_tf, eval_tf = build_transforms(img_size)

    train_ds = datasets.ImageFolder(data_processed / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(data_processed / "val",   transform=eval_tf)
    test_ds  = datasets.ImageFolder(data_processed / "test",  transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.classes