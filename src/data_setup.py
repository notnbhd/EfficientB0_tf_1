import torch
import torchvision
import pathlib
import os
import zipfile
import shutil


from torchvision import transforms, datasets
from pathlib import Path

import requests

def data_create():
    data_dir = pathlib.Path('/data')
    data_root = pathlib.Path('/data_root')

    train_data = datasets.Food101(
        root = data_root,
        split = 'train',
        download = True
    )

    test_data = datasets.Food101(
        root = data_root,
        split = 'test',
        download = True
    )

    data_path = data_root / 'food-101/images'
    target_classes = ['pho', 'ramen', 'spaghetti_carbonara']

    def get_subset(image_path=data_path, 
                   data_splits=['train', 'test'],    
                   target_classes=target_classes,
                   ):
        label_splits = {}

        for data_split in data_splits:
            label_path = data_root / 'food-101' / 'meta' / f'{data_split}.txt'
            with open(label_path, 'r') as f:
                labels = [line.strip('\n') for line in f.readlines() if line.split('/')[0] in target_classes]
            image_paths = [pathlib.Path(str(image_path / label) + '.jpg') for label in labels]
            label_splits[data_split] = image_paths

        return label_splits

    label_splits = get_subset()
    #label_splits['train']

    target_dir = pathlib.Path('../data/noodles')
    target_dir.mkdir(parents=True, exist_ok=True)



    for image_split in label_splits.keys():
        for image_path in label_splits[str(image_split)]:
            destination = target_dir / image_split / image_path.parent.stem / image_path.name
            if not destination.parent.is_dir():
                destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(image_path, destination)
            #print(image_path, destination)

    shutil.rmtree(data_root)


def data_setup():
    data_path = Path("./src/data/")
    image_path = data_path / "noodles"

    train_dir = image_path / 'train'
    test_dir = image_path / 'test'

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    
    base_transform = weights.transforms()
    base_transform
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    # Create datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(train_dir, transform=base_transform)

    # Define split ratio (2/10)
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size

    indices = list(range(len(train_dataset)))

    # Create the splits
    train_indices, val_indices = torch.utils.data.random_split(
        indices,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    # Create subsets for training and validation
    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)
    # Check the lengths of the datasets
    print(f"Training dataset size: {len(train_subset)}")
    print(f"Validation dataset size: {len(val_subset)}")

    # Create data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
    )

    test_dataloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(test_dir, transform=base_transform),
        batch_size=64,
        shuffle=False,
        num_workers=4,
    )
    # Check the class names
    class_names = test_dataloader.dataset.classes
    return train_dataloader, val_dataloader, test_dataloader, class_names