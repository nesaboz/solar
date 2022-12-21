from typing import List, Union, Dict, Tuple
from pathlib import Path
from collections import OrderedDict
from itertools import cycle
import random

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A


def imread(img_path: Union[Path, str], mode: str = 'color') -> np.ndarray:
    """
    Loads an image from a given path.
    """
    if isinstance(img_path, Path):
        img_path = str(img_path)
    if mode == 'color':
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif mode == 'gray':
        img = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
        # _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)        
    return img


def to_scaled_tensor(examples: Dict[str, torch.Tensor], norm: int = 1) -> Dict[str, torch.Tensor]:
    for k, v in examples.items():
        if k == 'mask':
            # Sharpen masks so they become binary once `norm` is used
            v = np.where(v > 100, 255, 0)
        v = v / float(norm)
        # change HWC to CHW for training in pytorch
        v = np.moveaxis(v, -1, 0)
        v = torch.Tensor(v).to(dtype=torch.float32)
        examples[k] = v
    return examples


class TransformedTensorDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        x = self.x[index]

        if self.transform:
            x = self.transform(x)

        return x, self.y[index]

    def __len__(self):
        return len(self.x)


class SegmentationMapDataset(Dataset):
    """
    Used to hold a dataset for training via the `torch.utils.data.DataLoader` class.
    """
    def __init__(self, 
                 images: List[Path], 
                 masks: List[Path] = None,
                 transforms=None, 
                 mode: str = 'color') -> None:
        self.images = images
        self.masks = masks
        self.transforms = transforms
        self.mode = mode

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        image = imread(self.images[idx])
        mask_path = self.masks[idx]
        # Training Modes
        if self.masks and self.transforms is not None:
            mask = imread(mask_path)
            examples = self.transforms(image=image, mask=mask)
            result = to_scaled_tensor(examples, norm=1)
        elif self.masks and self.transforms is None:
            mask = imread(mask_path)
            examples = {'image': image, 'mask': mask}
            result = to_scaled_tensor(examples, norm=255)
        # There is never a situation where it should have transforms 
        # without masks so here we just use `else` (b/c transforms and masks
        # are both only used during training)
        else:
            # Prediction mode
            examples = {'image': image}
            result = to_scaled_tensor(examples, norm=255)
        return result


def build_map_dataloaders(images: List[Path], 
                          masks: List[Path],
                          random_state: int,
                          valid_size: float = 0.1,
                          mode: str = 'color',
                          batch_size: int = 1,
                          num_workers: int = 0,
                          train_transforms_fn=None,
                          valid_transforms_fn=None) -> OrderedDict:
    """
    Build dataloaders for both train set and validation set.
    """
    indices = np.arange(len(images))
    # Use indices to split the dataset randomly
    train_indices, val_indices = train_test_split(
        indices, test_size=valid_size, random_state=random_state, shuffle=True
    )
    images = np.array(images)
    masks = np.array(masks)

    # Create dataset objects to be used in dataloaders
    train_dataset = SegmentationMapDataset(
        images=images[train_indices].tolist(),
        masks=masks[train_indices].tolist(),
        transforms=train_transforms_fn,
        mode=mode
    )
    valid_dataset = SegmentationMapDataset(
        images=images[val_indices].tolist(),
        masks=masks[val_indices].tolist(),
        transforms=valid_transforms_fn,
        mode=mode
    )
    # Create the dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    # Using OrderedDict here to comply with Catalyst library
    loaders = OrderedDict()
    loaders['train'] = train_loader
    loaders['valid'] = valid_loader

    return loaders    


if __name__ == '__main__':
    proj_dir = Path('.').resolve()
    data_dir = proj_dir / 'data'
    train_imgs = data_dir / 'all_labels/type_1/imgs'
    train_masks = data_dir / 'all_labels/type_1/masks'
    images = sorted([p for p in train_imgs.iterdir()])
    masks = sorted([p for p in train_masks.iterdir()])


    # Let's
    # let's get train and valid indices:
    torch.manual_seed(13)
    N = len(x_tensor)
    n_train = int(.8 * N)
    n_val = N - n_train
    train_subset, val_subset = random_split(x_tensor, [n_train, n_val])
    train_subset


    transforms = [
#        A.RandomRotate90(),
        A.HueSaturationValue(p=0.3),
    ]
    transforms = A.Compose(transforms)

    loaders = build_map_dataloaders(images, masks, 42, batch_size=16, train_transforms_fn=transforms)
    # for _ in range(5):
    #     result = next(iter(loaders['train']))
    #     print(result.keys())