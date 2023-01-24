import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import torch
torch.manual_seed(10)
import torch.optim as optim
import torch.nn as nn
import platform
import matplotlib.ticker as ticker

from PIL import Image
from PIL.ImageStat import Stat
import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler, SubsetRandomSampler
from torchvision.transforms import Compose, ToTensor, Normalize, RandomApply, ColorJitter, ToPILImage
import ipyplot
from torchmetrics import JaccardIndex
from torchmetrics.functional import jaccard_index
from processing import BigImage, prep_data
import datetime
import json
from constants import ROOT, RUNS_FOLDER
from processing import TransformedTensorDataset


from step_by_step import StepByStep, InverseNormalize, load_tensor, get_means_and_stdevs
from categorize import check_for_missing_files, LABELS, show_image, overlay_two_images
from models import Segnet
from pathlib import Path

import PIL
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose, ToTensor
from constants import ROOT

from step_by_step import StepByStep
from categorize import check_for_missing_files


def convert_class_mask_to_rgb_image(im):
    b = np.zeros([im.shape[0], im.shape[1], 3], dtype='uint8')
    for class_id in range(3):
        b[:, :, class_id] = np.array((im == (class_id + 1)) * 255, dtype='uint8')
    return b


def save_image(im: Image, filepath: Path, overwrite: bool = False):
    if filepath.exists() and not overwrite:
        print('File exists and overwrite flag is False')
        return
    filepath.parent.mkdir(parents=True, exist_ok=True)
    im.save(filepath, "PNG")


def display_images(x_batch, y_batch, y_pred_batch, normalizer,
                   idx_map=None, idx_offset=0, save=False, overwrite=False,
                   alpha=0.3, run_folder=None):
    if len(x_batch) > 64:
        raise Warning(f"There are {len(x_batch)} images to show, I set the limit to 64.")
        return

    inv_normalizer = InverseNormalize(normalizer)

    for idx in range(len(x_batch)):
        # convert to PIL images
        prediction = ToPILImage()(convert_class_mask_to_rgb_image(y_pred_batch[idx, :, :]))
        label_mask = ToPILImage()(y_batch[idx, :, :].float())
        original = ToPILImage()(inv_normalizer(x_batch[idx, :, :, :]))
        overlayed = overlay_two_images(original, prediction, alpha=alpha, to_numpy=False)

        name = '' if not idx_map else idx_map[idx_offset + idx]
        if not save:
            fig, ax = plt.subplots(1, 4)
            ax[0].imshow(prediction)
            ax[0].set_title(f"{idx_offset + idx} - {name}")
            ax[1].imshow(label_mask)
            ax[2].imshow(original)
            ax[3].imshow(overlayed)
            for i in range(4):
                ax[i].grid(False)
                ax[i].xaxis.set_major_locator(ticker.NullLocator())
                ax[i].yaxis.set_major_locator(ticker.NullLocator())
            plt.show()
        else:
            if not idx_map:
                raise ValueError('If save=True, idx_map must be provided.')
            save_image(prediction, run_folder / 'predicted' / name, overwrite)
            save_image(overlayed, run_folder / 'overlayed' / name, overwrite)


def evaluate_unlabeled(unlabeled_tensor_x, unlabeled_tensor_y, val_composer, normalizer,
                       sbs, run_folder, idx_map):
    unlabeled_dataset = TransformedTensorDataset(unlabeled_tensor_x, unlabeled_tensor_y, transform=val_composer)
    batch_size = 32
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size)

    unlabeled_loader_iter = iter(unlabeled_loader)
    batch_idx = 0
    for x_unlabeled_batch, y_unlabeled_batch in tqdm(unlabeled_loader_iter):
        y_unlabeled_pred_batch = sbs.predict(x_unlabeled_batch, to_numpy=False).argmax(1)
        display_images(x_unlabeled_batch, y_unlabeled_batch, y_unlabeled_pred_batch, normalizer,
                       idx_map, idx_offset=batch_idx * batch_size, save=True, overwrite=True,
                       run_folder=run_folder)
        batch_idx += 1

    big_image = BigImage('R1C1')

    big_image.stitch_images(run_folder / 'predicted',
                            run_folder / (big_image.tag + '_predicted.png'))

    big_image.stitch_images(run_folder / 'overlayed',
                            run_folder / (big_image.tag + '_overlayed.png'))
