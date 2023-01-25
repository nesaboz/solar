from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from categorize import overlay_two_images
from processing import BigImage
from processing import TransformedTensorDataset
from step_by_step import InverseNormalize


def convert_class_mask_to_rgb_image(im, n_classes):
    """
    im is of shape (h, w) with each pixel being assigned one of the values in (0, n_clases-1) range.
    We want to convert classes to colors.
    
    """
    h, w = im.shape
    b = np.zeros([h, w, 3], dtype='uint8')
    for class_id in range(1, n_classes):
        if class_id == 1:
            b[:, :, 0] += np.array((im == class_id) * 255, dtype='uint8')   # convert racks to red
        elif class_id == 2:
            b[:, :, 1] += np.array((im == class_id) * 255, dtype='uint8')   # convert common panel to green
        elif class_id == 3:
            b[:, :, 2] += np.array((im == class_id) * 255, dtype='uint8')   # convert dense panel to  blue
        elif class_id == 4:  # not used
            b[:, :, 0] += np.array((im == class_id) * 255, dtype='uint8')   # in case of an extra class to yellow
            b[:, :, 1] += np.array((im == class_id) * 255, dtype='uint8')
    return b


def save_image(im: Image, filepath: Path, overwrite: bool = False):
    if filepath.exists() and not overwrite:
        print('File exists and overwrite flag is False')
        return
    filepath.parent.mkdir(parents=True, exist_ok=True)
    im.save(filepath, "PNG")


def display_images(x_batch, y_batch, y_pred_batch, n_classes, normalizer,
                   idx_map=None, idx_offset=0, save=False, overwrite=False,
                   alpha=0.3, run_folder=None):
    if len(x_batch) > 64:
        raise Warning(f"There are {len(x_batch)} images to show, I set the limit to 64.")
        return

    inv_normalizer = InverseNormalize(normalizer)

    for idx in range(len(x_batch)):
        # convert to PIL images
        prediction = ToPILImage()(convert_class_mask_to_rgb_image(y_pred_batch[idx, :, :], n_classes))
        label_mask = ToPILImage()(convert_class_mask_to_rgb_image(y_batch[idx, :, :], n_classes))
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


def evaluate_unlabeled(unlabeled_tensor_x, unlabeled_tensor_y, val_composer, n_classes, normalizer,
                       sbs, run_folder, idx_map):
    
    unlabeled_dataset = TransformedTensorDataset(unlabeled_tensor_x, unlabeled_tensor_y, transform=val_composer)
    batch_size = 32
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size)

    print('Evaluating ... ')
    unlabeled_loader_iter = iter(unlabeled_loader)
    batch_idx = 0
    for x_unlabeled_batch, y_unlabeled_batch in tqdm(unlabeled_loader_iter):
        y_unlabeled_pred_batch = sbs.predict(x_unlabeled_batch, to_numpy=False).argmax(1)
        display_images(x_unlabeled_batch, y_unlabeled_batch, y_unlabeled_pred_batch, n_classes, normalizer,
                       idx_map, idx_offset=batch_idx * batch_size, save=True, overwrite=True,
                       run_folder=run_folder)
        batch_idx += 1

    print('Stitching ... ')
    for tag in ['R1C1', 'R1C2', 'R2C1', 'R2C2']:
        big_image = BigImage(tag)

        big_image.stitch_images(run_folder / 'predicted',
                                run_folder / (big_image.tag + '_predicted.png'))

        big_image.stitch_images(run_folder / 'overlayed',
                                run_folder / (big_image.tag + '_overlayed.png'))
