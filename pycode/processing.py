import os
import re
from math import cos
from math import degrees
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

PIL.Image.MAX_IMAGE_PIXELS = 339738624

# from index.html file
CONFIG = {
    'id': '6126758101',
    'folder': 'IMG_PHR1B_PMS-N_001',
    'prefix': 'IMG_PHR1B_PMS-N_202112241708104_ORT_6126758101_',
    'extension': 'TIF',
    'coordinates': {
        'R1C1': (-95.41759381264909, 33.50642345814874),
        'R1C2': (-95.22180129867475, 33.51009141565303),
        'R2C1': (-95.21864991285545, 33.38646550386663),
        'R2C2': (-95.41416534766181, 33.38281461688832)
    }
}


def find_new_coordinates(
    lat: float, long: float, x_meters: float, y_meters: float
) -> tuple:
    lat += y_meters / 111_111
    long += x_meters / degrees(111_111 * cos(10))
    return lat, long


def bbox_from_point(
    lat: float, long: float, pixels: int, pixel_size: float = 0.5
) -> list:
    length = pixel_size * pixels
    new_lat, new_long = find_new_coordinates(lat, long, length, length)
    return list((long, lat, new_long, new_lat))


def get_pixel_bbox(row: int, col: int, pixels: int) -> tuple:
    left = col * pixels
    top = row * pixels
    right = left + pixels
    bottom = top + pixels
    return left, top, right, bottom


def find_all_black(img: Image) -> float:
    if not img.getbbox():
        return 1
    else:
        return 0


class BigImage(object):
    """

    Args:
        tag (str): like 'R2C2'
        image_filename = 'IMG_PHR1B_PMS-N_202112241708104_ORT_6126758101_' + tag + '.TIF'
        image_pathname (Path): Path to the image.
        coordinates ((float, float)): Latitude and longitude.
        output_folder: folder with cropped images.
        tag: 'R1C1' for example.

    """
    def __init__(self, tag, config=CONFIG):
        self.id = config['id']
        self.folder = config['folder']
        self.tag = tag
        self.image_dirname = Path(ROOT) / 'data' / self.id / self.folder
        self.image_filename = f"{config['prefix']}{tag}.{config['extension']}"
        self.coordinates = config['coordinates'][tag]
        self.image_pathname = self.image_dirname / self.image_filename
        self.cropped_folder = self.image_dirname / 'cropped'
        self.cropped_info_file = Path(ROOT) / 'data' / (tag + '_cropped_images_info.csv')

    @property
    def size(self):
        
        if self.image_pathname.exists():
            with Image.open(self.image_pathname) as im:
                return im.size
        print('File was not found, using defaults.')
        sizes = {'R1C1': (18432, 18432), 'R1C2': (17968, 18432), 'R2C1': (18432, 9002), 'R2C2': (17968, 9002)}
        return sizes[self.tag]

    def _crop_and_write(self,
                        img: Image,
                        img_dir: Path,
                        start_lat: float,
                        start_long: float,
                        prefix: str,
                        pixels: int = 256,
                        ) -> pd.DataFrame:
        columns = [
            'img_name',
            'left_bound',
            'upper_bound',
            'right_bound',
            'lower_bound',
            'grid_row',
            'grid_col',
            'x_pixels',
            'y_pixels'
        ]
        df = pd.DataFrame(columns=columns)
        lat, long = start_lat, start_long
        next_lat, next_long = start_lat, start_long
        across, down = img.size
        across /= pixels
        down /= pixels
        for row in tqdm(range(int(down)), total=int(down)):
            bbox = bbox_from_point(next_lat, next_long, pixels)
            next_lat, next_long = bbox[3], bbox[0]
            for col in range(int(across)):
                bbox = bbox_from_point(lat, long, pixels)
                if row == 0 and col == 0:
                    lat, long = bbox[1], bbox[2]
                elif col == 0:
                    lat, long = next_lat, next_long
                else:
                    lat, long = bbox[1], bbox[2]
                pixel_bbox = get_pixel_bbox(row, col, pixels)
                img_crop = img.crop(pixel_bbox)
                img_id = f"{prefix}_row{row}_col{col}"
                filepath = img_dir / f"{img_id}.png"
                if img_crop.getbbox() is None:  # cropped image is all black
                    continue
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                img_crop.save(filepath)
                dat = {
                    'img_name': img_id,
                    'left_bound': bbox[0],
                    'upper_bound': bbox[1],
                    'right_bound': bbox[2],
                    'lower_bound': bbox[3],
                    'grid_row': row,
                    'grid_col': col,
                    'x_pixels': pixels,
                    'y_pixels': pixels
                }
                ser = pd.Series(dat)
                df = pd.concat([df, ser.to_frame().T], ignore_index=True)
        return df

    def crop_large_image(self):
        """
        Crops big image into smaller ones, saves them in output_folder. Also save csv file with all the info.

        Returns:
            pathname: cropped_images_info.csv (created in the parent folder)
        """
        with Image.open(self.image_pathname) as img:
            start_long, start_lat = self.coordinates
            df = self._crop_and_write(img, self.cropped_folder, start_lat, start_long, prefix=self.tag)
            df.to_csv(self.cropped_info_file)
        print('Done cropping')

    def stitch_images(self, cropped_images_folder, output_path: Path = None):
        """
        Find all images with a self.tag in the folder, extracts row and col and stitches them.
        """

        print('Expect 1-2 minutes ...')
        if output_path is None:
            output_path = self.image_dirname / (self.tag + '.png')

        if not isinstance(cropped_images_folder, Path):
            cropped_images_folder = Path(cropped_images_folder)

        stitched_image = Image.new('RGB', self.size)
        df = pd.read_csv(self.cropped_info_file)
        single_height = df['x_pixels'].iloc[0]
        single_width = df['y_pixels'].iloc[0]

        for cropped_image_path in tqdm(cropped_images_folder.glob(f'*{self.tag}*.png')):
            # extract row and column from the name:
            m = re.match(r".*row(\d+)_col(\d+).*.png", str(cropped_image_path))
            row = int(m.group(1))
            col = int(m.group(2))

            # open an image and paste it into the corresponding place
            with Image.open(cropped_image_path) as im:
                stitched_image.paste(im, (single_width * col, single_height * row))

        stitched_image.save(output_path)
        print(f'Done stitching to: {output_path}')


def crop_4_images():
    for tag in ['R1C1', 'R1C2', 'R2C1', 'R2C2']:
        big_image = BigImage(tag)
        big_image.crop_large_image()


def test_stitch_images():
    # stitch back images
    big_image = BigImage('R1C2')
    big_image.stitch_images(big_image.image_dirname / 'cropped')


def _get_tensors(image_paths, mask_paths, title_mapping):
    n_channels, h, w = ToTensor()(Image.open(image_paths[0]).convert('RGB')).shape

    x_tensor = torch.zeros([len(image_paths), n_channels, h, w])
    y_tensor = torch.zeros([len(image_paths), h, w], dtype=torch.long)  # we'll have only class number mask

    for i, (image_path, mask_path) in enumerate(zip(tqdm(image_paths), mask_paths)):
        x_tensor[i, :, :, :] = ToTensor()(Image.open(image_path).convert('RGB'))

        mask = Image.open(mask_path).convert('RGB')
        mask_tensor = ToTensor()(mask)

        # let's extracts only relevant channel (0 - red for rack, 2 - blue for panel):
        # we don't have mixed labels
        name = image_path.name
        if 'commonrack' in name:
            mask_tensor = mask_tensor[0, :, :]
            class_id = title_mapping['commonrack']
        elif 'commonpanel' in name:
            mask_tensor = mask_tensor[2, :, :]
            class_id = title_mapping['commonpanel']
        elif 'denserack' in name:
            mask_tensor = mask_tensor[2, :, :]
            class_id = title_mapping['denserack']
        elif 'densepanel' in name:
            mask_tensor = mask_tensor[2, :, :]
            class_id = title_mapping['densepanel']
        else:
            # this is pure zeros
            mask_tensor = mask_tensor[0, :, :]
            class_id = title_mapping['background']

        # make it binary
        mask_tensor = (1 - mask_tensor < mask_tensor.max()).long()  # this line has given me issues several times now due to some corner pixels

        # multiply with class_id
        mask_tensor *= class_id

        y_tensor[i, :, :] = mask_tensor

    return x_tensor, y_tensor



def get_labeled_tensors(data_dir, title_mapping):

    labeled_imgs_dir = data_dir / 'labeled/imgs'
    labeled_masks_dir = data_dir / 'labeled/masks'

    image_paths = labeled_imgs_dir.glob('*.png')
    mask_paths = labeled_masks_dir.glob('*.png')

    # some weird bug where linux finds extra files that start with '.'
    image_paths = filter(lambda x: x.name[0] != '.', image_paths)
    mask_paths = filter(lambda x: x.name[0] != '.', mask_paths)

    image_paths = sorted(list(image_paths))
    mask_paths = sorted(list(mask_paths))
    
    labeled_idx_map = {i: image_path.name for i, image_path in enumerate(image_paths)}
    
    check_for_missing_files(image_paths, mask_paths)
    
    labeled_tensor_x, labeled_tensor_y = _get_tensors(image_paths, mask_paths, title_mapping)

    return labeled_tensor_x, labeled_tensor_y, labeled_idx_map



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


def get_weights(train_tensor_y, n_classes):
    contributions = np.array([(train_tensor_y == i).float().sum() for i in range(n_classes)])
    weights = np.zeros(n_classes)
    for i, contrib in enumerate(contributions):
        if contrib != 0:
            weights[i] = 1.0 / contrib
    weights = weights / weights.sum() * n_classes
    weights = torch.tensor(weights, dtype=torch.float)  # required by the CrossEntropyLoss
    return weights


def get_unlabeled_tensors(data_dir, shape):
    unlabeled_imgs_dir = data_dir / 'unlabeled'
    unlabeled_image_paths = unlabeled_imgs_dir.glob('*.png')
    unlabeled_image_paths = filter(lambda x: x.name[0] != '.', unlabeled_image_paths)
    unlabeled_image_paths = sorted(list(filter(
        lambda x: 'R1C1' in x.name, unlabeled_image_paths)))
    N = len(unlabeled_image_paths)
    n_channels, h, w = shape
    unlabeled_tensor_x = torch.zeros([N, n_channels, h, w])
    unlabeled_tensor_y = torch.zeros([N, h, w], dtype=torch.long)  # we'll have only class number mask
    unlabeled_idx_map = {}
    for i, image_path in enumerate(tqdm(unlabeled_image_paths[:N])):
        unlabeled_tensor_x[i, :, :, :] = ToTensor()(Image.open(image_path))
        unlabeled_tensor_y[i, :, :] = torch.zeros(*unlabeled_tensor_x.shape[2:])
        unlabeled_idx_map[i] = image_path.name

    return unlabeled_tensor_x, unlabeled_tensor_y, unlabeled_idx_map


def prep_data(labeled_tensor_x, labeled_tensor_y, labeled_idx_map, applier, n_classes, batch_size=32):
    
    torch.manual_seed(13)
    N = len(labeled_tensor_x)
    n_train = int(.85 * N)
    n_val = N - n_train
    train_subset, val_subset = random_split(TransformedTensorDataset(labeled_tensor_x, labeled_tensor_y), [n_train, n_val])

    train_idx = train_subset.indices
    val_idx = val_subset.indices

    train_tensor_x = labeled_tensor_x[train_idx]
    train_tensor_y = labeled_tensor_y[train_idx]
    val_tensor_x = labeled_tensor_x[val_idx]
    val_tensor_y = labeled_tensor_y[val_idx]
    
    val_idx_map = {i: labeled_idx_map[idx] for i, idx in enumerate(val_idx)}

    torch.manual_seed(13)
    temp_train_dataset = TransformedTensorDataset(train_tensor_x, train_tensor_y, transform=applier)
    temp_train_loader = DataLoader(temp_train_dataset, batch_size=batch_size)
    normalizer = StepByStep.make_normalizer(temp_train_loader)

    train_composer = Compose([applier, normalizer])  # first is jitter with RandomApply, then normalizer!
    val_composer = Compose([normalizer])

    train_dataset = TransformedTensorDataset(train_tensor_x, train_tensor_y, transform=train_composer)
    val_dataset = TransformedTensorDataset(val_tensor_x, val_tensor_y, transform=val_composer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    weights = get_weights(train_tensor_y, n_classes)

    return train_loader, val_loader, val_idx_map, normalizer, val_composer, weights


if __name__ == '__main__':
    pass

