import datetime
import json
import os
import shutil
from pathlib import Path
from pathlib import PosixPath
from typing import List
from typing import TypeVar

import PIL.Image
import cv2
import numpy
import numpy as np
from PIL import Image

MyImageType = TypeVar('MyImageType', str, Path, PIL.Image.Image, numpy.array)

from constants import ROOT

LABELS = Path(ROOT) / 'data/all_labels'


def load_image(im: MyImageType) -> numpy.array:
    """
    Load image from any format into numpy.array.
    """
    if isinstance(im, (str, Path, PosixPath)):  # if str or Path load image
        im = cv2.imread(str(im))
    elif isinstance(im, Image.Image):  # convert PIL.Image.Image to numpy.array
        im = np.array(im)
    return im


def show_image(im: MyImageType, title: str = None, prompt=True) -> str:
    """
    Load and show (or not) image using cv2. Couldn't find neither for PIL show() nor plt.imshow() how to reuse
    the same figure. This works in cv2 by simply giving the same window name.
    """
    if prompt:
        print('Image is shown on desktop.')
        print('Press any button to close.')
    im = load_image(im)
    cv2.imshow(title if title else 'pic-display', im)
    key_pressed = chr(cv2.waitKey(0))  # 0 to wait for user input, >0 for milliseconds to wait
    return key_pressed


def show_images(imgs: List[MyImageType], title=None):
    """
    Show many images in the same window.
    """
    print('Images are shown on desktop.')
    print('Press any button to go to the next (if needed click on an image window too; Q (capital) to break).')
    for img in imgs:
        key_pressed = show_image(img, title)
        print(key_pressed)
        if key_pressed == 'Q':
            break


def overlay_two_images(im: MyImageType, mask: MyImageType, alpha=0.5, to_numpy=True) -> numpy.array or PIL.Image.Image:
    """
    Overlay image and mask using blend, alpha=0.3.
    """
    im = load_image(im)
    mask = load_image(mask)

    # convert to PIL.Image.Image
    im = Image.fromarray(im)
    mask = Image.fromarray(mask)

    new_img = Image.blend(im, mask, alpha)
    return np.array(new_img) if to_numpy else new_img


def concat_images(imga, imgb, vertical=False, gap=10):
    """
    Combines two ndarrays horizontally or vertically.

    Args:
        imga (ndarray): Image A.
        imgb (ndarray): Image B
        vertical (bool): If True stitch vertically.
        gap (pixels): Gap between stitched images.

    Returns:
        (ndarray): Stitched image.
    """
    imga = load_image(imga)
    imgb = load_image(imgb)

    if len(imga) == 0:  # special case of empty image
        ha, wa, ca = 0, 0, 0
    else:
        try:
            ha, wa, ca = imga.shape
        except ValueError:
            ha, wa = imga.shape
            ca = 1

    try:
        hb, wb, cb = imgb.shape
    except ValueError:
        hb, wb = imgb.shape
        cb = 1

    # make a new image `c`
    if vertical:
        hc = ha + hb + gap
        wc = np.max([wa, wb])
    else:
        hc = np.max([ha, hb])
        wc = wa + wb + gap
    cc = max(ca, cb)
    imgc = np.zeros([hc, wc, cc], dtype=np.uint8)

    if len(imga) != 0:
        imgc[:ha, :wa, :] = imga

    if vertical:
        imgc[ha + gap:ha + gap + hb, :wb, :] = imgb
    else:
        imgc[:hb, wa + gap:wa + gap + wb, :] = imgb

    return imgc


def create_matrix_from_list(n, m):
    """
    Takes an array of length n and chops it into segments of max length of m.
    Used for image stitching.

    For example n=10 and m=3 gives [[0,1,2], [3,4,5], [6,7,8], [10]]

    Args:
        n (int): number of indices.
        m (int): max length of subarray.

    Returns:
        list(list): List of chopped segments.

    """
    result = []
    current_row = []
    for counter in range(n):
        if counter > 0 and counter % m == 0:
            result.append(current_row)
            current_row = []
        current_row.append(counter)

    if current_row:
        result.append(current_row)

    return result


def concat_n_images(image_list, max_in_a_row=None, gap=10):
    """
    Combines images in the image_path_list.

    Args:
        image_list (list(str)): List of images to stitch.
        max_in_a_row (int): If more than max_in_a_row images are provided images are stitched into next row.

    Returns:
        (ndarray): Stitched image.

    """
    if not max_in_a_row:
        max_in_a_row = len(image_list)

    matrix = create_matrix_from_list(len(image_list), max_in_a_row)

    result = np.array([])
    for row in matrix:
        current = np.array([])
        for index in row:
            current = concat_images(current, image_list[index], vertical=False, gap=gap)
        result = concat_images(result, current, vertical=True, gap=gap)

    return result


def load_paths(label_type):
    imgs_folder = LABELS / label_type / 'imgs'
    masks_folder = LABELS / label_type / 'masks'
    image_paths = list(imgs_folder.glob('*.png'))
    mask_paths = list(masks_folder.glob('*.png'))
    return image_paths, mask_paths


def classify(image_paths: List[MyImageType], mask_paths: List[MyImageType], label_type):
    """
    Go over all images and assign a one-char label. Dump results into json file.
    """
    print('Images are shown on desktop.')
    print('Press any button to go to the next (if needed click on an image window too; Q (capital) to break).')

    logs = {}
    for im, mask in zip(image_paths, mask_paths):
        key_pressed = show_image(concat_n_images([im, mask, overlay_two_images(im, mask)]), label_type, prompt=False)
        if key_pressed == 'Q':
            break
        logs[im.name] = key_pressed
    prefix = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    user_qc_file = LABELS / label_type / f'{prefix}_{label_type}.json'
    with open(user_qc_file, 'w') as f:
        json.dump(logs, f, indent=4)
    print(f'Generated {user_qc_file}.')
    return user_qc_file


def _create_new_folders(new_folder_path):
    new_folder_path.mkdir(parents=True, exist_ok=True)
    print(f'Created {new_folder_path}.')
    imgs_folder = new_folder_path / 'imgs'
    masks_folder = new_folder_path / 'masks'
    imgs_folder.mkdir(parents=True, exist_ok=True)
    masks_folder.mkdir(parents=True, exist_ok=True)
    print(f'Created {imgs_folder}.')
    print(f'Created {masks_folder}.')


def create_new_folders(label_type, new_folder_names):
    folder_path = LABELS / label_type
    for new_folder_name in new_folder_names:
        new_folder_path = folder_path / new_folder_name
        _create_new_folders(new_folder_path)


def copy_all_files(user_qc, labels_path: Path):
    counter = 0
    for name, label in user_qc.items():
        src_path = labels_path / 'imgs' / name
        dst_path = labels_path / label / 'imgs' / name
        shutil.copy(src_path, dst_path)

        src_path = labels_path / 'masks' / name
        dst_path = labels_path / label / 'masks' / name
        shutil.copy(src_path, dst_path)
        counter += 1
    print(f'Copied 2 x {counter//2} files.')


def check_for_missing_files(image_paths, mask_paths):
    a = set([x.name for x in image_paths])
    b = set([x.name for x in mask_paths])
    extra = a ^ b
    if not (len(extra) == 0 and len(a) == len(b)):
        raise AssertionError(f'Following is extra in images:\n{a-b}\nand in masks:\n{b-a}')
        # TODO print(f'There are {len(a)} labels total. Do you want to delete files in question?') ... input() ... os.remove()


def rename_all_files(label_type, old_name, new_name):
    # rename all files under label_type
    counter = 0
    panels_path = LABELS / label_type
    for src in panels_path.glob('*/*.*'):
        dst = src.parent / src.name.replace(old_name, new_name)
        try:
            os.rename(src, dst)
        except:
            break
        counter += 1
    print(f'Renamed {counter} files.')


def test_show_images():
    label_type = 'labeled_data'
    image_paths, mask_paths = load_paths(label_type)
    idx = 10
    im = image_paths[idx]
    mask = mask_paths[idx]
    show_image(im, 'image')
    show_images(image_paths[:4], 'images')
    show_image(overlay_two_images(im, mask), 'overlay')
    show_image(concat_images(im, mask), 'concatenated')
    show_image(concat_n_images([im, mask, overlay_two_images(im, mask)]), 'concatenated')
    show_image(overlay_two_images(im, mask, False), 'overlayed')


if __name__ == '__main__':
    test_show_images()
