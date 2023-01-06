import datetime
import json
from pathlib import Path
from pathlib import PosixPath
from typing import List
from typing import TypeVar
import shutil

import PIL.Image
import cv2
import numpy
import numpy as np
from PIL import Image

MyImageType = TypeVar('MyImageType', str, Path, PIL.Image.Image, numpy.array)


from pycode.constants import ROOT

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


def show_image(im: MyImageType, title: str = None) -> str:
    """
    Load and show (or not) image using cv2. Couldn't find neither for PIL show() nor plt.imshow() how to reuse
    the same figure. This works in cv2 by simply giving the same window name.
    """
    im = load_image(im)
    print('Image is shown on desktop.')
    cv2.imshow(title if title else 'pic-display', im)
    print('Press any button to continue (if needed click on an image window too; Q to break).')
    key_pressed = chr(cv2.waitKey(0))  # 0 to wait for user input, >0 for milliseconds to wait
    return key_pressed


def show_images(imgs: List[MyImageType], title=None):
    """
    Show many images in the same window.
    """
    for img in imgs:
        key_pressed = show_image(img, title)
        print(key_pressed)
        if key_pressed == 'Q':
            break


def overlay_two_images(im: MyImageType, mask: MyImageType) -> numpy.array:
    """
    Overlay image and mask using blend, alpha=0.3.
    """
    im = load_image(im)
    mask = load_image(mask)

    # convert to PIL.Image.Image
    im = Image.fromarray(im)
    mask = Image.fromarray(mask)

    new_img = Image.blend(im, mask, 0.3)
    return np.array(new_img)


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


# def store_based_on_user_input(input_filename):
#     """
#     Categorize images based on an input_filename categorizations
#
#     Args:
#         input_filename:
#     """
#     if not os.path.exists(input_filename):
#         raise FileNotFoundError()
#     solar_module_types = read_csv_file_into_dict(input_filename)
#
#     for image_filename, solar_module_type in solar_module_types.items():
#         # locate img and mask
#         solar_module_type = solar_module_type.strip()
#         im_path = os.path.join(KIEWIT, MODULES_FOLDER, 'imgs', image_filename)
#         mask_path = os.path.join(KIEWIT, MODULES_FOLDER, 'masks', image_filename)
#
#         new_im_path = os.path.join(KIEWIT, MODULES_FOLDER, 'per_type', 'type_' + solar_module_type, 'imgs', image_filename)
#         new_mask_path = os.path.join(KIEWIT, MODULES_FOLDER, 'per_type', 'type_' + solar_module_type, 'masks', image_filename)
#
#         os.makedirs(os.path.dirname(new_im_path), exist_ok=True)
#         os.makedirs(os.path.dirname(new_mask_path), exist_ok=True)
#         shutil.copyfile(im_path, new_im_path)
#         shutil.copyfile(mask_path, new_mask_path)

def test_show_images(image_paths, mask_paths):
    im = image_paths[0]
    mask = mask_paths[0]
    show_image(im, 'image')
    show_images(image_paths[:4], 'images')
    show_image(overlay_two_images(im, mask), 'overlay')
    show_image(concat_images(im, mask), 'concatenated')
    show_image(concat_n_images([im, mask, overlay_two_images(im, mask)]), 'concatenated')


def load_paths():
    imgs_folder = LABELS / 'racks/imgs'
    masks_folder = LABELS / 'racks/masks'
    image_paths = list(imgs_folder.glob('*.png'))
    mask_paths = list(masks_folder.glob('*.png'))
    return image_paths, mask_paths


def classify(image_paths: List[MyImageType], mask_paths: List[MyImageType]):
    """
    Go over all images and assign a one-char label. Dump results into json file.
    """
    logs = {}
    for im, mask in zip(image_paths, mask_paths):
        key_pressed = show_image(concat_n_images([im, mask, overlay_two_images(im, mask)]), 'images')
        if key_pressed == 'Q':
            break
        logs[im.name] = key_pressed
    prefix = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'{prefix}_user_qc.json', 'w') as f:
        json.dump(logs, f, indent=4)


def create_new_folders():
    racks_path = Path(LABELS) / 'racks'
    # create all folders first
    for value in set(results.values()):
        new_dir = racks_path / value
        new_dir.mkdir(parents=True, exist_ok=True)
        (new_dir / 'imgs').mkdir(parents=True, exist_ok=True)
        (new_dir / 'masks').mkdir(parents=True, exist_ok=True)


def copy_all_files():
    for name, label in results.items():
        src_path = racks_path / 'imgs' / name
        dst_path = racks_path / label / 'imgs' / name
        shutil.copy(src_path, dst_path)

        src_path = racks_path / 'masks' / name
        dst_path = racks_path / label / 'masks' / name
        shutil.copy(src_path, dst_path)


if __name__ == '__main__':
    image_paths, mask_paths = load_paths()
    # test_show_images(image_paths, mask_paths)
    # classify(image_paths, mask_paths)

    # open json file and copy all the images into a folder based on their label
    with open('20230105_163949_user_qc.json', 'r') as f:
        results = json.load(f)
    print(results)

    racks_path = Path(LABELS) / 'racks'
    # create_new_folders()
    # copy_all_files()


    # imgs_folder = LABELS / 'racks/g/imgs'
    # masks_folder = LABELS / 'racks/g/masks'
    # image_paths = list(imgs_folder.glob('*.png'))
    # mask_paths = list(masks_folder.glob('*.png'))
    # classify(image_paths, mask_paths)


