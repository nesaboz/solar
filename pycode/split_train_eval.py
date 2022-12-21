import os
import shutil
from datetime import datetime
from glob import glob

import numpy as np
from tqdm import tqdm

from constants import KIEWIT


def get_mask_pathname_from_image(im_pathname):
    mask_pathname = os.path.join(os.path.dirname(os.path.dirname(im_pathname)), 'masks',
                                 os.path.basename(im_pathname))
    return mask_pathname if os.path.exists(mask_pathname) else None


class SplitTrainEval:
    """
    project_name (str): for example 'type_1'
    training_data_folder (str): path to the training data containing folders imgs and masks
    eval_split (float): fraction of eval split, 0.2 = 20%
    """

    def __init__(self, project_name, training_data_folder, eval_split=0.2):
        self.project_name = project_name
        self.input_dir = training_data_folder
        current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        self.output_dir = os.path.join(KIEWIT, 'projects', project_name, 'train_test_split', current_time)
        self.test_split = eval_split

    def _get_train_eval_sets(self):
        """
        Copy files into <project_name>_split folder.

        Returns:
            train_set, eval_set: Sets of pathnames for train/eval.
        """

        # split train/test
        pathnames = [pathname for pathname in glob(os.path.join(self.input_dir, 'imgs', '*'))]
        np.random.shuffle(pathnames)

        n = len(pathnames)
        k = int((1 - self.test_split) * n)

        train_set, test_set = pathnames[:k], pathnames[k:]

        return set(train_set), set(test_set)

    def _copy_image_and_mask_file(self, im_pathname, set_name):
        """
        Copy image and mask from input to train or test output dir.

        im_pathname =       self.input_dir/                      imgs/samson_20211128_row50_col0.png'
        new_im_pathname =   self.output_dir/    <set_name>/      imgs/samson_20211128_row50_col0.png'

        im_pathname =       self.input_dir/                      masks/samson_20211128_row50_col0.png'
        new_im_pathname =   self.output_dir/    <set_name>/      masks/samson_20211128_row50_col0.png'
        """

        parts = im_pathname.split('/')
        new_im_pathname = os.path.join(self.output_dir, set_name, *parts[-2:])
        os.makedirs(os.path.dirname(new_im_pathname), exist_ok=True)
        shutil.copyfile(im_pathname, new_im_pathname)

        mask_pathname = get_mask_pathname_from_image(im_pathname)
        if mask_pathname:
            parts = mask_pathname.split('/')
            new_mask_pathname = os.path.join(self.output_dir, set_name, *parts[-2:])
            os.makedirs(os.path.dirname(new_mask_pathname), exist_ok=True)
            shutil.copyfile(mask_pathname, new_mask_pathname)

    def split_train_test_files(self):
        train_set, eval_split = self._get_train_eval_sets()
        for curr_set, set_name in zip([train_set, eval_split], ('train', 'test')):
            for im_pathname in tqdm(curr_set):
                self._copy_image_and_mask_file(im_pathname, set_name)


if __name__ == '__main__':
    # The following line should be ran only one time, and evaluation files reused for all model evaluations:
    SplitTrainEval('type_2', eval_split=0.2).split_train_test_files()
