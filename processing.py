import PIL
from PIL import Image
from math import cos, degrees
from pathlib import Path
import pandas as pd
import os
from tqdm import tqdm

from constants import ROOT

PIL.Image.MAX_IMAGE_PIXELS = 339738624


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
        project_name (str): Like '6126758101'
        solar_farm_name(str): like 'IMG_PHR1B_PMS-N_001'
        tag (str): like 'R2C2'
        image_filename = 'IMG_PHR1B_PMS-N_202112241708104_ORT_6126758101_' + tag + '.TIF'

        image_pathname (Path): Path to the image.
        coordinates ((float, float)): Latitude and longitude.
        output_folder: folder with cropped images.
        tag: 'R1C1' for example.

    """

    # from index.html file
    COORDINATES = {'R1C1': (-95.41759381264909, 33.50642345814874),
                   'R1C2': (-95.22180129867475, 33.51009141565303),
                   'R2C1': (-95.21864991285545, 33.38646550386663),
                   'R2C2': (-95.41416534766181, 33.38281461688832)}

    def __init__(self, project_name, solar_farm_name, tag, image_filename):
        self.project_name = project_name
        self.solar_farm_name = solar_farm_name
        self.tag = tag
        self.image_filename = image_filename
        self.coordinates = self.COORDINATES[tag]

        self.image_pathname = Path(ROOT) / 'data' / project_name / solar_farm_name / image_filename
        self.cropped_folder = Path(ROOT) / 'data' / project_name / solar_farm_name / (tag + '_cropped')
        self.cropped_info_file = self.cropped_folder.parent / (tag + '_cropped_images_info.csv')

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
            df = self._crop_and_write(img, self.cropped_folder / , start_lat, start_long, prefix=tag)
            df.to_csv(self.cropped_info_file)
        print('Done cropping')

    def stitch_images(self,
                      image_pathname: Path,
                      info_file_pathname: Path,
                      dir_name: Path,
                      output_image_pathname: Path
    ):
        """

        Args:
            image_pathname:
            info_file_pathname:
            dir_name:
            output_image_pathname:

        Returns:

        """

        im = Image.open(image_pathname)
        nimg = Image.new('RGB', im.size)

        df = pd.read_csv(info_file_pathname)
        single_height = df['x_pixels'].iloc[0]
        single_width = df['y_pixels'].iloc[0]

        for row in tqdm(df.iterrows()):
            row = row[1]  # this gets only relevant info
            # open an image and paste it into the corresponding place
            image_name, image_row, image_col = row['img_name'], row['grid_row'], row['grid_col']
            with Image.open(os.path.join(dir_name, image_name + '.png')) as im:
                nimg.paste(im, (single_width * image_col, single_height * image_row))

        nimg.save(output_image_pathname)



if __name__ == '__main__':

    project_name = '6126758101'
    solar_farm_name = 'IMG_PHR1B_PMS-N_001'
    for tag in ['R1C2', 'R2C1', 'R2C2']:
        image_filename = 'IMG_PHR1B_PMS-N_202112241708104_ORT_6126758101_' + tag + '.TIF'
        big_image = BigImage(project_name, solar_farm_name, tag, image_filename)
        big_image.crop_large_image()

# /Users/nenad.bozinovic/PycharmProjects/solar_panel/data/6126758101/raw/IMG_PHR1B_PMS-N_001/IMG_PHR1B_PMS-N_202112241708104_ORT_6126758101_R2C2.TIF')
# /Users/nenad.bozinovic/PycharmProjects/solar_panel/data/6126758101/raw/IMG_PHR1B_PMS-N_202112241708104_ORT_6126758101_R2C2.TIF
