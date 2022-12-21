def evaluate_large_image(image_pathname,
                         proj_name,
                         tag,
                         coordinates,
                         model_pathname,
                         evaluations_folder,
                         skip_cropping=False):
    """
    Evaluates a large image with the following steps:
        1) crop the big image into smaller ones, save under output_dir / cropped_images, save csv file with all the info.
        Creates:
            -cropped_images  : folder with cropped images
            -cropped_images_info.csv
        2) evaluate all the images, find mean and stdev using only those that have 20% coverage with solar modules detected.
        3) using mean and stdev found, evaluate all images again. Creates:
            -predictions     : folder with predictions
            -overlays        : folder with overlays
        4) perform stitching
            -prediction.png   : png of the stitched predictions
            -overlay.png      : png of the stitched overlay

    Args:
        image_pathname (Path): Path to the image.
        proj_name (str): Short name for a project.
        tag (str): A tag for images like 'R1C1'
        coordinates ((float, float)): Latitude and longitude.
        model_pathname (str): Specifies folder that contains model to use.
        evaluations_folder (Path): Folder where to save all the results.
        skip_cropping (bool): True if to assume cropping already happened.
    """

    output_dir = evaluations_folder / proj_name / tag

    if not skip_cropping:
        # 1) crop the big image into smaller ones, save under output_dir / cropped_images, save csv file with all the info
        with Image.open(image_pathname) as img:
            start_long, start_lat = coordinates
            df = crop_and_write(img, output_dir / 'cropped_images', start_lat, start_long, proj_name='samson_' + proj_name)
            df.to_csv(output_dir / 'cropped_images_info.csv')
        print('Done cropping')
    #
    images_pathnames = list((output_dir / 'cropped_images').glob('*.png'))

    # 2) get first pass mean and stdev of all images
    mean, stdev = SegmentationProcessor.get_normalization_params(images_pathnames)
    # evaluate all images, don't save just yet anything since we don't know exact mean and stdev
    test_predictions, _ = evaluation(
        test_images_pathnames=images_pathnames,
        model_pathname=model_pathname,
        normalization_params=(mean, stdev)
    )
    print('Done first pass.')
    #
    # 3) using top predictions, get new mean and stdev values and evaluate all based on those
    mean_new, stdev_new = get_normalization_params(test_predictions, output_dir)
    if mean_new:
       mean = mean_new
       stdev = stdev_new

    test_predictions, _ = evaluation(
        test_images_pathnames=images_pathnames,
        model_pathname=model_pathname,
        normalization_params=(mean, stdev),
        output_dir=output_dir
    )
    print('Done evaluation')

    # 4) perform stitching
    stitch_images(
        image_pathname=image_pathname,
        info_file_pathname=str(output_dir / 'cropped_images_info.csv'),
        dir_name=str(output_dir / 'predictions'),
        output_image_pathname=str(output_dir / f'predictions_stitched_{tag}.png')
    )
    print('Done stitching predictions.')

    stitch_images(
        image_pathname=image_pathname,
        info_file_pathname=str(output_dir / 'cropped_images_info.csv'),
        dir_name=str(output_dir / 'overlays'),
        output_image_pathname=str(output_dir / f'overlays_stitched_{tag}.png')
    )
    print('Done stitching overlays.')
