Repo: solar_panel

```
pip install -U numpy pandas matplotlib torchviz scikit-learn tensorboard torchvision torch tqdm torch-lr-finder pytest
yes | conda install -c conda-forge jupyter_contrib_nbextensions graphviz python-graphviz
```

Goal of this project is to deliver segmentation masks of solar panel farms 
using satellite images. For this purpose I used deep segmentation model in pytorch.
Project has several steps:  
- data preparation  
  - split several large (1GB) satellite images into smaller ones, allowing for some overlap 
        as we expect edges not to be detected ideally. Once segmentation is done 
        I'll combine them again and edge effect should disappear
    - the structure of the folders should be:
      - train
        - images
        - masks
      - valid
        - images
        - masks
      - test
        - images
        - masks
      - rest
        - images
      - info_sheet.csv

      - experiments
        - experiment_id_1_(some_tag)
          - dataset (optional)
            - train
            - valid
          - train
            - predictions
          - val
            - predictions
          - model and other info
        - experiment_id_2
          - train
            - predictions
          - val
            - predictions
          - model and other info
          ...  

        (when happy with validation predictions)  
        - experiment_id_x
          - train
            - predictions
          - val
            - predictions
          - test
            - predictions
          - rest
            - predictions

      - split images into train, valid, and test set (we already have some train images that we can use as well) 
      - plot mean and standard deviation of the already labeled images and identify 
            necessary augmentations (what are normalization parameters in that case?)
      - process images into Datasets and DataLoaders
- write model in PyTorch
- start with simple optimizer and loss function
- train images, monitor loss and accuracy
- evaluate on a unknown test-set
- if not happy, re-label new images for training 
- once happy with the model: freeze, deploy, evaluate all the images that exist
- merge everything together into large satellite images that will be deliverable


b/c we know that the pixel resolution from the satellite that took the photos is 50cm/pixel. The modules used on the Samson project are LONGi-LR4 and LONGi-LR6 which have about 40 inches (101.6 cm) 
