Repo: solar_panel

Goal of this project is to deliver segmentation masks of solar panel farms 
using satellite images. For this purpose I used deep segmentation model in pytorch.
Project has several steps:  
- data preparation  
  - split several large (1GB) satellite images into smaller ones, allowing for some overlap 
        as we expect edges not to be detected ideally. Once segmentation is done 
        I'll combine them again and edge effect should disappear
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

