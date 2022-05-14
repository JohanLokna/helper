# Interpretability Project
## Setup
To use this template you can generate a Conda environment using `environment.yml` by running
```sh
conda env create -f environment.yml  --name <custom_name>
```
## Dataset
This dataset contains a mix of samples from the Kaggle datasets [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) and [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) datasets.

### Pyradiomics
To retrieve train, validation and test set for the [pyradiomics](https://pyradiomics.readthedocs.io/en/latest/) features we provide the function `get_radiomics_dataset()` in `data.py`. The function returns both datasets as numpy arrays.
```sh
from data import get_radiomics_dataset
train_data, train_labels, val_data, val_labels, test_data, test_labels = get_radiomics_dataset()
```
### Images
The image train, validation and test sets are provided as pytorch [ImageFolders](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) through `get_img_dataset(transform=None)` in `data.py`. Note that, in order to incorporate data augmentation, you are able to pass a list of [transforms](https://pytorch.org/vision/0.9/transforms.html) to this function.

```sh
from data import get_img_dataset
from torchvision import transforms
transform = [transforms.RandomRotation(90), transforms.RandomHorizontalFlip()]
train_dataset, val_dataset, test_dataset = get_img_dataset(transform)
```

## Baseline
We provide a simple Baseline CNN based on pytorch in `data.py` through `BaselineClf()`. To use it, simply do
```sh
from model import BaselineClf
model = BaselineClf()
```