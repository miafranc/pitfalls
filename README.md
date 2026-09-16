# Pitfalls in Evaluating Interpretable Models: Interpretability Metrics, Randomness, and Reproducibility

## `settings.py`

The file `settings.py` contains all the settable parameters for the models used in the experiments,
for example `IMG_SIZE`, `BATCH_SIZE`, `NUM_EPOCHS`, etc.
These can be set also from the code by importing the `settings` module.

## Datasets

The datasets used in the experiments are the following:
* CUB: [https://www.vision.caltech.edu/datasets/cub_200_2011/](https://www.vision.caltech.edu/datasets/cub_200_2011/)
* Stanford Dogs: [http://vision.stanford.edu/aditya86/ImageNetDogs/](http://vision.stanford.edu/aditya86/ImageNetDogs/)

The small Stanford Dogs dataset (using only 5 classes of the original dataset) used throughout the experiments can be found in the `data/stanford_dogs_small` directory.

## BagNet

The `bagnet` folder contains the code of [BagNets](https://github.com/wielandbrendel/bag-of-local-features-models),
which is used by `train_test_bagnet.py` to train and test a BagNet model.

### Train and test a BagNet model

Training a model can be done calling the `train` method of `train_test_bagnet.py`.
To test/evaluate a trained model one has to call the `test` method giving the path to the model file.

## ProtoPNet

The `protopnet` folder contains the code of [ProtoPNet](https://github.com/cfchen-duke/ProtoPNet), having some imports slighty modified in order to work properly.

### Preprocessing

Preprocessing of the CUB dataset can be performed by the `preprocess_cub.py`, setting the required paths correctly in the code.
The script (based on [https://github.com/M-Nauta/PIPNet/blob/main/util/preprocess_cub.py](https://github.com/M-Nauta/PIPNet/blob/main/util/preprocess_cub.py)) splits the dataset into train and test sets and also performs cropping the images according to the bounding box metadata of CUB.

### Static augmentation

Augmentation (based on `img_aug.py` of the original ProtoPNet code) performs static and reproducible augmentation using the [Augmentor](https://github.com/mdbloice/Augmentor) library.
Similarly to preprocessing, some paths are required to be set in the code before running `img_aug_new.py`.

### Test the provided trained models

Trained `PPNet` models can be tested by calling the `train_test_protopnet()` method of `purity_test_ppnet.py`.
Before running the script, the `TEST_DIR` variable from `settings.py` has to be set to the path containing the cropped CUB images.

### Prototype purity calculation

Purity calculations can be performed calling the `calculate_purity()` method.
The path pointing to the CUB dataset has to contain the following 3 files (included in the official dataset as annotations):
* `images.txt`
* `parts.txt`
* `part_locs.txt`

In this case, `TEST_DIR` has to point to the full test directory instead of the cropped images.


## Conda environments

The conda environments used when evaluating the trained ProtoPNet models using different GPUs and Pytorch/CUDA environments, 
can be found in the `envs` directory.
