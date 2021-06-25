"""Open datasets and process them to be used by a neural network."""


import json
import os
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader, random_split
import functools
from PIL import Image

CUDA = torch.cuda.is_available()


KWARGS = {'num_workers': 1, 'pin_memory': True} if CUDA else {}


def open_dataset(path, new_size, is_3d):
    """Open datasets and processes data in ordor to make tensors.

    Parameters
    ----------
    path : string path.
    new_size : int.
    is_3d : boolean if 2d or 3d.

    Returns
    -------
    tensor, images in black and white.
    """
    if not os.path.exists(path):
        raise OSError
    if path.lower().endswith(".h5"):
        data_dict = h5py.File(path, 'r')
        all_datasets = data_dict['particles'][:]
    else:
        all_datasets = np.load(path)
    dataset = np.asarray(all_datasets)
    img_shape = dataset.shape
    n_imgs = img_shape[0]
    new_dataset = []
    if is_3d:
        dataset = torch.Tensor(dataset)
        dataset = normalization_linear(dataset)
        dataset = dataset.reshape((len(dataset), 1)+dataset.shape[1:])
    else:
        if img_shape.ndim == 3:
            for i in range(n_imgs):
                new_dataset.append(np.asarray(Image.fromarray(
                    dataset[i]).resize([new_size, new_size])))
        elif img_shape.ndim == 4:
            for i in range(n_imgs):
                new_dataset.append(np.asarray(Image.fromarray(
                    dataset[i][0]).resize([new_size, new_size])))
        dataset = torch.Tensor(new_dataset)
        dataset = normalization_linear(dataset)
        if len(img_shape) != 4:
            dataset = dataset.reshape(
                (img_shape[0], 1, img_shape[1], img_shape[1]))
    return dataset


def normalization_linear(dataset):
    """Normalize a tensor.

    Parameters
    ----------
    dataset : tensor, images.

    Returns
    -------
    dataset : tensor, normalized images.
    """
    for i, data in enumerate(dataset):
        min_data = torch.min(data)
        max_data = torch.max(data)
        if max_data == min_data:
            raise ZeroDivisionError
        dataset[i] = (data - min_data) / (max_data - min_data)
    return dataset


def split_dataset(dataset, batch_size, frac_val):
    """Separate data in train and validation sets.

    Parameters
    ----------
    dataset : tensor, images.
    batch_size : int, batch_size.
    frac_val : float, ratio between validation and training datasets.

    Returns
    -------
    trainset : tensor of training images.
    testset : tensor of test images.
    trainloader : tensor ready to be used by the NN for training images.
    testloader : tensor ready to be used by the NN for test images.
    """
    n_imgs = len(dataset)
    n_val = int(n_imgs*frac_val)
    trainset, testset = random_split(dataset, [n_imgs-n_val, n_val])
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, **KWARGS)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, **KWARGS)
    return trainset, testset, trainloader, testloader


def hinted_tuple_hook(obj):
    """Transform a list into tuple.

    Parameters
    ----------
    obj : *, value of a dic.

    Returns
    -------
    tuple, transform the value of a dic into dic.
    obj: *,value of a dic.
    """
    if '__tuple__' in obj:
        return tuple(obj['items'])
    return obj


def load_parameters(path):
    """Load metadata for the VAE.

    Parameters
    ----------
    path : string, path to the file.

    Returns
    -------
    paths : dic, path to the data.
    shapes: dic, shape of every dataset.
    constants: dic, meta information for the vae.
    search_space: dic, meta information for the vae.
    meta_param_names: dic, names of meta parameters.
    """
    with open(path) as json_file:
        parameters = json.load(json_file, object_hook=hinted_tuple_hook)
        paths = parameters["paths"]
        shapes = parameters["shape"]
        constants = parameters["constants"]
        search_space = parameters["search_space"]
        meta_param_names = parameters["meta_param_names"]
        constants["conv_dim"] = len(constants["img_shape"][1:])
        constants["dataset_name"] = paths["simulated_2d"]
        constants["dim_data"] = functools.reduce(
            (lambda x, y: x * y), constants["img_shape"])
        return paths, shapes, constants, search_space, meta_param_names


if __name__ == "__main__":
    PATHS, SHAPES, CONSTANTS, SEARCH_SPACE, META_PARAM_NAMES = load_parameters(
        "vae_parameters.json")
