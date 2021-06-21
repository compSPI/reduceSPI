"""Open datasets and processes them to be used by a neural network"""


import os
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader, random_split
import functools
from PIL import Image

CUDA = torch.cuda.is_available()

if CUDA:
    CRYO_TRAIN_VAL_DIR = os.getcwd() + "/Cryo/VAE_Cryo_V3/Data/"
else:
    CRYO_TRAIN_VAL_DIR = os.getcwd() + "\\Data\\"


KWARGS = {'num_workers': 1, 'pin_memory': True} if CUDA else {}


def open_dataset(path, new_size, is_3d):
    """
    Open datasets and processes data in ordor to make tensors.
    Parameters
    ----------
    path : string path.
    new_size : int.
    is_3d : boolean if 2d or 3d
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
    N = img_shape[0]
    Dataset = []
    if is_3d:
        dataset = torch.Tensor(dataset)
        dataset = normalization_linear(dataset)
    else:
        if len(img_shape) == 3:
            for i in range(N):
                Dataset.append(np.asarray(Image.fromarray(
                    dataset[i]).resize([new_size, new_size])))
        elif len(img_shape) == 4:
            for i in range(N):
                Dataset.append(np.asarray(Image.fromarray(
                    dataset[i][0]).resize([new_size, new_size])))
        dataset = torch.Tensor(Dataset)
        dataset = normalization_linear(dataset)
        if len(img_shape) != 4:
            dataset = dataset.reshape(
                (img_shape[0], 1, img_shape[1], img_shape[1]))
    return dataset


def normalization_linear(dataset):
    """
    Normalize a tensor
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


def organize_dataset(dataset, batch_size, frac_val):
    """
    Separate data in train and validation sets.
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


PATHS = {}
PATHS['simulated2d'] = "cryo_sim_128x128.npy"
PATHS['real_2d30'] = "class2D_30_sort.h5"
PATHS['real_2d39'] = "class2D_39_sort.h5"
PATHS['real_2d93'] = "class2D_93_sort.h5"
PATHS['refine3D'] = "refine3D_180x180_sort.h5"
PATHS['class3D9090'] = "class3D_90x90_sort.h5"
PATHS['simulated3d'] = "concat_simulated.npy"
PATHS['simulated3DNoise'] = "cryo_sim_128x128.npy"
PATHS['real3d'] = 'data.hdf5'
PATHS['4points'] = "4points.npy"
PATHS['4points1'] = "4points1.npy"
PATHS['4points3d'] = "3d_images.npy"


SHAPES = {}
SHAPES['simulated3d'] = (1, 320, 320)
SHAPES['simulated2d'] = (1, 128, 128)
SHAPES['4points'] = (1, 128, 128)
SHAPES['4points1'] = (1, 64, 64)
SHAPES['4points3d'] = (64, 64, 64)

CONSTANTES = {}
CONSTANTES['img_shape'] = (1, 128, 128)
CONSTANTES['is_3d'] = False
CONSTANTES['with_sigmoid'] = True
CONSTANTES['OUT_CHANNELS'] = [32, 64]
CONSTANTES['conv_dim'] = len(CONSTANTES['img_shape'][1:])
CONSTANTES['dataset_name'] = PATHS['simulated2d']
CONSTANTES['ENC_KS'] = 4
CONSTANTES['ENC_STR'] = 2
CONSTANTES['ENC_PAD'] = 1
CONSTANTES['ENC_DIL'] = 1
CONSTANTES['ENC_C'] = 1
CONSTANTES['DEC_KS'] = 3
CONSTANTES['DEC_STR'] = 1
CONSTANTES['DEC_PAD'] = 0
CONSTANTES['DEC_DIL'] = 1
CONSTANTES['DEC_C'] = 1
CONSTANTES['DIS_KS'] = 4
CONSTANTES['DIS_STR'] = 2
CONSTANTES['DIS_PAD'] = 1
CONSTANTES['DIS_DIL'] = 1
CONSTANTES['DIS_C'] = 1
CONSTANTES['regularizations'] = ('kullbackleibler')
CONSTANTES['class_2d'] = 39
CONSTANTES['weights_init'] = 'xavier'
CONSTANTES['nn_type'] = 'conv'
CONSTANTES['beta1'] = 0.9
CONSTANTES['beta2'] = 0.999
CONSTANTES['frac_val'] = 0.2
CONSTANTES['BCE'] = True
CONSTANTES['DATA_DIM'] = functools.reduce(
    (lambda x, y: x * y), CONSTANTES['img_shape'])
CONSTANTES['reconstructions'] = ('bce_on_intensities', 'adversarial')
CONSTANTES['SKIP_Z'] = False


SEARCH_SPACE = {}
SEARCH_SPACE['encLayN'] = 2
SEARCH_SPACE['decLayN'] = 2
SEARCH_SPACE['latent_dim'] = 3
SEARCH_SPACE['batch_size'] = 20
SEARCH_SPACE['adversarial'] = False
SEARCH_SPACE['GANLayN'] = 3
SEARCH_SPACE['lr'] = 0.001
SEARCH_SPACE['regu_factor'] = 0.003
SEARCH_SPACE['lambda_regu'] = 0.2
SEARCH_SPACE['lambda_adv'] = 0.2
SEARCH_SPACE['reconstructions'] = ('bce_on_intensities', 'adversarial')
