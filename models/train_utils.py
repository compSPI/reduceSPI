"""Utils to factorize code for learning and visualization."""

import glob
import logging
import os
import torch
import torch.nn as tnn
from scipy.spatial.transform import Rotation as R

import nn
import dataset as ds

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

CKPT_PERIOD = 1

W_INIT, B_INIT, NONLINEARITY_INIT = (
    {0: [[1.0], [0.0]],
     1: [[1.0, 0.0], [0.0, 1.0]]},
    {0: [0.0, 0.0],
     1: [0.01935, -0.02904]},
    'softplus')


def init_xavier_normal(m):
    """
    Initiate weigth of a Neural Network with xavier weigth.

    Parameters
    ----------
    m : Neural Network.

    Returns
    -------
    None.

    """
    if type(m) is tnn.Linear:
        tnn.init.xavier_normal_(m.weight)
    if type(m) is tnn.Conv2d:
        tnn.init.xavier_normal_(m.weight)


def init_kaiming_normal(m):
    """
    Initiate weigth of a Neural Network with kaiming weigth.

    Parameters
    ----------
    m : Neural Network.

    Returns
    -------
    None.

    """
    if type(m) is tnn.Linear:
        tnn.init.kaiming_normal_(m.weight)
    if type(m) is tnn.Conv2d:
        tnn.init.kaiming_normal_(m.weight)


def init_custom(m):
    """
    Initiate weigth of a Neural Network with own custom weigth.

    Parameters
    ----------
    m : Neural Network.

    Returns
    -------
    None.

    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def init_function(weights_init='xavier'):
    """
    Choose the function to initialize the weight of NN.

    Parameters
    ----------
    weights_init : string, optional, initiate weights.

    Raises
    ------
    NotImplementedError
        DESCRIPTION.

    Returns
    -------
    function depending on initialization goal.

    """
    if weights_init == 'xavier':
        return init_xavier_normal
    if weights_init == 'kaiming':
        return init_kaiming_normal
    if weights_init == 'custom':
        return init_custom
    raise NotImplementedError(
        "This weight initialization is not implemented.")


def init_modules_and_optimizers(train_params, config):
    """
    Initialization of the different modules and optimizer of the NN.

    Parameters
    ----------
    train_params : dic, meta parameters for the NN.
    config : dic, meta parameters for the NN.

    Returns
    -------
    modules : dic, dic of modules encoder and decoder and gan of the NN.
    optimizers : dic, dic of optimizer of the NN.

    """
    modules = {}
    optimizers = {}
    lr = train_params['lr']
    beta1 = train_params['beta1']
    beta2 = train_params['beta2']
    vae = nn.VaeConv(config).to(DEVICE)

    modules['encoder'] = vae.encoder
    modules['decoder'] = vae.decoder

    if 'adversarial' in train_params['reconstructions']:
        discriminator = nn.Discriminator(config).to(DEVICE)
        modules['discriminator_reconstruction'] = discriminator

    if 'adversarial' in train_params['regularizations']:
        discriminator = nn.Discriminator(config).to(DEVICE)
        modules['discriminator_regularization'] = discriminator

    # Optimizers
    optimizers['encoder'] = torch.optim.Adam(
        modules['encoder'].parameters(), lr=lr, betas=(beta1, beta2))
    optimizers['decoder'] = torch.optim.Adam(
        modules['decoder'].parameters(), lr=lr, betas=(beta1, beta2))

    if 'adversarial' in train_params['reconstructions']:
        optimizers['discriminator_reconstruction'] = torch.optim.Adam(
            modules['discriminator_reconstruction'].parameters(),
            lr=train_params['lr'],
            betas=(train_params['beta1'], train_params['beta2']))

    if 'adversarial' in train_params['regularizations']:
        optimizers['discriminator_regularization'] = torch.optim.Adam(
            modules['discriminator_regularization'].parameters(),
            lr=train_params['lr'],
            betas=(train_params['beta1'], train_params['beta2']))

    return modules, optimizers


def init_training(train_dir, train_params, config):
    """
    Initialization; Load ckpts or init.

    Parameters
    ----------
    train_dir : string, dir where to save the modules.
    train_params : dic, meta parameters for the NN.
    config : dic, meta parameters for the NN.

    Returns
    -------
    modules : dic, dic of modules encoder and decoder and gan of the NN.
    optimizers : dic, dic of optimizer of the NN.
    start_epoch : int, the number of epoch the NN has already done.
    train_losses_all_epochs : list,  value of the train_loss for every epoch.
    val_losses_all_epochs : list, value of the val_loss for every epoch.
    """
    start_epoch = 0
    train_losses_all_epochs = []
    val_losses_all_epochs = []

    modules, optimizers = init_modules_and_optimizers(
        train_params, config)

    path_base = os.path.join(train_dir, 'epoch_*_checkpoint.pth')
    ckpts = glob.glob(path_base)
    if len(ckpts) == 0:
        weights_init = train_params['weights_init']
        logging.info(
            "No checkpoints found. Initializing with %s.", weights_init)

        for module_name, module in modules.items():
            module.apply(init_function(weights_init))

    else:
        ckpts_ids_and_paths = [
            (int(f.split('_')[-2]), f) for f in ckpts]
        _, ckpt_path = max(
            ckpts_ids_and_paths, key=lambda item: item[0])
        logging.info("Found checkpoints. Initializing with %s.", ckpt_path)
        if torch.cuda.is_available():
            def map_location(storage): return storage.cuda()
        else:
            map_location = 'cpu'
        ckpt = torch.load(ckpt_path, map_location=map_location)
        # ckpt = torch.load(ckpt_path, map_location=DEVICE)
        for module_name in modules:
            module = modules[module_name]
            optimizer = optimizers[module_name]
            module_ckpt = ckpt[module_name]
            module.load_state_dict(module_ckpt['module_state_dict'])
            optimizer.load_state_dict(
                module_ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            train_losses_all_epochs = ckpt['train_losses']
            val_losses_all_epochs = ckpt['val_losses']

    return (modules, optimizers, start_epoch,
            train_losses_all_epochs, val_losses_all_epochs)


def save_checkpoint(epoch, modules, optimizers, dir_path,
                    train_losses_all_epochs, val_losses_all_epochs,
                    nn_architecture, train_params):
    """
    Save NN's weights at a precise epoch.

    Parameters
    ----------
    epoch : int, current epoch.
    modules : dic, dic of modules encoder and decoder and gan of the NN.
    optimizers : dic, dic of optimizer of the NN.
    dir_path : string, dir where to save modules
    train_losses_all_epochs : list,  value of the train_loss for every epoch.
    val_losses_all_epochs : list, value of the val_loss for every epoch.
    nn_architecture : dic, meta parameters for the NN.
    train_params : dic, meta parameters for the NN.

    Returns
    -------
    None.

    """
    checkpoint = {}
    for module_name in modules.keys():
        module = modules[module_name]
        optimizer = optimizers[module_name]
        checkpoint[module_name] = {
            'module_state_dict': module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}
        checkpoint['epoch'] = epoch
        checkpoint['train_losses'] = train_losses_all_epochs
        checkpoint['val_losses'] = val_losses_all_epochs
        checkpoint['nn_architecture'] = nn_architecture
        checkpoint['train_params'] = train_params
    checkpoint_path = os.path.join(
        dir_path, 'epoch_%d_checkpoint.pth' % epoch)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(output, epoch_id=None):
    """
    Loads a NN and all information about it at one expecting stage
    of the learning

    Parameters
    ----------
    output : string, dir where a NN has been saved.
    epoch_id : int, optional. The default is None.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    ckpt : NN ,just loaded NN network.

    """
    if epoch_id is None:
        ckpts = glob.glob(
            '%s/checkpoint_*/epoch_*_checkpoint.pth' % output)
        if len(ckpts) == 0:
            raise ValueError('No checkpoints found.')
        ckpts_ids_and_paths = [(int(f.split('_')[-2]), f) for f in ckpts]
        _, ckpt_path = max(
            ckpts_ids_and_paths, key=lambda item: item[0])
    else:
        # Load module corresponding to epoch_id
        ckpt_path = f"{output}/checkpoint_{epoch_id:0>6d}/" + \
            "epoch_{epoch_id}_checkpoint.pth"

        print(ckpt_path)
        if not os.path.isfile(ckpt_path):
            raise ValueError(
                'No checkpoints found for epoch %d in output %s.' % (
                    epoch_id, output))

    print('Found checkpoint. Getting: %s.' % ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    return ckpt


def load_module_state(output, module, module_name, epoch_id=None):
    """
    Affects weights of the considered epoch_id to NN's weights.

    Parameters
    ----------
    output : string, dir where to find the NN.
    module : NN, NN with initialized weight.
    module_name : string, name of the considered module
    epoch_id : int, optional. Epoch we are interested in. The default is None.

    Returns
    -------
    module : NN, NN with the weight of the NN after the epoch_id.

    """
    ckpt = load_checkpoint(
        output=output, epoch_id=epoch_id)

    module_ckpt = ckpt[module_name]
    module.load_state_dict(module_ckpt['module_state_dict'])

    return module


def get_under_dic_cons(const, list_arg):
    """
    Take a sub dictionnary of a dictionnary.

    Parameters
    ----------
    const : dic.
    list_arg : list, liste of keys you want to save.

    Returns
    -------
    new_dic : dic, sub dic of const.

    """
    new_dic = {}
    for key in list_arg:
        if key in const:
            new_dic[key] = const[key]
    return new_dic


def quaternion_to_euler(labels):
    """
    Transform the quaternion representation of rotation in zyx euler
    representation.

    Parameters
    ----------
    labels : dataframe, description of the orientation of each image.

    Returns
    -------
    liste : list, liste of triples describibg the rotation with the zyx
    euler angles.

    """
    n = len(labels)
    liste = []
    for i in range(n):
        A = labels['rotation_quaternion'].iloc[i].replace(' ]', ']')
        A = A.replace('  ', ' ')
        A = A.replace('  ', ' ')
        A = A.replace('  ', ' ')
        A = A.replace(' ]', ']')
        A = A.replace('  ', ' ')
        A = A[1:-1].split(' ')
        B = list(map(float, A))
        r = R.from_quat(B)
        liste.append(r.as_euler('zyx', degrees=True))
    return liste


def load_module(output, module_name='encoder', epoch_id=None):
    """
    Affects weights of the considered epoch_id to NN's weights.

    Parameters
    ----------
    output : string, dir where to find the NN.
    module : NN, NN with initialized weight.
    module_name : string, name of the considered module
    epoch_id : int, optional. Epoch we are interested in. The default is None.

    Returns
    -------
    module : NN, NN with the weight of the NN after the epoch_id.

    """
    ckpt = load_checkpoint(
        output=output, epoch_id=epoch_id)
    nn_architecture = ckpt['nn_architecture']
    nn_architecture['conv_dim'] = ds.CONSTANTS['conv_dim']
    nn_architecture.update(get_under_dic_cons(
        ds.CONSTANTS, ds.META_PARAM_NAMES))
    nn_type = nn_architecture['nn_type']
    print('Loading %s from network of architecture: %s...' % (
        module_name, nn_type))
    vae = nn.VaeConv(nn_architecture)
    modules = {}
    modules['encoder'] = vae.encoder
    modules['decoder'] = vae.decoder
    module = modules[module_name].to(DEVICE)
    module_ckpt = ckpt[module_name]
    module.load_state_dict(module_ckpt['module_state_dict'])

    return module


def get_logging_shape(tensor):
    """
    Convert shape of a tensor into a string.

    Parameters
    ----------
    tensor : tensor.

    Returns
    -------
    logging_shape : string, shape of the tensor.

    """
    shape = tensor.shape
    logging_shape = '(' + ('%s, ' * len(shape) % tuple(shape))[:-2] + ')'
    return logging_shape
