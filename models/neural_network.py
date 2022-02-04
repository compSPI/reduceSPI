import operator
from functools import reduce
import functools
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
import torch
import os
import numpy as np
import torch.nn as nn
import latent_space_computation
from reparameterize import SO3reparameterize, N0reparameterize, \
    AlgebraMean, QuaternionMean, S2S1Mean, S2S2Mean, EulerYzyMean, ThetaMean,\
    VectorMean, SO2reparameterize, Nreparameterize

import cnn_initialization

import lie_tools


os.environ["GEOMSTATS_BACKEND"] = "pytorch"

CUDA = torch.cuda.is_available()

DEVICE = torch.device('cuda' if CUDA else 'cpu')


os.environ["GEOMSTATS_BACKEND"] = "pytorch"

CUDA = torch.cuda.is_available()

OUT_PAD = 0


class Encoder(nn.Module):
    """This class compute the Encoder"""

    def __init__(self,  config):
        """
        Initialization of the encoder.

        Parameters
        ----------
        config : dic, principal constants to build a encoder.

        Returns
        -------
        None.

        """
        super(Encoder, self).__init__()
        self.config = config
        self.wigner_dim = config["wigner_dim"]
        self.latent_mode = config["latent_mode"]
        self.latent_space = config["latent_space"]
        self.mean_mode = config["mean_mode"]
        self.fixed_sigma = None
        self.transpose = True
        self.item_rep = config["item_rep"]
        self.rep_copies = config["rep_copies"]
        self.conv_dim = config["dimension"]
        self.n_enc_lay = config["n_enc_lay"]

        self.compression_conv = cnn_initialization.CompressionConv(config)
        self.fcs_infeatures = int(
            (config["img_shape"][-1]**self.conv_dim)/(2**(self.n_enc_lay*(self.conv_dim-1)+1)))

        self.init_layer_latent_space()
        self.reparameterize = nn.ModuleList([self.rep_group])
        self.init_action_net()

    def init_action_net(self):
        if self.latent_space == "so3":
            self.action_net = latent_space_computation.ActionNetSo3(
                self.wigner_dim)
        elif self.latent_space == "so2":
            self.action_net = latent_space_computation.ActionNetSo2()
        else:
            self.action_net = latent_space_computation.ActionNetRL()

    def init_layer_latent_space(self):
        print(self.mean_mode)
        if self.latent_space == 'so3':
            normal = N0reparameterize(self.fcs_infeatures, z_dim=3,
                                      fixed_sigma=self.fixed_sigma)
            if self.mean_mode == 'alg':
                mean_module = AlgebraMean(self.fcs_infeatures)
            elif self.mean_mode == 'q':
                mean_module = QuaternionMean(self.fcs_infeatures)
            elif self.mean_mode == 's2s1':
                mean_module = S2S1Mean(self.fcs_infeatures)
            elif self.mean_mode == 's2s2':
                mean_module = S2S2Mean(self.fcs_infeatures)
            elif self.mean_mode == 'eulyzy':
                mean_module = EulerYzyMean(self.fcs_infeatures)
            self.rep_group = SO3reparameterize(normal, mean_module, k=10)
            self.group_dims = 9
        elif self.latent_space == "so2":
            normal = N0reparameterize(self.fcs_infeatures, z_dim=1,
                                      fixed_sigma=self.fixed_sigma)
            if self.mean_mode == "theta":
                mean_module = ThetaMean(self.fcs_infeatures)
            elif self.mean_mode == "v":
                mean_module = VectorMean(self.fcs_infeatures)
            self.rep_group = SO2reparameterize(normal, mean_module)
        else:
            normal = N0reparameterize(self.fcs_infeatures,
                                      z_dim=self.config["latent_dim"],
                                      fixed_sigma=self.fixed_sigma)
            self.rep_group = Nreparameterize(
                self.fcs_infeatures, z_dim=self.config["latent_dim"], fixed_sigma=self.fixed_sigma)

    def forward(self, h, n=1):
        """
        Compute the passage through the neural network

        Parameters
        ----------
        h : tensor, image or voxel.

        Returns
        -------
        mu : tensor, latent space mu.
        logvar : tensor, latent space sigma.

        """
        h = self.compression_conv(h)
        rot_mat_enc = [r(h, n) for r in self.reparameterize][0]
        self.rot_mat_enc = rot_mat_enc[0]
        if self.latent_space == "so3":
            self.eayzy = lie_tools.group_matrix_to_eazyz(
                rot_mat_enc[0])
            batch_size = self.eayzy.shape[0]
            items = self.action_net(self.eayzy)
        elif self.latent_space == "so2":
            items = self.action_net(self.rot_mat_enc)
            items = nn.functional.normalize(items, eps=1e-30)
            if len(items) != 0:
                assert len(items[items.T[0]**2 + items.T[1]**2 < 0.999]
                           ) == 0, print(items, items.shape)
        else:
            items = self.action_net(self.rot_mat_enc)
        return items, self.rot_mat_enc

    def kl(self):
        kl = [r.kl() for r in self.reparameterize]
        return kl


SO3 = SpecialOrthogonal(3, point_type="vector")


class VaeConv(nn.Module):
    """This class compute the VAE"""

    def __init__(self, config):
        """
        Initialization of the VAE.

        Parameters
        ----------
        config : dic, principal constants to build a encoder.

        Returns
        -------
        None.

        """
        super(VaeConv, self).__init__()
        self.config = config
        self.img_shape = config["img_shape"]
        self.conv_dim = config["dimension"]
        self.with_sigmoid = config["with_sigmoid"]
        self.n_encoder_blocks = config["n_enc_lay"]
        self.n_decoder_blocks = config["n_dec_lay"]

        self.encoder = Encoder(
            self.config)

        self.decoder = cnn_initialization.DecoderConv(
            self.config)

    def forward(self, x):
        """
        Compute the passage through the neural network

        Parameters
        ----------
        x : tensor, image or voxel.

        Returns
        -------
        res : tensor, image or voxel.
        scale_b: tensor, image or voxel.
        mu : tensor, latent space mu.
        logvar : tensor, latent space sigma.

        """
        z, matrix = self.encoder(x)
        res, scale_b = self.decoder(z)
        return res, scale_b, z


def reparametrize(mu, logvar, n_samples=1):
    """
    Transform the probabilistic latent space into a deterministic latent space

    Parameters
    ----------
    mu : tensor, latent space mu.
    logvar : tensor, latent space sigma.
    n_samples : int, number of samples.

    Returns
    -------
    z_flat : tensor, deterministic latent space

    """
    n_batch_data, latent_dim = mu.shape

    std = logvar.mul(0.5).exp_()
    std_expanded = std.expand(
        n_samples, n_batch_data, latent_dim)
    mu_expanded = mu.expand(
        n_samples, n_batch_data, latent_dim)

    if CUDA:
        eps = torch.cuda.FloatTensor(
            n_samples, n_batch_data, latent_dim).normal_()
    else:
        eps = torch.FloatTensor(n_samples, n_batch_data, latent_dim).normal_()
    eps = torch.autograd.Variable(eps)

    z = eps * std_expanded + mu_expanded
    z_flat = z.reshape(n_samples * n_batch_data, latent_dim)
    z_flat = z_flat.squeeze(dim=1)
    return z_flat


def sample_from_q(mu, logvar, n_samples=1):
    """
    Transform a probabilistic latent space into a deterministic latent space

    Parameters
    ----------
    mu : tensor, latent space mu.
    logvar : tensor, latent space sigma.
    n_samples : int, number of samples.

    Returns
    -------
    tensor, deterministic latent space.

    """
    return reparametrize(mu, logvar, n_samples)


def sample_from_prior(latent_dim, n_samples=1):
    """
    Transform a probabilistic latent space into a deterministic latent.

    Parameters
    ----------
    latent_dim : int, latent dimension.
    n_samples : int, optional, number of sample.

    Returns
    -------
    tensor, deterministic latent space.

    """
    if CUDA:
        mu = torch.cuda.FloatTensor(n_samples, latent_dim).fill_(0)
        logvar = torch.cuda.FloatTensor(n_samples, latent_dim).fill_(0)
    else:
        mu = torch.zeros(n_samples, latent_dim)
        logvar = torch.zeros(n_samples, latent_dim)
    return reparametrize(mu, logvar)
