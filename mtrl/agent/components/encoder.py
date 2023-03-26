# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Implementation based on Denis Yarats' implementation of [SAC](https://github.com/denisyarats/pytorch_sac).
"""Encoder component for the agent."""
import os
from copy import deepcopy
from typing import List, cast

from itertools import cycle
import torch
import torch.nn as nn
import bnpy
from bnpy.data.XData import XData
from matplotlib import pylab
import numpy as np

from mtrl.agent import utils as agent_utils
from mtrl.agent.components import base as base_component
from mtrl.agent.components import moe_layer
from mtrl.agent.ds.mt_obs import MTObs
from mtrl.utils import config as config_utils
from mtrl.utils.types import ConfigType, ModelType, TensorType


def tie_weights(src, trg):
    assert type(src) == type(trg)
    if hasattr(src, "weight"):
        trg.weight = src.weight
    if hasattr(src, "bias"):
        trg.bias = src.bias

class Encoder(base_component.Component):
    def __init__(
        self,
        env_obs_shape: List[int],
        multitask_cfg: ConfigType,
        *args,
        **kwargs,
    ):
        """Interface for the encoder component of the agent.

        Args:
            env_obs_shape (List[int]): shape of the observation that the actor gets.
            multitask_cfg (ConfigType): config for encoding the multitask knowledge.
        """
        super().__init__()
        self.multitask_cfg = multitask_cfg

    def forward(self, mtobs: MTObs, detach: bool = False) -> TensorType:
        """Encode the input observation.

        Args:
            mtobs (MTObs): multi-task observation.
            detach (bool, optional): should detach the observation encoding
                from the computation graph. Defaults to False.

        Raises:
            NotImplementedError:

        Returns:
            TensorType: encoding of the observation.

        """
        raise NotImplementedError

    def copy_conv_weights_from(self, source: "Encoder") -> None:
        """Copy convolutional weights from the `source` encoder.

        The no-op implementation should be overridden only by encoders
        that take convnets.

        Args:
            source (Encoder): encoder to copy weights from.

        """
        pass

############################################
######## Implements of VAE model ###########
############################################

class VAE_Encoder(Encoder):
    def __init__(
            self,
            env_obs_shape: List[int],
            multitask_cfg: ConfigType,
            # feature_dim: int, # 50
            num_layers: int, # 2
            hidden_dim: int, # 50
            latent_dim: int, # 10
            type:str,
            should_reconstruct:bool,
            # should_tie_encoders: bool, # true
            # log_std_bounds : List
            ):
        super().__init__(env_obs_shape=env_obs_shape, multitask_cfg=multitask_cfg)
        assert len(env_obs_shape) == 1
        self.type_name = type
        self.num_layers = num_layers
        self.multitask_cfg = multitask_cfg
        self.should_reconstruct = should_reconstruct

        # conditioned context encoding on vae encoder
        if self.multitask_cfg.conditioned_on_context_encoding:
            self.input_dim = env_obs_shape[0] + self.multitask_cfg.task_encoder_cfg.model_cfg.output_dim
        else:
            self.input_dim = env_obs_shape[0]
            # 2023/03/24 only receive onehot as input
            # self.input_dim = 10
        
        # 2023/03/22 only receive metadata encoding
        # self.input_dim = self.multitask_cfg.task_encoder_cfg.model_cfg.output_dim

        # self.trunk = agent_utils.build_mlp(
        #     input_dim=self.input_dim,
        #     hidden_dim=hidden_dim,
        #     num_layers=num_layers,
        #     output_dim=hidden_dim, # not used
        #     last_activation=True # shared layers last output using ReLU
        # )
        # # output head mu, mean of gaussian
        # self.mu_latent = agent_utils.build_mlp(
        #     input_dim=hidden_dim,
        #     hidden_dim=hidden_dim, # not used
        #     output_dim=latent_dim,
        #     num_layers=0
        # )
        # # output head log_variance
        # self.logvar_latent = agent_utils.build_mlp(
        #     input_dim=hidden_dim,
        #     hidden_dim=hidden_dim, # not used
        #     output_dim=latent_dim,
        #     num_layers=0
        # )
        
        # encoder
        self.trunk = self.build_mlp(input_dim=self.input_dim, 
                                    hidden_dim=hidden_dim, 
                                    output_dim=hidden_dim, 
                                    num_layers=num_layers, 
                                    output_activation=True)
        self.mu_latent = nn.Linear(hidden_dim, latent_dim)
        self.log_var_latent = nn.Linear(hidden_dim, latent_dim)
        
        # decoder
        if self.should_reconstruct:
            self.decoder = self.build_mlp(input_dim=latent_dim,
                                          hidden_dim=100,
                                          output_dim=self.input_dim,
                                          num_layers=1,
                                          output_activation=False)


    def build_mlp(self, input_dim, hidden_dim, output_dim, num_layers, output_activation=False):
        '''
        num_layers: number of hidden layers
        output_activation: whether should add ReLU at end of output
        '''
        mods: List[nn.Module]
        # input layer
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        # hidden layer
        for _ in range(num_layers):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        # output_layer
        if output_activation:
            mods += [nn.Linear(hidden_dim, output_dim), nn.ReLU()]
        else:
            mods += [nn.Linear(hidden_dim, output_dim)]
        return nn.Sequential(*mods)

    def sample(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) log variance of the latent Gaussian [B x D]
        :return: z (Tensor) [B x D]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, mtobs: MTObs, detach: bool = False):
        
        if self.multitask_cfg.conditioned_on_context_encoding:
            env_obs = torch.cat((mtobs.env_obs, mtobs.task_info.encoding), dim=1)
        else:
            env_obs = mtobs.env_obs
            # 2023/03/24 only receive onehot as VAE input
            # env_obs = mtobs.env_obs[:,12:]

        # 2023/03/22 only receive metadata encoding as input
        # env_obs = obs.task_info.encoding
        
        mu = self.mu_latent(self.trunk(env_obs))
        log_var = self.log_var_latent(self.trunk(env_obs))
        
        z = self.sample(mu, log_var)

        if self.should_reconstruct:
            reconst = self.decoder(z)
        else:
            reconst = None
        
        if detach:
            z.detach()
            # mu.detach()
            # log_var.detach()

        return z, mu, log_var, reconst


###################################################
######### Implements from origin library ##########
###################################################


class PixelEncoder(Encoder):
    def __init__(
        self,
        env_obs_shape: List[int],
        multitask_cfg: ConfigType,
        feature_dim: int,
        num_layers: int = 2,
        num_filters: int = 32,
    ):
        """Convolutional encoder for pixels observations.

        Args:
            env_obs_shape (List[int]): shape of the observation that the actor gets.
            multitask_cfg (ConfigType): config for encoding the multitask knowledge.
            feature_dim (int): feature dimension.
            num_layers (int, optional): number of layers. Defaults to 2.
            num_filters (int, optional): number of conv filters per layer. Defaults to 32.
        """
        super().__init__(env_obs_shape=env_obs_shape, multitask_cfg=multitask_cfg)

        assert len(env_obs_shape) == 3
        self.convs = nn.ModuleList(
            [nn.Conv2d(env_obs_shape[0], num_filters, 3, stride=2)]
        )
        for _ in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        # self.feature_dim = feature_dim
        self.num_layers = num_layers
        layer_to_dim_mapping = {2: 39, 4: 35, 6: 31}
        out_dim = layer_to_dim_mapping[self.num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)

    def reparameterize(self, mu: TensorType, logstd: TensorType) -> TensorType:
        """Reparameterization Trick

        Args:
            mu (TensorType): mean of the gaussian.
            logstd (TensorType): log of standard deviation of the gaussian.

        Returns:
            TensorType: sample from the gaussian.
        """
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, env_obs: TensorType) -> TensorType:
        """Encode the environment observation using the convolutional layers.

        Args:
            env_obs (TensorType): observation from the environment.

        Returns:
            TensorType: encoding of the observation.
        """
        env_obs = env_obs / 255.0

        conv = torch.relu(self.convs[0](env_obs))

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, mtobs: MTObs, detach: bool = False):
        env_obs = mtobs.env_obs
        h = self.forward_conv(env_obs=env_obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        h_norm = self.ln(h_fc)
        out = torch.tanh(h_norm)

        return out

    def copy_conv_weights_from(self, source: Encoder):
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])  # type: ignore[index]


class IdentityEncoder(Encoder):
    def __init__(
        self,
        env_obs_shape: List[int],
        multitask_cfg: ConfigType,
        feature_dim: int,
        # num_layers: int = 2,
        # num_filters: int = 32,
    ):
        """Identity encoder that does not perform any operations.

        Args:
            env_obs_shape (List[int]): shape of the observation that the actor gets.
            multitask_cfg (ConfigType): config for encoding the multitask knowledge.
            # feature_dim (int): feature dimension.
            # num_layers (int, optional): number of layers. Defaults to 2.
            # num_filters (int, optional): number of conv filters per layer. Defaults to 32.
        """
        super().__init__(env_obs_shape=env_obs_shape, multitask_cfg=multitask_cfg)

        assert len(env_obs_shape) == 1
        # assert num_layers == 0
        # assert num_filters == 0
        # self.feature_dim = obs_shape[0]
        self.feature_dim = feature_dim

    def forward(self, mtobs: MTObs, detach: bool = False):
        return mtobs.env_obs


class FeedForwardEncoder(Encoder):
    def __init__(
        self,
        env_obs_shape: List[int],
        multitask_cfg: ConfigType,
        feature_dim: int,
        num_layers: int,
        hidden_dim: int,
        should_tie_encoders: bool,
    ):
        """Feedforward encoder for state observations.

        Args:
            env_obs_shape (List[int]): shape of the observation that the actor gets.
            multitask_cfg (ConfigType): config for encoding the multitask knowledge.
            feature_dim (int): feature dimension.
            num_layers (int, optional): number of layers. Defaults to 2.
            hidden_dim (int, optional): number of conv filters per layer. Defaults to 32.
            should_tie_encoders (bool): should the feed-forward layers be tied.
        """

        super().__init__(env_obs_shape=env_obs_shape, multitask_cfg=multitask_cfg)

        assert len(env_obs_shape) == 1

        self.num_layers = num_layers
        self.trunk = agent_utils.build_mlp(
            input_dim=env_obs_shape[0],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=feature_dim,
        )
        self.should_tie_encoders = should_tie_encoders

    def forward(self, mtobs: MTObs, detach: bool = False):
        env_obs = mtobs.env_obs
        h = self.trunk(env_obs)

        if detach:
            h = h.detach()

        return h

    def copy_conv_weights_from(self, source: Encoder):
        if self.should_tie_encoders:
            for src, trg in zip(source.trunk, self.trunk):  # type: ignore[call-overload]
                tie_weights(src=src, trg=trg)


class FiLM(FeedForwardEncoder):
    def __init__(
        self,
        env_obs_shape: List[int],
        multitask_cfg: ConfigType,
        feature_dim: int,
        num_layers: int,
        hidden_dim: int,
        should_tie_encoders: bool,
    ):
        super().__init__(
            env_obs_shape=env_obs_shape,
            multitask_cfg=multitask_cfg,
            feature_dim=feature_dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            should_tie_encoders=should_tie_encoders,
        )

        # overriding the type from base class.
        self.trunk: List[ModelType] = agent_utils.build_mlp_as_module_list(  # type: ignore[assignment]
            input_dim=env_obs_shape[0],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=feature_dim,
        )

    def forward(self, mtobs: MTObs, detach: bool = False):
        env_obs = mtobs.env_obs
        task_encoding: TensorType = cast(TensorType, mtobs.task_info.encoding)  # type: ignore[union-attr]
        # mypy raises a false alarm. mtobs.task if already checked to be not None.
        gammas_and_betas: List[TensorType] = torch.split(
            task_encoding.unsqueeze(2), split_size_or_sections=2, dim=1
        )
        # assert len(gammas_and_betas) == len(self.trunk)
        h = env_obs
        for layer, gamma_beta in zip(self.trunk, gammas_and_betas):
            h = layer(h) * gamma_beta[:, 0] + gamma_beta[:, 1]
        if detach:
            h = h.detach()

        return h


#################################################
########## CARE Mixture of Encoder ##############
#################################################
class MixtureofExpertsEncoder(Encoder):
    def __init__(
        self,
        env_obs_shape: List[int],
        multitask_cfg: ConfigType,
        encoder_cfg: ConfigType,
        task_id_to_encoder_id_cfg: ConfigType,
        num_experts: int,
        # device: torch.device,
    ):
        """Mixture of Experts based encoder.

        Args:
            env_obs_shape (List[int]): shape of the observation that the actor gets.
            multitask_cfg (ConfigType): config for encoding the multitask knowledge.
            encoder_cfg (ConfigType): config for the experts in the mixture.
            task_id_to_encoder_id_cfg (ConfigType): mapping between the tasks and the encoders.
            num_experts (int): number of experts.

        """
        super().__init__(env_obs_shape=env_obs_shape, multitask_cfg=multitask_cfg)
        num_tasks = task_id_to_encoder_id_cfg.num_envs

        _mode = task_id_to_encoder_id_cfg.mode
        if _mode == "identity":
            _cls = moe_layer.OneToOneExperts  # type: ignore [assignment]
        elif _mode == "ensemble":
            _cls = moe_layer.EnsembleOfExperts  # type: ignore [assignment]
        elif _mode == "cluster":
            _cls = moe_layer.ClusterOfExperts  # type: ignore [assignment]
        elif _mode == "gate":
            _cls = moe_layer.AttentionBasedExperts  # type: ignore [assignment]
        elif _mode == "attention": # CARE
            _cls = moe_layer.AttentionBasedExperts  # type: ignore [assignment]
        else:
            raise ValueError(
                f"task_id_to_encoder_id_cfg.mode={_mode} is not supported."
            )
        
        #############################################
        ######### CARE AttentionBasedExperts ########
        ######### MLP after attention calc   ########
        #############################################
        self.selection_network = _cls(
            num_tasks=num_tasks,
            num_experts=num_experts,
            multitask_cfg=multitask_cfg,
            **task_id_to_encoder_id_cfg[_mode],
        )
        '''
        # CARE
        (selection_network): AttentionBasedExperts(
            (trunk): Sequential(
            (0): Linear(in_features=50, out_features=50, bias=True)
            (1): ReLU()
            (2): Linear(in_features=50, out_features=50, bias=True)
            (3): ReLU()
            (4): Linear(in_features=50, out_features=6, bias=True)
            )
            (_softmax): Softmax(dim=1)
        )
        '''
        #############################################
        ########## Mixture of Task Encoder ##########
        #############################################
        self.moe = moe_layer.FeedForward(
            num_experts=num_experts,
            in_features=env_obs_shape[0],
            # encoder_cfg.feature_dim,
            out_features=encoder_cfg.feature_dim,
            num_layers=encoder_cfg.num_layers,
            hidden_features=encoder_cfg.hidden_dim,
            bias=True,
        )
        '''
        (moe): Sequential(
            (0): Linear(num_experts=6, in_features=12, out_features=50, bias=True)
            (1): ReLU()
            (2): Linear(num_experts=6, in_features=50, out_features=50, bias=True)
        )

        '''
        self.should_tie_encoders = encoder_cfg.should_tie_encoders # CARE True

    def forward(self, mtobs: MTObs, detach: bool = False):
        env_obs = mtobs.env_obs
        task_info = mtobs.task_info
        encoder_mask = self.selection_network(task_info=task_info)
        encoding = self.moe(env_obs)
        if detach:
            encoding = encoding.detach()
        sum_of_masked_encoding = (encoding * encoder_mask).sum(dim=0)
        sum_of_encoder_count = encoder_mask.sum(dim=0)
        encoding = sum_of_masked_encoding / sum_of_encoder_count
        return encoding

    def copy_conv_weights_from(self, source):
        if self.should_tie_encoders:
            # for src, trg in zip(source.base, self.base):
            #     tie_weights(src=src, trg=trg)
            for src, trg in zip(source.moe._model, self.moe._model):
                tie_weights(src=src, trg=trg)


_AVAILABLE_ENCODERS = {
    "pixel": PixelEncoder,
    "identity": IdentityEncoder,
    "film": FiLM,
    "feedforward": FeedForwardEncoder,
    "moe": MixtureofExpertsEncoder,
    # modify
    "vae": VAE_Encoder,
    # "dpmm_vae" : DPMM_VAE_Encoder,
}


def make_encoder(
    env_obs_shape: List[int],
    encoder_cfg: ConfigType,
    multitask_cfg: ConfigType,
    # device: Optional[torch.device] = None,
):
    key = "type_to_select"
    if key in encoder_cfg:
        encoder_type_to_select = encoder_cfg[key]
        encoder_cfg = encoder_cfg[encoder_type_to_select]
    assert encoder_cfg.type in _AVAILABLE_ENCODERS
    cfg_to_use = config_utils.make_config_mutable(
        config_utils.unset_struct(deepcopy(encoder_cfg))
    )
    cfg_to_use.pop("type")
    return _AVAILABLE_ENCODERS[encoder_cfg.type](
        env_obs_shape=env_obs_shape,
        multitask_cfg=multitask_cfg,
        **cfg_to_use, # encoder config
    )
