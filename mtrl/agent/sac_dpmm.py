# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Implementation based on Denis Yarats' implementation of [SAC](https://github.com/denisyarats/pytorch_sac).
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import warnings
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mtrl.agent import utils as agent_utils
from mtrl.agent.abstract import Agent as AbstractAgent
from mtrl.agent.ds.mt_obs import MTObs
from mtrl.agent.ds.task_info import TaskInfo
from mtrl.agent.components import encoder, moe_layer
from mtrl.agent.components.encoder import Alpha_net
from mtrl.env.types import ObsType
from mtrl.logger import Logger
from mtrl.replay_buffer import ReplayBuffer, ReplayBufferSample
from mtrl.utils.types import ConfigType, ModelType, ParameterType, TensorType

import bnpy
from mtrl.agent.components.bnp_model import BNPModel

# TODO: add VAE & DP Mixture

class Agent(AbstractAgent):
    """VAE + DPMM + SAC algorithm.
    Interactive with Experiment(environment) through act() & update() API
    sample_action(): get observation of env, sample action (pi) using current policy (train mode)
    select_action(): get observation of env, select action (mu) using current policy (eval mode)
    update(): get batch from replaybuffer, update critic, actor, alpha, encoder parameters etc.
    """

    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: List[int],
        action_range: Tuple[int, int],
        device: torch.device,

        encoder_cfg:ConfigType,
        # decoder_cfg:ConfigType,

        actor_cfg: ConfigType, # conti.._actor.yaml
        critic_cfg: ConfigType, # conti.._critic.yaml

        encoder_optimizer_cfg: ConfigType,

        alpha_optimizer_cfg: ConfigType,
        actor_optimizer_cfg: ConfigType,
        critic_optimizer_cfg: ConfigType,
        multitask_cfg: ConfigType, # conti.._multitask.yaml
        discount: float,
        init_temperature: float,
        actor_update_freq: int,
        critic_tau: float,
        critic_target_update_freq: int,
        encoder_tau: float, # 0.05
        loss_reduction: str = "mean",
        cfg_to_load_model: Optional[ConfigType] = None,
        should_complete_init: bool = True,
    ):
        key = "type_to_select"
        if key in encoder_cfg:
            encoder_type_to_select = encoder_cfg[key]
            self.encoder_cfg = encoder_cfg[encoder_type_to_select]
        super().__init__(
            env_obs_shape=env_obs_shape,
            action_shape=action_shape,
            action_range=action_range,
            multitask_cfg=multitask_cfg,
            device=device,
        )

        
        # use context metadata
        self.should_use_task_encoder = multitask_cfg.should_use_task_encoder
        self.should_use_dpmm = multitask_cfg.should_use_dpmm
        self.kl_div_update_freq = multitask_cfg.dpmm_cfg.kl_div_update_freq
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_freq = actor_update_freq # 1
        self.critic_target_update_freq = critic_target_update_freq # 1
        
        ####################################################
        if self.should_use_dpmm:
            self.beta_kl_z = multitask_cfg.dpmm_cfg.beta_kl_z
            self.dpmm_update_freq = multitask_cfg.dpmm_cfg.dpmm_update_freq
            self.dpmm_update_start_step = multitask_cfg.dpmm_cfg.dpmm_update_start_step
        else:
            self.beta_kl_z = multitask_cfg.dpmm_cfg.beta_kl_z
            self.dpmm_update_freq = None
            self.dpmm_update_start_step = None
        
        # VAE Encoder
        if self.encoder_cfg.type == 'vae':
            self.encoder = hydra.utils.instantiate(
                self.encoder_cfg, env_obs_shape=env_obs_shape
            ).to(self.device)
        else:
            # CARE
            self.encoder = self._make_encoder(
            env_obs_shape=env_obs_shape,
            encoder_cfg=self.encoder_cfg,
            multitask_cfg=multitask_cfg,
            ).to(self.device)
        
        self.encoder.apply(agent_utils.weight_init)
        ####################################################
        
        # SAC Policy
        self.actor = hydra.utils.instantiate(
            actor_cfg, env_obs_shape=env_obs_shape, action_shape=action_shape
        ).to(self.device)
        # SAC Critic
        self.critic = hydra.utils.instantiate(
            critic_cfg, env_obs_shape=env_obs_shape, action_shape=action_shape
        ).to(self.device)
        # SAC Target Critic
        self.critic_target = hydra.utils.instantiate(
            critic_cfg, env_obs_shape=env_obs_shape, action_shape=action_shape
        ).to(self.device)
        # TODO: change to shared alpha net or single alpha
        if self.multitask_cfg.should_use_disentangled_alpha:
            self.log_alpha = torch.nn.Parameter(
                torch.tensor(
                    [
                        np.log(init_temperature, dtype=np.float32)
                        for _ in range(self.num_envs)
                    ]
                ).to(self.device)
            )
        else:
            #################################################################################
            self.log_alpha = Alpha_net(input_dim=self.encoder_cfg.latent_dim).to(self.device)
            self.log_alpha.apply(agent_utils.weight_init)
            #################################################################################

        # self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        self._components = {
            "encoder": self.encoder,
            "actor": self.actor,
            "critic": self.critic,
            "critic_target": self.critic_target,
            "log_alpha": self.log_alpha,  # type: ignore[dict-item]
        }
        ##########################################
        ########### build DPMM ###################
        ##########################################
        # Bayesian Non-parametric Model
        if self.should_use_dpmm and 'dpmm_cfg' in multitask_cfg:
            self.dpmm_cfg=multitask_cfg.dpmm_cfg
            self.bnp_model = self._make_bnpModel(self.dpmm_cfg)
        else:
            self.dpmm_cfg = None
            self.bnp_model = None
        ##########################################

        # optimizers
        if self.encoder_cfg.type == 'vae':
            self.encoder_optimizer = hydra.utils.instantiate(
                encoder_optimizer_cfg, params=self.get_parameters(name="encoder")
            )
        self.actor_optimizer = hydra.utils.instantiate(
            actor_optimizer_cfg, params=self.get_parameters(name="actor")
        )
        self.critic_optimizer = hydra.utils.instantiate(
            critic_optimizer_cfg, params=self.get_parameters(name="critic")
        )
        self.log_alpha_optimizer = hydra.utils.instantiate(
            alpha_optimizer_cfg, params=self.get_parameters(name="log_alpha")
        )

        if loss_reduction not in ["mean", "none"]:
            raise ValueError(
                f"{loss_reduction} is not a supported value for `loss_reduction`."
            )
        self.loss_reduction = loss_reduction # mean
        
        ##########################################################################
        if self.encoder_cfg.should_reconstruct:
            self.reconstruction_loss = nn.MSELoss(reduction='sum').to(self.device)
        ##########################################################################
        
        if self.encoder_cfg.type == 'vae':
            self._optimizers = {
                "encoder": self.encoder_optimizer,
                "actor": self.actor_optimizer,
                "critic": self.critic_optimizer,
                "log_alpha": self.log_alpha_optimizer,
            }
        else:
            self._optimizers = {
                "actor": self.actor_optimizer,
                "critic": self.critic_optimizer,
                "log_alpha": self.log_alpha_optimizer,
            }

        ################################################
        ########### Metadata Context Encoder ###########
        ################################################
        if self.should_use_task_encoder:
            self.task_encoder = hydra.utils.instantiate(
                self.multitask_cfg.task_encoder_cfg.model_cfg,
            ).to(self.device) # components.task_encoder.TaskEncoder
            name = "task_encoder"
            self._components[name] = self.task_encoder
            # self.task_encoder_optimizer = hydra.utils.instantiate(
            #     self.multitask_cfg.task_encoder_cfg.optimizer_cfg,
            #     params=self.get_parameters(name=name),
            # )
            # self._optimizers[name] = self.task_encoder_optimizer

        if should_complete_init:
            self.complete_init(cfg_to_load_model=cfg_to_load_model)
    #############################################################################


    def complete_init(self, cfg_to_load_model: Optional[ConfigType]):
        if cfg_to_load_model:
            self.load(**cfg_to_load_model)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic
        # self.actor.encoder.copy_conv_weights_from(self.critic.encoder)
        self.train()

    def train(self, training: bool = True) -> None:
        self.training = training
        for name, component in self._components.items():
            if name != "log_alpha" and name != 'dpmm':
                component.train(training)

    def _make_encoder(
        self,
        env_obs_shape: List[int],
        encoder_cfg: ConfigType,
        multitask_cfg: ConfigType,
    ) -> encoder.Encoder:
        """Make the encoder.

        Args:
            env_obs_shape (List[int]):
            encoder_cfg (ConfigType):
            multitask_cfg (ConfigType):

        Returns:
            encoder.Encoder: encoder
        """
        return encoder.make_encoder(
            env_obs_shape=env_obs_shape,
            encoder_cfg=encoder_cfg,
            multitask_cfg=multitask_cfg,
        )

    def _make_bnpModel(self, bnp_cfg: ConfigType):
        '''
        build bayesian non-parametric model (DP-Mixture model) as prior of VAE Encoder
        before implement the model, the 'bnpy' package should be installed.
        for more detail please refer to https://bnpy.readthedocs.io/en/latest/
        '''
        save_dir = bnp_cfg.save_dir
        gamma0 = bnp_cfg.gamma0
        num_lap = bnp_cfg.num_lap
        sF = bnp_cfg.sF
        birth_kwargs = bnp_cfg.birth_kwargs
        merge_kwargs = bnp_cfg.merge_kwargs

        return BNPModel(
            save_dir=save_dir,
            gamma0=gamma0,
            num_lap=num_lap,
            sF=sF,
            birth_kwargs=birth_kwargs,
            merge_kwargs=merge_kwargs,
        )
    
    def encode(self,mtobs: MTObs, detach_encoder: bool = False) -> TensorType:
        '''
        encoding from VAE to get latent encoding z, check whether to combine with context encoding
        '''
        # env obs encoding
        if self.encoder_cfg.type == 'vae':
            # latent_encoding, latent_mu, latent_log_var = self.encoder(mtobs=mtobs, detach=detach_encoder)
            # return torch.cat((latent_encoding, context_encoding), dim=1), latent_mu, latent_log_var
            # mode 2: use context encoding as part of input of vae encoder
            latent_encoding, latent_mu, latent_log_var, reconstruction = self.encoder(mtobs=mtobs, detach=detach_encoder)
            return latent_encoding, latent_mu, latent_log_var, reconstruction
        else:
            # CARE
            task_info = mtobs.task_info
            context_encoding = task_info.encoding
            latent_encoding = self.encoder(mtobs=mtobs, detach=detach_encoder)
            return torch.cat((latent_encoding, context_encoding), dim=1), [], [], []
        
    def get_alpha(self, env_index: TensorType, latent_obs:TensorType=None) -> TensorType:
        """Get the alpha value for the given environments.

        Args:
            env_index (TensorType): environment index.

        Returns:
            TensorType: alpha values.
        """
        if self.multitask_cfg.should_use_disentangled_alpha:
            return self.log_alpha[env_index].exp()
        else:
            # return self.log_alpha[0].exp()
            #######################################
            return self.log_alpha(latent_obs).exp()
            #######################################

    def get_task_encoding(
        self, env_index: TensorType, modes: List[str], disable_grad: bool
    ) -> TensorType:
        """Get the Metadata context task encoding for the different environments.

        Args:
            env_index (TensorType): environment index.
            modes (List[str]):
            disable_grad (bool): should disable tracking gradient.

        Returns:
            TensorType: task encodings.
        """
        if disable_grad:
            with torch.no_grad():
                return self.task_encoder(env_index.to(self.device))
        return self.task_encoder(env_index.to(self.device))

    def act(
        self,
        multitask_obs: ObsType,
        # obs, env_index: TensorType,
        modes: List[str],
        sample: bool,
    ) -> np.ndarray:
        """Select/sample the action to perform.

        Args:
            multitask_obs (ObsType): Observation from the multitask environment.
            mode (List[str]): mode in which to select the action.
            sample (bool): sample (if `True`) or select (if `False`) an action.

        Returns:
            np.ndarray: selected/sample action.

        """
        env_obs = multitask_obs["env_obs"]
        env_index = multitask_obs["task_obs"]
        env_index = env_index.to(self.device, non_blocking=True)
        
        with torch.no_grad():
            if self.should_use_task_encoder:
                task_encoding = self.get_task_encoding(
                    env_index=env_index, modes=modes, disable_grad=True
                )
            else:
                task_encoding = None  # type: ignore[assignment]
            task_info = self.get_task_info(
                task_encoding=task_encoding, component_name="", env_index=env_index
            )
            obs = env_obs.float().to(self.device)
            
            if len(obs.shape) == 1 or len(obs.shape) == 3:
                obs = obs.unsqueeze(0)  # Make a batch
            mtobs = MTObs(env_obs=obs, task_obs=env_index, task_info=task_info)
            ##################################################
            # TODO modify to add vae encoder
            encoding, _, _, _ = self.encode(mtobs=mtobs)
            mu, pi, _, _ = self.actor(mtobs=mtobs, latent_obs=encoding)
            ##################################################
            if sample:
                action = pi
            else:
                action = mu
            action = action.clamp(*self.action_range)
            # assert action.ndim == 2 and action.shape[0] == 1
            return action.detach().cpu().numpy()

    def select_action(self, multitask_obs: ObsType, modes: List[str]) -> np.ndarray:
        '''
        used in eval mode
        '''
        return self.act(multitask_obs=multitask_obs, modes=modes, sample=False)

    def sample_action(self, multitask_obs: ObsType, modes: List[str]) -> np.ndarray:
        '''
        used in training mode
        '''
        return self.act(multitask_obs=multitask_obs, modes=modes, sample=True)

    def get_last_shared_layers(self, component_name: str) -> Optional[List[ModelType]]:  # type: ignore[return]

        if component_name in [
            "actor",
            "critic",
            "transition_model",
            "reward_decoder",
            "decoder",
        ]:
            return self._components[component_name].get_last_shared_layers()  # type: ignore[operator]
            # The mypy error is because self._components can contain a tensor as well.
        if component_name in ["log_alpha", "encoder", "task_encoder"]:
            return None
        if component_name not in self._components:
            raise ValueError(f"""Component named {component_name} does not exist""")

    def _compute_gradient(
        self,
        loss: TensorType,
        parameters: List[ParameterType],
        step: int,
        component_names: List[str],
        retain_graph: bool = False,
    ):
        """Method to override the gradient computation.

            Useful for algorithms like PCGrad and GradNorm.

        Args:
            loss (TensorType):
            parameters (List[ParameterType]):
            step (int): step for tracking the training of the agent.
            component_names (List[str]):
            retain_graph (bool, optional): if it should retain graph. Defaults to False.
        """
        loss.backward(retain_graph=retain_graph)

    def _get_target_V(
        self, batch: ReplayBufferSample, task_info: TaskInfo
    ) -> TensorType:
        """Compute the target values.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.

        Returns:
            TensorType: target values.
        """
        mtobs = MTObs(env_obs=batch.next_env_obs, task_obs=None, task_info=task_info)
        encoding,_,_,_ =self.encode(mtobs=mtobs)
        _, policy_action, log_pi, _ = self.actor(mtobs=mtobs, latent_obs = encoding)
        target_Q1, target_Q2 = self.critic_target(mtobs=mtobs, latent_obs = encoding ,action=policy_action)
        return (
            torch.min(target_Q1, target_Q2)
            - self.get_alpha(env_index=batch.task_obs, latent_obs=encoding).detach() * log_pi
        )
    
    def get_critic_loss(self, batch:ReplayBufferSample, task_info:TaskInfo):
        
        with torch.no_grad():
            target_V = self._get_target_V(batch=batch, task_info=task_info)
            target_Q = batch.reward + (batch.not_done * self.discount * target_V)

        # get current Q estimates
        mtobs = MTObs(env_obs=batch.env_obs, task_obs=None, task_info=task_info)
        encoding,_,_,_ =self.encode(mtobs=mtobs)
        current_Q1, current_Q2 = self.critic(
            mtobs=mtobs,
            latent_obs = encoding,
            action=batch.action,
            detach_encoder=False,
        )
        # mean loss
        critic_loss = F.mse_loss(
            current_Q1, target_Q, reduction=self.loss_reduction
        ) + F.mse_loss(current_Q2, target_Q, reduction=self.loss_reduction)

        return critic_loss
    
    def get_policy_alpha_loss(self, batch:ReplayBufferSample, task_info:TaskInfo):
        
        mtobs = MTObs(
            env_obs=batch.env_obs,
            task_obs=None,
            task_info=task_info,
        )
        encoding, _, _, _ = self.encode(mtobs=mtobs, detach_encoder=True) # encoding.detach()
        _, pi, log_pi, log_std = self.actor(mtobs=mtobs, latent_obs=encoding)
        actor_Q1, actor_Q2 = self.critic(mtobs=mtobs, latent_obs=encoding ,action=pi)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        
        if self.loss_reduction == "mean":
            
            policy_loss = (
                self.get_alpha(batch.task_obs, encoding).detach() * log_pi - actor_Q
            ).mean()
            alpha_loss = (
                self.get_alpha(batch.task_obs, encoding)
                * (-log_pi - self.target_entropy).detach()
            ).mean()

        elif self.loss_reduction == "none":

            policy_loss = self.get_alpha(batch.task_obs, encoding).detach() * log_pi - actor_Q
            alpha_loss = (self.get_alpha(batch.task_obs, encoding) * (-log_pi - self.target_entropy).detach())
        
        else:
            raise NotImplementedError('unable to compute policy loss, unsupported loss_reduction type: {}'.format(self.loss_reduction))
        
        return policy_loss, alpha_loss

    ######################################################
    ######## calculate KLD between q(z|s) & p(z) #########
    ######################################################
    # TODO check sampled mu_q and log_var dimension
    
    def kl_divergence(self, latent_obs, mu_q, log_var_q, kl_method='soft'):
        '''
        return the KL divergence D_kl(q||p), if bnp_model not exists, then calculate D_kl(q||N(0,I))
        method: soft, using soft assignment to calculate KLD between q(z|s) and DPMixture p(z)
        Input: 
            latent_obs: latent encoding from encoder, concat with context encoding
            mu_q: mean vector of latent encoding
            log_var_q: log variance vector of latent encoding

        Output:
            KL divergence value between 2 gaussian distributions KL(q(z|s)||p(z))
        '''
        assert not torch.isnan(mu_q).any(), mu_q
        assert not torch.isnan(log_var_q).any(), log_var_q
        assert not torch.isnan(latent_obs).any(), latent_obs
        
        
        if self.bnp_model.model is None:
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var_q - mu_q ** 2 - log_var_q.exp(), dim = 1), dim = 0).to(self.device)
        
        else:
            # get probability assignment of each z to each component & parameters of each component distribution
            prob_comps, comps = self.bnp_model.cluster_assignments(latent_obs)
            var_q = torch.exp(0.5 * log_var_q)**2
            
            if kl_method == 'soft':
                # get a distribution of the latent variable
                dist = torch.distributions.MultivariateNormal(
                    loc=mu_q.cpu(),
                    covariance_matrix=torch.diag_embed(var_q).cpu()
                )
                # get a distribution for each cluster
                B, K = prob_comps.shape # batch_shape, number of active components
                kl_qz_pz = torch.zeros(B)
                # build each component distribution & calculate the kl divergence D_kl(q || p)
                for k in range(K):
                    prob_k = prob_comps[:, k]
                    dist_k = torch.distributions.MultivariateNormal(
                        loc=self.bnp_model.comp_mu[k],
                        covariance_matrix=torch.diag_embed(self.bnp_model.comp_var[k])
                    )
                    expanded_dist_k = dist_k.expand(dist.batch_shape)    # batch_shape [batch_size], event_shape [latent_dim]
                    kld_k = torch.distributions.kl_divergence(dist, expanded_dist_k)   #  shape [batch_shape, ]
                    # soft assignment
                    kl_qz_pz += torch.from_numpy(prob_k) * kld_k
                
            else: # kl_method = hard
                # calcualte kl divergence via hard assignment: assigning to the most  likely learned DPMM cluster
                mu_comp = torch.zeros_like(mu_q)
                var_comp = torch.zeros_like(log_var_q)
                for i, k in enumerate(comps):
                    mu_comp[i, :] = self.bnp_model.comp_mu[k]
                    var_comp[i, :] = self.bnp_model.comp_var[k]
                var_q = torch.exp(0.5 * log_var_q)**2
                kl_qz_pz = self.bnp_model.kl_divergence_diagonal_gaussian(mu_q, mu_comp, var_q, var_comp)
            
            kld_loss = torch.mean(kl_qz_pz).to(self.device)
        
        return kld_loss
        
    def update_critic(
        self,
        # critic_loss,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
    ) -> None:
        """Update the critic component.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """

        critic_loss = self.get_critic_loss(batch=batch, task_info=task_info)
        
        loss_to_log = critic_loss
        if self.loss_reduction == "none":
            loss_to_log = loss_to_log.mean()
        logger.log("train/critic_loss", loss_to_log, step)

        if loss_to_log > 1e8:
            warnings.warn("critic_loss = {} is too high. Stopping training.".format(loss_to_log))
            # raise RuntimeError(
            #     f"critic_loss = {loss_to_log} is too high. Stopping training."
            # )
        
        component_names = ["critic"]
        parameters: List[ParameterType] = []
        for name in component_names:
            self._optimizers[name].zero_grad()
            parameters += self.get_parameters(name)
        if task_info.compute_grad:
            component_names.append("task_encoder")
            kwargs_to_compute_gradient["retain_graph"] = True
            parameters += self.get_parameters("task_encoder")
        
        # critic loss backward
        self._compute_gradient(
            loss=critic_loss,
            parameters=parameters, # critic + task encoder
            step=step,
            component_names=component_names,
            **kwargs_to_compute_gradient,
        )
        # Optimize the critic
        self.critic_optimizer.step()

    def update_actor_and_alpha(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
    ) -> None:
        """Update the actor and alpha component.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """
        
        # detached context encoder, so we don't update it with the actor loss
        mtobs = MTObs(
            env_obs=batch.env_obs,
            task_obs=None,
            task_info=task_info, # encoding.detach()
        )
        # policy loss
        ###################################################################################
        encoding, _, _, _ = self.encode(mtobs=mtobs, detach_encoder=True)
        _, pi, log_pi, log_std = self.actor(mtobs=mtobs, latent_obs=encoding)
        actor_Q1, actor_Q2 = self.critic(mtobs=mtobs, latent_obs=encoding ,action=pi)
        ###################################################################################

        actor_Q = torch.min(actor_Q1, actor_Q2)
        
        if self.loss_reduction == "mean":
            actor_loss = (
                self.get_alpha(batch.task_obs, encoding).detach() * log_pi - actor_Q
            ).mean()
            logger.log("train/actor_loss", actor_loss, step)

        elif self.loss_reduction == "none":
            actor_loss = self.get_alpha(batch.task_obs, encoding).detach() * log_pi - actor_Q
            logger.log("train/actor_loss", actor_loss.mean(), step)

        # logger.log("train/actor_target_entropy", self.target_entropy, step)
        # entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(
        #     dim=-1
        # )
        # logger.log("train/actor_entropy", entropy.mean(), step)

        component_names = ["actor"]
        parameters: List[ParameterType] = []
        for name in component_names:
            self._optimizers[name].zero_grad()
            parameters += self.get_parameters(name)
        if task_info.compute_grad: # False
            component_names.append("task_encoder")
            kwargs_to_compute_gradient["retain_graph"] = True
            parameters += self.get_parameters("task_encoder")
        # actor loss backward
        self._compute_gradient(
            loss=actor_loss,
            parameters=parameters,
            step=step,
            component_names=component_names,
            **kwargs_to_compute_gradient,
        )
        self.actor_optimizer.step()

        # alpha loss
        self.log_alpha_optimizer.zero_grad()

        if self.loss_reduction == "mean":
            alpha_loss = (
                self.get_alpha(batch.task_obs, encoding)
                * (-log_pi - self.target_entropy).detach()
            ).mean()
            logger.log("train/alpha_loss", alpha_loss, step)
        elif self.loss_reduction == "none":
            alpha_loss = (
                self.get_alpha(batch.task_obs, encoding)
                * (-log_pi - self.target_entropy).detach()
            )
            logger.log("train/alpha_loss", alpha_loss.mean(), step)
        # breakpoint()
        # logger.log("train/alpha_value", self.get_alpha(batch.task_obs, encoding), step)
        ########################################################
        if not self.multitask_cfg.should_use_disentangled_alpha:
            kwargs_to_compute_gradient["retain_graph"] = True
        ########################################################
        self._compute_gradient(
            loss=alpha_loss,
            parameters=self.get_parameters(name="log_alpha"),
            step=step,
            component_names=["log_alpha"],
            **kwargs_to_compute_gradient,
        )
        self.log_alpha_optimizer.step()

    def get_task_info(
        self, task_encoding: TensorType, component_name: str, env_index: TensorType
    ) -> TaskInfo:
        """Encode task encoding into task info.

        Args:
            task_encoding (TensorType): encoding of the task.
            component_name (str): name of the component.
            env_index (TensorType): index of the environment.

        Returns:
            TaskInfo: TaskInfo object.
        """
        if self.should_use_task_encoder:
            # critic, tesk_encoder, encoder
            if component_name in self.multitask_cfg.task_encoder_cfg.losses_to_train:
                task_info = TaskInfo(
                    encoding=task_encoding, compute_grad=True, env_index=env_index
                )
            else: # actor
                task_info = TaskInfo(
                    encoding=task_encoding.detach(),
                    compute_grad=False,
                    env_index=env_index,
                )
        else:
            task_info = TaskInfo(
                encoding=task_encoding, compute_grad=False, env_index=env_index
            )
        return task_info

    # out of date
    def update_task_encoder(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
    ) -> None:
        """Update the task encoder component.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """
        self.task_encoder_optimizer.step()

    def update_vae(self,
        batch:ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,):
        
        # update VAE using information bottleneck via DPMixture Model
        mtobs = MTObs(env_obs=batch.env_obs, task_obs=batch.task_obs, task_info=task_info)
        latent_variable, latent_mu, latent_log_var, reconstruction =self.encode(mtobs=mtobs)
        
        # update the VAE according to the frequence
        if (step+1) % self.kl_div_update_freq == 0:
            # 1. KL Divergence
            kld_loss = self.kl_divergence(latent_obs=latent_variable, mu_q=latent_mu, log_var_q=latent_log_var)

            # 2. reconstruction loss 
            reconstruction_loss = self.reconstruction_loss(reconstruction, task_info.encoding)
            # 3. total VAE loss E[ logP(x|z) - beta * KL[q(z)|p(z)] ]
            vae_loss = reconstruction_loss + self.beta_kl_z * kld_loss
            
            # 4. backpropargation
            self.encoder_optimizer.zero_grad()
            vae_loss.backward()
            self.encoder_optimizer.step()
            # ELBO objective (only for logging)
            vae_loss_to_log = vae_loss
            logger.log("train/vae_loss", vae_loss_to_log, step)

        return latent_variable

    # TODO add VAE, dpmm update
    def update(
        self,
        replay_buffer: ReplayBuffer,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Optional[Dict[str, Any]] = None,
        buffer_index_to_sample: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """Update the agent.

        Args:
            replay_buffer (ReplayBuffer): replay buffer to sample the data.
            logger (Logger): logger for logging.
            step (int): step for tracking the training progress.
            kwargs_to_compute_gradient (Optional[Dict[str, Any]], optional): Defaults
                to None.
            buffer_index_to_sample (Optional[np.ndarray], optional): if this parameter
                is specified, use these indices instead of sampling from the replay
                buffer. If this is set to `None`, sample from the replay buffer.
                buffer_index_to_sample Defaults to None.

        Returns:
            np.ndarray: index sampled (from the replay buffer) to train the model. If
                buffer_index_to_sample is not set to None, return buffer_index_to_sample.

        """

        if kwargs_to_compute_gradient is None:
            kwargs_to_compute_gradient = {}

        if buffer_index_to_sample is None:
            batch = replay_buffer.sample()
        else:
            batch = replay_buffer.sample(buffer_index_to_sample)
        
        # self.encoder_optimizer.zero_grad()

        if self.should_use_task_encoder:
            # self.task_encoder_optimizer.zero_grad()
            task_encoding = self.get_task_encoding(
                env_index=batch.task_obs.squeeze(1),
                disable_grad=False,
                modes=["train"],
            )
        else:
            task_encoding = None  # type: ignore[assignment]

        task_info = self.get_task_info(
            task_encoding=task_encoding,
            component_name="critic",
            env_index=batch.task_obs,
        )
        
        if self.encoder_cfg.type in ['vae']:
            latent_variable = self.update_vae(
                batch, task_info, 
                logger, step, 
                kwargs_to_compute_gradient
                )

        # update critic
        self.update_critic(
            batch=batch,
            task_info=task_info,
            logger=logger,
            step=step,
            kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
        )

        # update actor & alpha
        if step % self.actor_update_freq == 0:
            task_info = self.get_task_info(
                task_encoding=task_encoding,
                component_name="actor", # compute_grad=False, task_encoding.detach()
                env_index=batch.task_obs,
            )
            self.update_actor_and_alpha(
                batch=batch,
                task_info=task_info,
                logger=logger,
                step=step,
                kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
            )
        # update target critic (soft update)
        if step % self.critic_target_update_freq == 0:
            agent_utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            agent_utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )

        ############################################################################
        # update DPMM model at certain interval
        if self.should_use_dpmm:
            if kwargs['local_step'] is not None:
                local_step = kwargs['local_step']
            else:
                local_step = step
            if ((local_step+1) == self.dpmm_update_start_step or 
                (local_step+1) % self.dpmm_update_freq == 0
                ):
                print('fit bnp_model at step: {}'.format(step+1))
                # using latent encoding as training objective
                self.bnp_model.fit(latent_variable)
                self.bnp_model.plot_clusters(latent_variable, suffix=str(step))
                logger.log('train/K_comps', self.bnp_model.model.obsModel.K, step)
                # save a sample of latent variable during training
                if kwargs['task_name_to_idx'] is not None:
                    if kwargs['subtask'] is not None:
                        prefix = 'step{}_subtask{}'.format(step+1, kwargs["subtask"])
                    else:
                        prefix = 'step{}'.format(step+1)
                    
                    self.evaluate_latent_clustering(replay_buffer=replay_buffer, 
                                                task_name_to_idx_map=kwargs['task_name_to_idx'],
                                                num_save=1,
                                                prefix=prefix)

        ############################################################################
        return batch.buffer_index

    def evaluate_latent_clustering(self, 
                                   replay_buffer: ReplayBuffer, 
                                   task_name_to_idx_map:dict, 
                                   num_save:int=1, 
                                   prefix:str="", 
                                   save_info_dict:bool=False
                                   ):
        '''
        sample a batch of obs, get latent encoding & evaluate the clustering performance, save the
        necessary data to *csv file
        '''
        for i in range(num_save):
            # sample batch
            batch = replay_buffer.sample()
            if self.should_use_task_encoder:
                task_encoding = self.get_task_encoding(
                        env_index=batch.task_obs.squeeze(1),
                        disable_grad=False,
                        modes=["train"],
                    )
            else:
                task_encoding=None
            
            task_info = self.get_task_info(
                task_encoding=task_encoding,
                component_name="",
                env_index=batch.task_obs,
            )
            # get env name
            env_name_list = []
            def get_env_name(map, idx):
                return [name for name, val in map.items() if val==idx]
            for idx in batch.task_obs.squeeze(1):
                env_name_list.append(get_env_name(task_name_to_idx_map, idx)[0])
            mtobs = MTObs(env_obs=batch.env_obs, task_obs=None, task_info=task_info)
            # get encoding
            z, _, _, _ = self.encode(mtobs=mtobs)
            
            if self.bnp_model is not None and self.bnp_model.model is not None:
                self.bnp_model.manage_latent_representation(z=z, 
                                                            env_idx=batch.task_obs, 
                                                            env_name_list=env_name_list, 
                                                            prefix=prefix+'_'+str(i+1),
                                                            save_info_dict=save_info_dict,
                                                            )
            else:
                data = dict(
                    z = z.detach().cpu().numpy(),
                    env_name = env_name_list,
                    env_idx = batch.task_obs.detach().cpu().numpy()
                )
                
                np.savez(self.multitask_cfg.dpmm_cfg.save_dir+'/latent_ohne_DPMixture_{}.npz'.format(i), **data)

    def get_parameters(self, name: str) -> List[torch.nn.parameter.Parameter]:
        """Get parameters corresponding to a given component.

        Args:
            name (str): name of the component.

        Returns:
            List[torch.nn.parameter.Parameter]: list of parameters.
        """
        if name == "actor":
            return list(self.actor.model.parameters())
        
        
        elif name in ["log_alpha", "alpha"]:
            if self.multitask_cfg.should_use_disentangled_alpha:
                return [self.log_alpha]
            else:
                return list(self._components[name].parameters())

        else:
            return list(self._components[name].parameters())
    
