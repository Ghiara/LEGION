# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Dict
import gym
import torch
from gym.vector.async_vector_env import AsyncVectorEnv
import numpy as np

class VecEnv(AsyncVectorEnv):
    def __init__(
        self,
        env_metadata: Dict[str, Any],
        env_fns,
        observation_space=None,
        action_space=None,
        shared_memory=True,
        copy=True,
        context=None,
        daemon=True,
        worker=None,
    ):
        """Return only every `skip`-th frame"""
        super().__init__(
            env_fns=env_fns,
            observation_space=observation_space,
            action_space=action_space,
            shared_memory=shared_memory,
            copy=copy,
            context=context,
            daemon=daemon,
            worker=worker,
        )
        self.num_envs = len(env_fns)
        assert "mode" in env_metadata
        assert "ids" in env_metadata
        self._metadata = env_metadata

    @property
    def mode(self):
        return self._metadata["mode"]

    @property
    def ids(self):
        return self._metadata["ids"]

    def reset(self):
        multitask_obs = super().reset()
        return _cast_multitask_obs(multitask_obs=multitask_obs)

    def step(self, actions):
        multitask_obs, reward, done, info = super().step(actions)
        return _cast_multitask_obs(multitask_obs=multitask_obs), reward, done, info


def _cast_multitask_obs(multitask_obs):
    return {key: torch.tensor(value) for key, value in multitask_obs.items()}


class MetaWorldVecEnv(AsyncVectorEnv):
    def __init__(
        self,
        env_metadata: Dict[str, Any],
        env_fns,
        observation_space=None,
        action_space=None,
        shared_memory=True,
        copy=True,
        context=None,
        daemon=True,
        worker=None,
        ################
        use_onehot=False,
        ################
    ):
        """Return only every `skip`-th frame"""
        super().__init__(
            env_fns=env_fns,
            observation_space=observation_space,
            action_space=action_space,
            shared_memory=shared_memory,
            copy=copy,
            context=context,
            daemon=daemon,
            worker=worker,
        )
        self.num_envs = len(env_fns)
        self.task_obs = torch.arange(self.num_envs)
        assert "mode" in env_metadata
        assert "ids" in env_metadata
        self._metadata = env_metadata
        ##############################
        self.env_version = env_metadata['env_version']
        self.use_onehot = use_onehot
        ##############################

    @property
    def mode(self):
        return self._metadata["mode"]

    @property
    def ids(self):
        return self._metadata["ids"]

    def _check_observation_spaces(self):
        return

    def reset(self):
        env_obs = super().reset()
        return self.create_multitask_obs(env_obs=env_obs)

    def step(self, actions):
        env_obs, reward, done, info = super().step(actions)
        return self.create_multitask_obs(env_obs=env_obs), reward, done, info

    def create_multitask_obs(self, env_obs):
        #########################################################
        if self.use_onehot:
            one_hot = self.one_hot(self.task_obs)
            return {"env_obs": torch.cat([torch.tensor(env_obs), one_hot],dim=-1), 
                    "task_obs": self.task_obs, }
        #########################################################
        return {"env_obs": torch.tensor(env_obs), "task_obs": self.task_obs, }
        
        # rerturn env obs and task ID

    def one_hot(self, task_obs):
        one_hot = torch.diag_embed(task_obs+1)
        one_hot = (one_hot>0).long()
        return one_hot




################################################
############ CRL Env setting - V2 ##############
################################################

class CRL_Env():
    def __init__(
            self,
            env_list:list,
            config,
            ordered_task_list:list,
            ):
        self.exp_config = config.experiment
        self.env_config = config.env
        self.env_list = env_list
        self.ordered_task_list = ordered_task_list

        self.training_mode = self.exp_config.training_mode
        assert self.training_mode in ['crl_queue', 'crl_expand'], \
            'invalid training mode: {}'.format(self.training_mode)
        
        self.current_phase = 0 # count for which phase currently running on (10 envs have 10 phases)
        self.current_env_num = 1
        self.current_env_name = self.ordered_task_list[self.current_phase]
        
        
    def step(self, actions):
        
        if self.training_mode == 'crl_queue':
            env_obs, reward, done, info = self.env_list[self.current_phase].step(actions[0])
            env_obs = np.array([env_obs])
            reward = np.array([reward])
            done = np.array([done])
            info = tuple([info])
        else: 
            # crl expand
            assert actions.shape[0] == self.current_env_num, \
                'number of actions {} is not equal as number of envs {}'.format(actions.shape[0], self.current_env_num)
            env_obs = []
            reward = []
            done = []
            info = []
            for i in range(self.current_env_num):
                obs, rew, d, ifo = self.env_list[i].step(actions[i])
                env_obs.append(obs)
                reward.append(rew)
                done.append(d)
                info.append(ifo)
            env_obs = np.array(env_obs)
            reward = np.array(reward)
            done = np.array(done)
            info = tuple(info)
        
        return self.create_crl_obs(env_obs=env_obs, 
                                task_idx=[self.current_phase] 
                                if self.training_mode == 'crl_queue' 
                                else list(np.arange(self.current_phase+1))), reward, done, info
        

    def reset(self, subtask_idx):
        '''
        reset the environments according to the current subtask_idx (number of phase) & training mode
        '''
        if self.current_phase == subtask_idx:
            pass
        else:
            # activate at each phase beginning, confirm which env(s) should output
            self.current_phase = subtask_idx
            self._reset_subtask_setup()

        if self.training_mode == 'crl_queue':
            env_obs = self.env_list[self.current_phase].reset()
        else: # crl expand
            env_obs = []
            for i in range(self.current_env_num):
                obs = self.env_list[i].reset()
                env_obs.append(obs)
            env_obs = np.array(env_obs)
        return self.create_crl_obs(env_obs=env_obs, 
                                task_idx=[self.current_phase] 
                                if self.training_mode == 'crl_queue' 
                                else list(np.arange(self.current_phase+1)))

    def _reset_subtask_setup(self):
        # check where the phase the agent is running on
        if self.training_mode == 'crl_expand':
            self.current_env_num = int(self.current_phase + 1)
            self.current_env_name = self.ordered_task_list[:self.current_env_num]
        else:
            # crl_queue
            self.current_env_num = 1
            self.current_env_name = self.ordered_task_list[self.current_phase]

    def set_train_mode(self):
        return ["train" for _ in range(self.current_env_num)]
    

    def create_crl_obs(self, env_obs, task_idx):
        return {"env_obs": torch.tensor(env_obs), "task_obs": torch.tensor(task_idx)}
        # rerturn env obs and task ID
    
    def close(self):
        for env in self.env_list:
            env.close()
    
    # def remove_goal_bounds(obs_space: gym.spaces.Box) -> None:
    #     obs_space.low[9:12] = -np.inf
    #     obs_space.high[9:12] = np.inf