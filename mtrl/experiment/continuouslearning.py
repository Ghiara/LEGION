import os
import errno
import time
from typing import Dict, List, Tuple
import hydra
import numpy as np
import torch

from mtrl.agent import utils as agent_utils
from mtrl.env import builder as env_builder
from mtrl.env.types import EnvType
from mtrl.env.vec_env import VecEnv, CRL_Env  # type: ignore
from mtrl.experiment import experiment
from mtrl.utils.types import ConfigType, EnvMetaDataType, EnvsDictType, ListConfigType
from mtrl.logger import CRL_Metrics
from mtrl.utils import utils


class Experiment(experiment.Experiment):
    def __init__(self, config: ConfigType, experiment_id: str = "0"):
        """Experiment Class to manage the lifecycle of a multi-task model.

        Args:
            config (ConfigType):
            experiment_id (str, optional): Defaults to "0".
        """
        self.list_envs = None
        self.env_id_to_task_map_recording = None

        super().__init__(config, experiment_id)
        self.eval_modes_to_env_ids = self.create_eval_modes_to_env_ids()
        self.should_reset_env_manually = True
        self.metrics_to_track = {
            x[0] for x in self.config.metrics["train"] if not x[0].endswith("_")
        }
        # save for best model with hightest success rate
        self.best_success_rate = 0.0
        self.video_save_count = 0
        if self.config.experiment.training_mode not in ['multitask']:
            self.crl_metrics_dir = utils.make_dir(
            os.path.join(self.config.setup.save_dir, "crl_metrics")
            )
            self.crl_metrics = CRL_Metrics(save_dir=self.crl_metrics_dir)
        
    

    def create_eval_modes_to_env_ids(self):
        eval_modes_to_env_ids = {}
        eval_modes = [
            key for key in self.config.metrics.keys() if not key.startswith("train")
        ]
        for mode in eval_modes:
            if self.config.env.benchmark._target_ in [
                "metaworld.ML1",
                "metaworld.MT1",
                "metaworld.MT10",
                "metaworld.MT50",
            ]:
                eval_modes_to_env_ids[mode] = list(range(self.config.env.num_envs))
            else:
                raise ValueError(
                    f"`{self.config.env.benchmark._target_}` env is not supported by metaworld experiment."
                )

        return eval_modes_to_env_ids


    def build_envs(self) -> Tuple[EnvsDictType, EnvMetaDataType]:
        
        if self.config.env.use_kuka_env is not True:
            benchmark = hydra.utils.instantiate(self.config.env.benchmark)
        else:
            benchmark = hydra.utils.instantiate(self.config.env.kuka_benchmark)
        envs = {}
        
        if self.config.experiment.training_mode in ['multitask']:
            mode = "train"
            envs[mode], env_id_to_task_map = env_builder.build_metaworld_vec_env(
                config=self.config, benchmark=benchmark, mode=mode, env_id_to_task_map=None
            )
            mode = "eval"
            envs[mode], env_id_to_task_map = env_builder.build_metaworld_vec_env(
                config=self.config,
                benchmark=benchmark,
                mode="train",
                env_id_to_task_map=env_id_to_task_map,
            )
        else: # CRL
            mode = "train"
            env_list, env_id_to_task_map = env_builder.build_metaworld_env_list(
                config=self.config, 
                benchmark=benchmark, 
                mode=mode, 
                env_id_to_task_map=None
            )
            envs[mode] = CRL_Env(env_list=env_list, 
                                 config=self.config, 
                                 ordered_task_list=list(env_id_to_task_map.keys()))
            mode = "eval"
            envs[mode], env_id_to_task_map = env_builder.build_metaworld_vec_env(
                config=self.config,
                benchmark=benchmark,
                mode="train", # in MT setting there is no eval mode
                env_id_to_task_map=env_id_to_task_map,
            ) # for CRL the eval phase use the vec env
        
        # build a set of envs for video recording
        if self.config.experiment.save_video:
            list_envs, env_id_to_task_map_recording = env_builder.build_metaworld_env_list(
                config=self.config,
                benchmark=benchmark,
                mode="train",
                env_id_to_task_map=env_id_to_task_map,
            )
            self.list_envs = list_envs
            self.env_id_to_task_map_recording = env_id_to_task_map_recording


        max_episode_steps = 150
        # hardcoding the steps as different environments return different
        # values for max_path_length. MetaWorld uses 150 as the max length.
        metadata = self.get_env_metadata(
            # env=envs["train"], # change to eval to suit for CRL
            env=envs['eval'],
            max_episode_steps=max_episode_steps,
            ordered_task_list=list(env_id_to_task_map.keys()),
            config=self.config
        )
        return envs, metadata
    

    def create_env_id_to_index_map(self) -> Dict[str, int]:
        env_id_to_index_map: Dict[str, int] = {}
        current_id = 0
        for env in self.envs.values():
            assert isinstance(env, VecEnv)
            for env_name in env.ids:
                if env_name not in env_id_to_index_map:
                    env_id_to_index_map[env_name] = current_id
                    current_id += 1
        return env_id_to_index_map

    # from metaworld check & modify
    def evaluate_vec_env_of_tasks(self, vec_env: VecEnv, step: int, episode: int, record_crl_metrics:bool=False):
        """Evaluate the agent's performance on the different environments,
        vectorized as a single instance of vectorized environment.

        Since we are evaluating on multiple tasks, we track additional metadata
        to track which metric corresponds to which task.

        Args:
            vec_env (VecEnv): vectorized environment.
            step (int): step for tracking the training of the agent.
            episode (int): episode for tracking the training of the agent.

            in CRL setting we use "episode" to record the number of current subtask phase
            in CRL setting we use "step" to record the global step of current evaluation process
        """
        agent = self.agent
        for mode in self.eval_modes_to_env_ids:
            if self.config.experiment.training_mode in ['multitask']:
                self.logger.log(f"{mode}/episode", episode, step)
            else:
                self.logger.log(f"{mode}/subtask", episode, step)


        episode_reward, mask, done, success = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 1.0, False, 0.0]
        ]  # (num_envs, 1)
        
        # number of evaluate episodes
        num_eval_episodes = self.config.experiment.num_eval_episodes

        for _ in range(num_eval_episodes):
            multitask_obs = vec_env.reset() 
            # multitask_obs['env_obs'].shape (torch.Size([10, 12]))
            # 'task_obs': tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            episode_step = 0
            offset = 1 # by default we assume each type env has 1 instance
            success_eps = np.full(shape=vec_env.num_envs, fill_value=0.0)
            while episode_step < self.max_episode_steps:
                with agent_utils.eval_mode(agent):
                    action = agent.select_action(
                        multitask_obs=multitask_obs, modes=["eval"]
                    )
                # action dims MT10: (10, 4)
                multitask_obs, reward, done, info = vec_env.step(action)
                success_eps += np.asarray([x["success"] for x in info])
                mask = mask * (1 - done.astype(int))
                episode_reward += reward * mask
                episode_step += 1
            success += (success_eps > 0).astype("float")

        
        start_index = 0
        # success = (success > 0).astype("float")
        for mode in self.eval_modes_to_env_ids: 
            num_envs = len(self.eval_modes_to_env_ids[mode])
            # total mean reward of all envs
            self.logger.log(
                f"{mode}/episode_reward",
                # episode_reward[start_index : start_index + offset * num_envs].mean(),
                episode_reward[start_index : start_index + offset * num_envs].sum() / (num_eval_episodes*num_envs),
                step,
            )
            # total mean success rate of all envs
            self.logger.log(
                f"{mode}/success",
                # success[start_index : start_index + offset * num_envs].mean(),
                success[start_index : start_index + offset * num_envs].sum() / (num_eval_episodes*num_envs),
                step,
            )

            # check & save best model & showcase videos (when avg success > 0.95, record 1 time)
            if (success[start_index : start_index + offset * num_envs].sum() / (num_eval_episodes*num_envs) > self.best_success_rate):
            
                self.best_success_rate = success[start_index : start_index + offset * num_envs].sum() / (num_eval_episodes*num_envs)
                
                self.agent.save_best_model(
                    self.model_dir,
                    str(self.best_success_rate),
                )
                if self.config.experiment.save_video and self.best_success_rate >= 0.95 and self.video_save_count < 1:
                    print('save best video records.')
                    self.record_videos(step=step)
                    self.video_save_count += 1
            
            # every env reward & success rate, num of each env = config.experiment.num_eval_episodes
            for _current_env_index, _current_env_id in enumerate(
                self.eval_modes_to_env_ids[mode] # 'eval': [0,1,2,...9]
            ):
                subtask_reward = episode_reward[start_index + _current_env_index * offset : 
                                                start_index + (_current_env_index + 1) * offset
                                               ].sum() / num_eval_episodes
                subtask_success = success[start_index + _current_env_index * offset : 
                                          start_index + (_current_env_index + 1) * offset
                                         ].sum()/num_eval_episodes
                self.logger.log(
                    f"{mode}/episode_reward_env_index_{_current_env_index}",
                    # episode_reward[
                    #     start_index
                    #     + _current_env_index * offset : start_index
                    #     + (_current_env_index + 1) * offset
                    # ].sum() / num_eval_episodes,
                    subtask_reward,
                    step,
                )
                self.logger.log(
                    f"{mode}/success_env_index_{_current_env_index}",
                    # success[
                    #     start_index
                    #     + _current_env_index * offset : start_index
                    #     + (_current_env_index + 1) * offset
                    # ].sum()/num_eval_episodes,
                    subtask_success,
                    step,
                )
                self.logger.log(
                    f"{mode}/env_index_{_current_env_index}", _current_env_id, step, tb_log=False
                )
                if self.config.experiment.training_mode in ['crl_queue', 'crl_expand']:
                    if record_crl_metrics:
                        self.crl_metrics.add(reward=subtask_reward, success_rate=subtask_success)  
            start_index += offset * num_envs
        self.logger.dump(step)



        #######################################################################
        if self.config.experiment.training_mode in ['crl_queue', 'crl_expand']:
            if record_crl_metrics:
                self.crl_metrics.update()
                self.crl_metrics.save_metrics()
        #######################################################################

    def run(self):
        '''
        Run the training loop depend on learning modes:
        1. Multitask learning mode
        2. Lifelong learning mode
        '''

        try:
            if self.config.experiment.training_mode in ['multitask']:
                self.run_multitask()
            else:
                self.run_crl()
        except IOError as e: # prevent from pipe broken error [Errno 32]
            if e.errno == errno.EPIPE:
                pass




    def run_multitask(self) -> None:
        """Run the experiment under multitask setting."""


        exp_config = self.config.experiment
        vec_env = self.envs["train"]
        
        # logged values for training phase
        episode_reward, episode_step, done = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 0, True]
        ]  # (num_envs, 1)
        if "success" in self.metrics_to_track:
            success = np.full(shape=vec_env.num_envs, fill_value=0.0)
        info = {}
        assert self.start_step >= 0
        episode = self.start_step // self.max_episode_steps

        start_time = time.time()
        # reset envs
        multitask_obs = vec_env.reset()
        env_indices = multitask_obs["task_obs"]
        train_mode = ["train" for _ in range(vec_env.num_envs)]
        
        # counter for saving best model
        best_success_rate = 0.0
        ##############
        # train loop #
        ##############
        for step in range(self.start_step, exp_config.num_train_steps):
            
            # evaluate agent periodically
            if step % exp_config.eval_freq == 0:
                self.evaluate_vec_env_of_tasks(vec_env=self.envs["eval"], 
                                                step=step, episode=episode)
                # save model
                if exp_config.save.model:
                    self.agent.save(
                        self.model_dir,
                        step=step,
                        retain_last_n=exp_config.save.model.retain_last_n, # 1, last 1
                    )
                if exp_config.save.buffer.should_save:
                    self.replay_buffer.save(
                        self.buffer_dir,
                        size_per_chunk=exp_config.save.buffer.size_per_chunk,
                        num_samples_to_save=exp_config.save.buffer.num_samples_to_save,
                    )

            # log after every episode
            if step % self.max_episode_steps == 0:  # Perform logging & Evaluation after every episode
                if step > 0:
                    if "success" in self.metrics_to_track:
                        success = (success > 0).astype("float")
                        for index, _ in enumerate(env_indices):
                            self.logger.log(
                                f"train/success_env_index_{index}",
                                success[index],
                                step,
                            )
                        self.logger.log("train/success", success.mean(), step)

                    for index, env_index in enumerate(env_indices):
                        self.logger.log(
                            f"train/episode_reward_env_index_{index}",
                            episode_reward[index],
                            step,
                        )
                        self.logger.log(f"train/env_index_{index}", env_index, step, tb_log=False)

                    self.logger.log("train/duration", time.time() - start_time, step, tb_log=False)
                    start_time = time.time()
                    self.logger.dump(step) # dump to file & console
                episode += 1
                # reset logged value
                episode_reward = np.full(shape=vec_env.num_envs, fill_value=0.0)
                if "success" in self.metrics_to_track:
                    success = np.full(shape=vec_env.num_envs, fill_value=0.0)
                self.logger.log("train/episode", episode, step, tb_log=False)



            if step < exp_config.init_steps:
                # warmup steps
                action = np.asarray(
                    [self.action_space.sample() for _ in range(vec_env.num_envs)]
                )  # (num_envs, action_dim)

            else:
                with agent_utils.eval_mode(self.agent):
                    # multitask_obs = {"env_obs": obs, "task_obs": env_indices}
                    action = self.agent.sample_action(
                        multitask_obs=multitask_obs,
                        modes=[
                            train_mode,
                        ],
                    )  # (num_envs, action_dim)

            # run training update (by default each step update 1 time)
            if step >= exp_config.init_steps:
                num_updates = (
                    exp_config.init_steps if step == exp_config.init_steps else 1
                )
                for _ in range(num_updates):
                    self.agent.update(self.replay_buffer, self.logger, step, 
                                      task_name_to_idx=self.task_name_to_idx) # used for evaluation
            
            # Interaction with envs
            next_multitask_obs, reward, done, info = vec_env.step(action)
            if self.should_reset_env_manually:
                if (episode_step[0] + 1) % self.max_episode_steps == 0:
                    # we do a +2 because we started the counting from 0 and episode_step is incremented after updating the buffer
                    next_multitask_obs = vec_env.reset()

            episode_reward += reward
            if "success" in self.metrics_to_track:
                success += np.asarray([x["success"] for x in info])

            # allow infinite bootstrap
            # Add transitions into replay buffer
            for index, env_index in enumerate(env_indices):
                done_bool = (
                    0
                    if episode_step[index] + 1 == self.max_episode_steps
                    else float(done[index])
                )
                # add each separate env data into buffer
                if index not in self.envs_to_exclude_during_training:
                    self.replay_buffer.add(
                        multitask_obs["env_obs"][index],
                        action[index],
                        reward[index],
                        next_multitask_obs["env_obs"][index],
                        done_bool,
                        task_obs=env_index,
                        # one_hot=multitask_obs["one_hot"][index],
                        # true_label=index
                    )

            multitask_obs = next_multitask_obs
            episode_step += 1
        ######################################
        ########### End of Training ##########
        ######################################

        ####################################################################
        if self.config.experiment.eval_latent_representation:
            print('saving latent clustering data for further evaluation...')
            self.agent.evaluate_latent_clustering(self.replay_buffer, 
                                                  self.task_name_to_idx, 
                                                  num_save=self.config.experiment.num_save,
                                                  prefix='final'
                                                  )

        if self.config.experiment.save_video:
            print('start recording videos ...')
            self.record_videos(step=step)
            print('video recording finished. Check folder:{}'.format(self.video.dir_name))
        ####################################################################
        
        self.replay_buffer.delete_from_filesystem(self.buffer_dir)
        self.close_envs()
        print('Training finished')

    ##############################################
    ########### Run continual learning ###########
    ##############################################
    def run_crl(self):
        """Run the experiment under crl setting.
        """

        exp_config = self.config.experiment
        assert exp_config.training_mode in ['crl_queue', 'crl_expand'], \
            'incorrect training_mode for crl training.'
        # crl_env = CRL_Env instance
        crl_env = self.envs["train"]
        
        ########################
        ### start train loop ###
        ########################
        global_step = 0
        global_episode = 0

        # for subtask in range(self.config.env.num_envs):
        for subtask in range(0, self.config.env.num_envs):
 
            # set up each subtask
            crl_obs = crl_env.reset(subtask)
            env_indices = crl_obs["task_obs"]
            # change action space if needed
            if exp_config.training_mode in ['crl_queue']:
                self.action_space = crl_env.env_list[subtask].action_space
            # reset critics & target critics
            if exp_config.should_reset_critics and subtask>0:
                is_critics_reset = self.agent.reset_critics()
            else:
                is_critics_reset = False
            # reset vae
            if exp_config.should_reset_vae and subtask>0:
                is_vae_reset = self.agent.reset_vae()
            else:
                is_vae_reset = False

            # reset optimizer (make sure only reset optimizer after reseting the model)
            if exp_config.should_reset_optimizer and subtask>0:
                is_optim_reset = self.agent.reset_optimizer()
            else:
                is_optim_reset = False
            # reset metrics
            success, episode_reward, episode_step, done = [
                np.full(shape=crl_env.current_env_num, fill_value=fill_value)
                for fill_value in [0.0, 0.0, 0, True]
            ]
            info = {}
            episode = 0
            
            self.logger.log_text('train/subtask_name', crl_env.current_env_name, subtask)
            print(f'start training sub task(s): {crl_env.current_env_name}, with env indices: {env_indices}')
            print('is replay buffer reset: ',self.replay_buffer.is_empty())
            print(f'reset SAC critics: {is_critics_reset}')
            print(f'reset VAE weights: {is_vae_reset}')
            print(f'reset optimizer: {is_optim_reset}')
            print(f'buffer content: {self.replay_buffer.idx}')
            start_time = time.time()

            # start subtask training
            for step in range(exp_config.num_train_steps):

                # evaluate all envs performance
                if step % exp_config.eval_freq == 0:
                    self.evaluate_vec_env_of_tasks(vec_env=self.envs["eval"], 
                                                step=global_step, episode=subtask)
                    self.logger.dump(global_step)
                
                # Perform logging & Evaluation after every episode  
                if step % self.max_episode_steps == 0:
                    if step > 0:
                        
                        # log avg total success rate
                        success = (success > 0).astype("float")
                        self.logger.log("train/success", success.mean(), global_step)
                        # log individual success rate & reward & index
                        for index, env_index in enumerate(env_indices):
                            self.logger.log(
                                f"train/success_env_index_{int(env_index.detach().cpu().numpy())}",
                                success[index],
                                global_step,
                                )
                            self.logger.log(
                                f"train/episode_reward_env_index_{int(env_index.detach().cpu().numpy())}",
                                episode_reward[index],
                                global_step,
                                )
                            self.logger.log(
                                f"train/env_index_{int(env_index.detach().cpu().numpy())}", 
                                index, 
                                global_step, 
                                tb_log=False
                                )       
                        # log time cost
                        self.logger.log(
                            "train/duration", 
                            time.time() - start_time, 
                            global_step, 
                            tb_log=False)
                        
                        start_time = time.time()
                        self.logger.dump(global_step) # dump to file & console
                    episode += 1
                    global_episode += 1
                    
                    self.logger.log("train/episode", global_episode, global_step, tb_log=False)
                    # reset log metrics
                    episode_reward = np.full(shape=crl_env.current_env_num, fill_value=0.0)
                    success = np.full(shape=crl_env.current_env_num, fill_value=0.0)


                # interactive with the envs
                if step < exp_config.init_steps:
                    # warmup steps
                    action = np.asarray(
                        [self.action_space.sample() for _ in range(crl_env.current_env_num)]
                    )  # (num_envs, action_dim)

                else:
                    with agent_utils.eval_mode(self.agent):
                        train_mode = crl_env.set_train_mode()
                        action = self.agent.sample_action(
                            multitask_obs=crl_obs,
                            modes=[train_mode,],
                        )  # (num_envs, action_dim)

                # run training update (by default each step update 1 time)
                if step >= exp_config.init_steps:
                    num_updates = (
                        exp_config.init_steps if step == exp_config.init_steps else 1
                    )
                    for _ in range(num_updates):
                        self.agent.update(self.replay_buffer, 
                                          self.logger, 
                                          global_step, #step, 
                                          global_step = global_step,
                                          local_step = step,
                                          task_name_to_idx = self.task_name_to_idx,
                                          subtask = subtask,
                                          ) # used for evaluation
                
                # Interaction with envs
                next_crl_obs, reward, done, info = crl_env.step(action)
                if self.should_reset_env_manually:
                    if (episode_step[0] + 1) % self.max_episode_steps == 0:
                        # we do a +2 because we started the counting from 0 and episode_step is incremented after updating the buffer
                        next_crl_obs = crl_env.reset(subtask)
                        env_indices = next_crl_obs['task_obs']

                episode_reward += reward
                success += np.asarray([x["success"] for x in info])

                # allow infinite bootstrap
                # Add transitions into replay buffer
                for index, env_index in enumerate(env_indices):
                    done_bool = (
                        0
                        if episode_step[index] + 1 == self.max_episode_steps
                        else float(done[index])
                    )
                    # add each separate env data into buffer
                    if index not in self.envs_to_exclude_during_training:
                        self.replay_buffer.add(
                            crl_obs["env_obs"][index],
                            action[index],
                            reward[index],
                            next_crl_obs["env_obs"][index],
                            done_bool,
                            task_obs=env_index,
                        )


                crl_obs = next_crl_obs
                episode_step += 1
                global_step += 1
            #######################################################
            ############## end of subtask training ################
            #######################################################

            # final evaluation of each subtask phase
            print('subtask final evaluation')
            self.evaluate_vec_env_of_tasks(vec_env=self.envs["eval"], 
                                        step=global_step, 
                                        episode=subtask, 
                                        record_crl_metrics=True)
            
            if exp_config.save.model:
                self.agent.save(
                    self.model_dir,
                    step=subtask,
                    # retain_last_n=exp_config.save.model.retain_last_n,
                    retain_last_n=-1,
                )
            if exp_config.save.buffer.should_save:
                self.replay_buffer.save(
                    self.buffer_dir,
                    size_per_chunk=exp_config.save.buffer.size_per_chunk,
                    num_samples_to_save=exp_config.save.buffer.num_samples_to_save,
                )
        ####################################################
        ############## end of total training ###############
        ####################################################
        

        if self.config.experiment.save_video:
            print('start recording videos ...')
            self.record_videos(step=global_step)
            print('video recording finished. Check folder:{}'.format(self.video.dir_name))
        ####################################################################
        self.replay_buffer.delete_from_filesystem(self.buffer_dir)
        # self.close_envs()
        self.logger.tb_writer.close()
        self.crl_metrics.to_csv()
        print('====== Training finished ======')





    def collect_trajectory(self, vec_env: VecEnv, num_steps: int) -> None:
        multitask_obs = vec_env.reset()
        env_indices = multitask_obs["task_obs"]
        episode_reward, episode_step, done = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 0, True]
        ]  # (num_envs, 1)

        for _ in range(num_steps):
            with agent_utils.eval_mode(self.agent):
                action = self.agent.sample_action(
                    multitask_obs=multitask_obs, mode="train"
                )  # (num_envs, action_dim)
            next_multitask_obs, reward, done, info = vec_env.step(action)
            if self.should_reset_env_manually:
                if (episode_step[0] + 1) % self.max_episode_steps == 0:
                    # we do a +2 because we started the counting from 0 and episode_step is incremented after updating the buffer
                    next_multitask_obs = vec_env.reset()
            episode_reward += reward

            # allow infinite bootstrap
            for index, env_index in enumerate(env_indices):
                done_bool = (
                    0
                    if episode_step[index] + 1 == self.max_episode_steps
                    else float(done[index])
                )
                self.replay_buffer.add(
                    multitask_obs["env_obs"][index],
                    action[index],
                    reward[index],
                    next_multitask_obs["env_obs"][index],
                    done_bool,
                    env_index=env_index,
                )

            multitask_obs = next_multitask_obs
            episode_step += 1

    def record_videos(self, step, num_eps_per_task_to_record=3):
        """
        # TODO modified to multiple Video recoder in a list
        Record videos of all envs, each env performs one episodes.
        """
        agent = self.agent
        num_record = num_eps_per_task_to_record
        self.task_obs = torch.arange(len(self.list_envs))
        assert self.env_id_to_task_map_recording is not None
        env_names =  list(self.env_id_to_task_map_recording.keys())
        assert len(env_names)==len(self.list_envs)
        
        for env_idx in range(len(env_names)):
            # run for all envs
            print(f'start recording env {env_names[env_idx]} ...')
            # reset
            self.video.reset()
            episode_step = 0           
            env_obs = []
            success = 0.0

            obs = self.list_envs[env_idx].reset()
            env_obs.append(obs)
            multitask_obs = {"env_obs": torch.tensor(env_obs), "task_obs": torch.tensor([env_idx])}
            
            self.video.record(frame=self.list_envs[env_idx].render('rgb_array'))

            for episode_step in range(self.max_episode_steps * num_record):

                # record for env_idx env
                self.video.record(frame=self.list_envs[env_idx].render('rgb_array'))
                
                # agent select action
                with agent_utils.eval_mode(agent):
                    action = agent.select_action(
                        multitask_obs=multitask_obs, modes=["eval"]
                    )
                
                # interactive with envs get new obs
                env_obs = []
                obs,reward,done,info = self.list_envs[env_idx].step(action[0])
                
                if (episode_step+1) % self.max_episode_steps == 0:
                    obs = self.list_envs[env_idx].reset()
                    self.video.record(frame=self.list_envs[env_idx].render('rgb_array'))
                env_obs.append(obs)
                success += info['success']

                multitask_obs = {"env_obs": torch.tensor(env_obs), "task_obs": torch.tensor([env_idx])}
                episode_step += 1

            success = float(success > 0)
            self.video.save(file_name=f'{env_names[env_idx]}_success_{success}_step_{step}')

    def collect_eval_transitions_for_final_evaluation(self):
        '''
        reset & collect some transitions for final evaluation of latent representation learning
        under multitask setting (only used in CRL)
        '''
        print('collect multitask transitions for latent representation learning evaluation.')
        self.replay_buffer.reset()
        vec_env = self.envs['eval']
        agent = self.agent

        for _ in range(20): # each task save 20 episodes
            multitask_obs = vec_env.reset()
            env_indices = multitask_obs['task_obs']
            # multitask_obs['env_obs'].shape (torch.Size([10, 12]))
            # 'task_obs': tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            # episode_step = 0
            # while episode_step < self.max_episode_steps:
            for episode_step in range(self.max_episode_steps):
                with agent_utils.eval_mode(agent):
                    action = agent.select_action(
                        multitask_obs=multitask_obs, modes=["eval"]
                    )
                # action dims MT10: (10, 4)
                next_multitask_obs, reward, done, info = vec_env.step(action)

                for index, env_index in enumerate(env_indices):
                    done_bool = (
                        0
                        if episode_step + 1 == self.max_episode_steps
                        else float(done[index])
                    )
                    self.replay_buffer.add(
                                        multitask_obs["env_obs"][index],
                                        action[index],
                                        reward[index],
                                        next_multitask_obs["env_obs"][index],
                                        done_bool,
                                        task_obs=env_index,
                                    )
                multitask_obs = next_multitask_obs
        print('transitions collection finished.')


        