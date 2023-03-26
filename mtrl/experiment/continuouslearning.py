
import time
from typing import Dict, List, Tuple
import hydra
import numpy as np
import torch

from mtrl.agent import utils as agent_utils
from mtrl.env import builder as env_builder
from mtrl.env.types import EnvType
from mtrl.env.vec_env import VecEnv  # type: ignore
from mtrl.experiment import experiment
from mtrl.utils.types import ConfigType, EnvMetaDataType, EnvsDictType, ListConfigType
from mtrl.utils.utils import pretty_print


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
        # metrics to track: 
        # {'actor_loss', 'alpha_value', 'env_index', 'alpha_loss', 'duration', 'ae_loss', 
        # 'max_rat', 'actor_entropy', 'episode', 'batch_reward', 'critic_loss', 
        # 'success', 'ae_transition_loss', 'actor_target_entropy', 'contrastive_loss', 
        # 'episode_reward', 'step', 'reward_loss'}
        
    
    # from metaworld check what is IO here?
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
        # MT10 {'eval': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
        return eval_modes_to_env_ids

    # from metaworld TODO modify
    def build_envs(self) -> Tuple[EnvsDictType, EnvMetaDataType]:
        
        benchmark = hydra.utils.instantiate(self.config.env.benchmark)

        envs = {}
        
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
        ##########################################
        # build a set of envs for video recording
        if self.config.experiment.save_video:
            list_envs, env_id_to_task_map_recording = env_builder.build_metaworld_env_list_for_eval(
                config=self.config,
                benchmark=benchmark,
                mode="train",
                env_id_to_task_map=env_id_to_task_map,
            )
            self.list_envs = list_envs
            self.env_id_to_task_map_recording = env_id_to_task_map_recording
        ##########################################

        # In MT10 and MT50, the tasks are always sampled in the train mode.
        # For more details, refer https://github.com/rlworkgroup/metaworld

        max_episode_steps = 150
        # hardcoding the steps as different environments return different
        # values for max_path_length. MetaWorld uses 150 as the max length.
        metadata = self.get_env_metadata(
            env=envs["train"],
            max_episode_steps=max_episode_steps,
            ordered_task_list=list(env_id_to_task_map.keys()),
            config=self.config
        )
        return envs, metadata
    
    # Not used
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
    def evaluate_vec_env_of_tasks(self, vec_env: VecEnv, step: int, episode: int):
        """Evaluate the agent's performance on the different environments,
        vectorized as a single instance of vectorized environment.

        Since we are evaluating on multiple tasks, we track additional metadata
        to track which metric corresponds to which task.

        Args:
            vec_env (VecEnv): vectorized environment.
            step (int): step for tracking the training of the agent.
            episode (int): episode for tracking the training of the agent.
        """
        agent = self.agent
        for mode in self.eval_modes_to_env_ids:
            self.logger.log(f"{mode}/episode", episode, step)

        episode_reward, mask, done, success = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 1.0, False, 0.0]
        ]  # (num_envs, 1)
        
        # number of evaluate episodes
        num_eval_episodes = self.config.experiment.num_eval_episodes
        ##################################
        for _ in range(num_eval_episodes):
            multitask_obs = vec_env.reset() 
            # multitask_obs['env_obs'].shape (torch.Size([10, 12]))
            # 'task_obs': tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            episode_step = 0
            # offset = self.config.experiment.num_eval_episodes
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
            #############################################
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
            # every env reward & success rate, num of each env = config.experiment.num_eval_episodes
            for _current_env_index, _current_env_id in enumerate(
                self.eval_modes_to_env_ids[mode] # 'eval': [0,1,2,...9]
            ):
                self.logger.log(
                    f"{mode}/episode_reward_env_index_{_current_env_index}",
                    # episode_reward[
                    #     start_index
                    #     + _current_env_index * offset : start_index
                    #     + (_current_env_index + 1) * offset
                    # ].mean(),
                    episode_reward[
                        start_index
                        + _current_env_index * offset : start_index
                        + (_current_env_index + 1) * offset
                    ].sum() / num_eval_episodes,
                    step,
                )
                self.logger.log(
                    f"{mode}/success_env_index_{_current_env_index}",
                    # success[
                    #     start_index
                    #     + _current_env_index * offset : start_index
                    #     + (_current_env_index + 1) * offset
                    # ].mean(),
                    success[
                        start_index
                        + _current_env_index * offset : start_index
                        + (_current_env_index + 1) * offset
                    ].sum()/num_eval_episodes,
                    step,
                )
                self.logger.log(
                    f"{mode}/env_index_{_current_env_index}", _current_env_id, step, tb_log=False
                )
            start_index += offset * num_envs
        self.logger.dump(step)

    def run(self):
        '''
        env_metadata, ordered_task_list ['reach-v1', 'push-v1', 
                                        'pick-place-v1', 'door-open-v1', 
                                        'drawer-open-v1', 'drawer-close-v1', 
                                        'button-press-topdown-v1', 'peg-insert-side-v1', 
                                        'window-open-v1', 'window-close-v1']
        task_name_to_idx_map:  {'reach-v1': 0, 'push-v1': 1, 
                                'pick-place-v1': 2, 'door-open-v1': 3, 
                                'drawer-open-v1': 4, 'drawer-close-v1': 5, 
                                'button-press-topdown-v1': 6, 'peg-insert-side-v1': 7, 
                                'window-open-v1': 8, 'window-close-v1': 9}
        env_indices:  tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) <class 'torch.Tensor'>
        '''
        # print('env_metadata, ordered_task_list', self.env_metadata['ordered_task_list'])
        # print('task_name_to_idx_map: ', self.task_name_to_idx)
        # print('agent task encoder: ', self.agent.task_encoder)
        # print('agent encoder: ',self.agent.encoder)
        # print('encoder params: ', self.agent.encoder.parameters())
        # print('agent actor: ', self.agent.actor)
        # print('agent critic: ', self.agent.critic)
        # print('agent: log_alpha: ', self.agent.log_alpha)

        # vec_env = self.envs["train"]
        # multitask_obs = vec_env.reset()
        # print('task obs: \n', multitask_obs['env_obs'])
        # print('task obs: \n', multitask_obs['task_obs'])
        
        self.run_multitask()
    


    
    def run_multitask(self) -> None:
        
        """Run the experiment."""
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
        
        # train loop
        for step in range(self.start_step, exp_config.num_train_steps):

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

                # evaluate agent periodically
                if step % exp_config.eval_freq == 0:
                    self.evaluate_vec_env_of_tasks(
                        vec_env=self.envs["eval"], step=step, episode=episode
                    )
                    if exp_config.save.model:
                        self.agent.save(
                            self.model_dir,
                            step=step,
                            retain_last_n=exp_config.save.model.retain_last_n,
                        )
                    if exp_config.save.buffer.should_save:
                        self.replay_buffer.save(
                            self.buffer_dir,
                            size_per_chunk=exp_config.save.buffer.size_per_chunk,
                            num_samples_to_save=exp_config.save.buffer.num_samples_to_save,
                        )
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
            self.record_videos()
            print('video recording finished. Check folder:{}'.format(self.video.dir_name))
        ####################################################################
        
        self.replay_buffer.delete_from_filesystem(self.buffer_dir)
        self.close_envs()
        print('Training finished')
    

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


    def record_videos(self):
            """
            # TODO modified to multiple Video recoder in a list
            Record videos of all envs, each env performs one episodes.
            """
            agent = self.agent
            self.task_obs = torch.arange(len(self.list_envs))
            assert self.env_id_to_task_map_recording is not None
            env_names =  list(self.env_id_to_task_map_recording.keys())
            assert len(env_names)==len(self.list_envs)
            
            for env_idx in range(len(env_names)):
                # run for all envs
                print('start recording env {} ...'.format(env_names[env_idx]))
                self.video.init()
                episode_step = 0
                env_obs = []
                success = 0.0
                
                for i in range(len(env_names)):
                    obs = self.list_envs[i].reset()  # (num_envs, 9, 84, 84)
                    env_obs.append(obs)
                multitask_obs = {"env_obs": torch.tensor(env_obs), "task_obs": self.task_obs}
                

                while episode_step < self.max_episode_steps:
                    # record for env_idx env
                    self.video.record(frame=None, env=self.list_envs[env_idx])
                    
                    # agent select action
                    with agent_utils.eval_mode(agent):
                        action = agent.select_action(
                            multitask_obs=multitask_obs, modes=["eval"]
                        )
                    
                    # interactive with envs get new obs
                    env_obs = []
                    for i in range(len(env_names)):
                        obs, reward, done, info = self.list_envs[i].step(action[i])
                        env_obs.append(obs)
                        if i == env_idx:
                            success += info['success']
                    multitask_obs = {"env_obs": torch.tensor(env_obs), "task_obs": self.task_obs}
                    episode_step += 1

                success = float(success > 0)
                self.video.save(file_name='{}_success_{}'.format(env_names[env_idx], success))
            