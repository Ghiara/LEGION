[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/facebookresearch/mtrl/blob/main/LICENSE)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Zulip Chat](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://mtenv.zulipchat.com)

# LEGION: A Language Embedding based Generative Incremental Off-policy Reinforcement Learning Framework with Non-parametric Bayes

Implementation Inherited from [MTRL](https://mtrl.readthedocs.io/en/latest/index.html) library 

## Contents


1. [Setup](#Setup)

2. [Train](#Train)

3. [Usage](#Usage)

4. [Documentation](#Documentation)

5. [Contributing to MTRL](#Contributing-to-MTRL)

6. [Community](#Community)

7. [Acknowledgements](#Acknowledgements)


## Setup

* Clone the repository: `git clone https://github.com/Ghiara/LEGION.git`.

* Install dependencies: `pip install -r requirements/dev.txt`
  
* Install modified Meta-World benchmark `git clone https://github.com/Ghiara/metaworld.git`
  
* Install bnpy library `git clone https://github.com/bnpy/bnpy.git` + `cd bnpy/`+ `pip install -e .`
  
* Note that we use mtenv to manage Meta-World environment, and add slight modification under mtenv/envs/metaworld/env.py, we added following function allowing for output a list of env instances:
```
def get_list_of_envs(
    benchmark: Optional[metaworld.Benchmark],
    benchmark_name: str,
    env_id_to_task_map: Optional[EnvIdToTaskMapType],
    should_perform_reward_normalization: bool = True,
    task_name: str = "pick-place-v1",
    num_copies_per_env: int = 1,
) -> Tuple[List[Any], Dict[str, Any]]:

    if not benchmark:
        if benchmark_name == "MT1":
            benchmark = metaworld.ML1(task_name)
        elif benchmark_name == "MT10":
            benchmark = metaworld.MT10()
        elif benchmark_name == "MT50":
            benchmark = metaworld.MT50()
        else:
            raise ValueError(f"benchmark_name={benchmark_name} is not valid.")

    env_id_list = list(benchmark.train_classes.keys())

    def _get_class_items(current_benchmark):
        return current_benchmark.train_classes.items()

    def _get_tasks(current_benchmark):
        return current_benchmark.train_tasks

    def _get_env_id_to_task_map() -> EnvIdToTaskMapType:
        env_id_to_task_map: EnvIdToTaskMapType = {}
        current_benchmark = benchmark
        for env_id in env_id_list:
            for name, _ in _get_class_items(current_benchmark):
                if name == env_id:
                    task = random.choice(
                        [
                            task
                            for task in _get_tasks(current_benchmark)
                            if task.env_name == name
                        ]
                    )
                    env_id_to_task_map[env_id] = task
        return env_id_to_task_map

    if env_id_to_task_map is None:
        env_id_to_task_map: EnvIdToTaskMapType = _get_env_id_to_task_map()  # type: ignore[no-redef]
    assert env_id_to_task_map is not None

    def make_envs_use_id(env_id: str):
        current_benchmark = benchmark
        
        
        def _make_env():
            for name, env_cls in _get_class_items(current_benchmark):
                if name == env_id:
                    env = env_cls()
                    task = env_id_to_task_map[env_id]
                    env.set_task(task)
                    if should_perform_reward_normalization:
                        env = NormalizedEnvWrapper(env, normalize_reward=True)
                    return env
        # modified return built single envs
        single_env = _make_env()
        return single_env

    if num_copies_per_env > 1:
        env_id_list = [
            [env_id for _ in range(num_copies_per_env)] for env_id in env_id_list
        ]
        env_id_list = [
            env_id for env_id_sublist in env_id_list for env_id in env_id_sublist
        ]

    list_of_envs = [make_envs_use_id(env_id) for env_id in env_id_list]
    return list_of_envs, env_id_to_task_map
```

## Train

To run the LEGION under multi-task setting
```
python3 -u main.py \
setup=continuouslearning \
env=metaworld-mt10 \
env.use_kuka_env=False \
env.use_onehot=False \
agent=sac_dpmm \
agent.encoder.type_to_select=vae \
agent.encoder.vae.should_reconstruct=True \
agent.encoder.vae.latent_dim=10 \
agent.multitask.should_use_task_encoder=True \
agent.multitask.should_use_disentangled_alpha=True \
agent.multitask.encoder_input_setup=context_obs \
agent.multitask.dpmm_cfg.dpmm_update_start_step=6000 \
agent.multitask.dpmm_cfg.dpmm_update_freq=50000 \
agent.multitask.dpmm_cfg.kl_div_update_freq=30 \
agent.multitask.dpmm_cfg.beta_kl_z=0.002 \
agent.multitask.dpmm_cfg.sF=0.00001 \
agent.multitask.num_envs=10 \
experiment.training_mode=multitask \
experiment.eval_freq=7500 \
experiment.num_eval_episodes=10 \
experiment.num_train_steps=1000000 \
experiment.save_video=False \
setup.seed=1 \
setup.device=cuda:0 \
replay_buffer.batch_size=1280 \
replay_buffer.dpmm_batch_size=3000 
```

To run the LEGION under CRL setting
```
python3 -u main.py \
setup=continuouslearning \
env=metaworld-mt10 \
env.use_onehot=False \
env.use_kuka_env=False \
agent=sac_dpmm \
agent.encoder.type_to_select=vae \
agent.encoder.vae.should_reconstruct=True \
agent.multitask.should_use_task_encoder=True \
agent.multitask.should_use_disentangled_alpha=True \
agent.multitask.encoder_input_setup=context_obs \
agent.multitask.dpmm_cfg.dpmm_update_start_step=10000 \
agent.multitask.dpmm_cfg.dpmm_update_freq=100000 \
agent.multitask.dpmm_cfg.kl_div_update_freq=50 \
agent.multitask.dpmm_cfg.sF=0.00001 \
agent.multitask.dpmm_cfg.beta_kl_z=0.001 \
experiment.training_mode=crl_queue \
experiment.should_reset_optimizer=True \
experiment.should_reset_replay_buffer=False \
experiment.should_reset_critics=False \
experiment.should_reset_vae=False \
experiment.eval_freq=7500 \
experiment.num_eval_episodes=10 \
experiment.num_train_steps=1000000 \
agent.multitask.num_envs=10 \
experiment.save_video=True \
setup.seed=1 \
replay_buffer.batch_size=512 \
replay_buffer.capacity=10000000 \
replay_buffer.dpmm_batch_size=3000 \
replay_buffer.rehearsal.should_use=False
```

To run the Meta-World (KUKA), add following command (Only valid for MT10_KUKA)
```
env.use_kuka_env=True
```


### Reference:

* [Multi-Task Reinforcement Learning with Context-based Representations](https://arxiv.org/abs/2102.06177)
    *  We use the `af8417bfc82a3e249b4b02156518d775f29eb289` commit for the MetaWorld environments for our experiments.


## Usage

* MTRL supports 8 different multi-task RL algorithms as described [here](https://mtrl.readthedocs.io/en/latest/pages/tutorials/overview.html).

* MTRL supports multi-task environments using [MTEnv](https://github.com/facebookresearch/mtenv). These environments include [MetaWorld](https://meta-world.github.io/)


## Documentation

[https://mtrl.readthedocs.io](https://mtrl.readthedocs.io)


## Acknowledgements

* Project file pre-commit, mypy config, towncrier config, circleci etc are based on same files from [Hydra](https://github.com/facebookresearch/hydra).
