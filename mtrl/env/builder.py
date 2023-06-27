# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import mtenv
from gym.vector.async_vector_env import AsyncVectorEnv

from mtrl.env.vec_env import MetaWorldVecEnv, VecEnv
from mtrl.utils.types import ConfigType


def build_dmcontrol_vec_env(
    domain_name: str,
    task_name: str,
    prefix: str,
    make_kwargs: ConfigType,
    env_id_list: List[int],
    seed_list: List[int],
    mode_list: List[str],
) -> VecEnv:
    def get_func_to_make_envs(seed: int, initial_task_state: int):
        def _func() -> mtenv.MTEnv:
            kwargs = deepcopy(make_kwargs)
            kwargs["seed"] += seed
            kwargs["initial_task_state"] = initial_task_state
            return mtenv.make(
                f"MT-HiPBMDP-{domain_name.capitalize()}-{task_name.capitalize()}-vary-{prefix.replace('_', '-')}-v0",
                **kwargs,
            )

        return _func

    funcs_to_make_envs = [
        get_func_to_make_envs(seed=seed, initial_task_state=task_state)
        for (seed, task_state) in zip(seed_list, env_id_list)
    ]

    env_metadata = {"ids": env_id_list, "mode": mode_list}

    env = VecEnv(env_metadata=env_metadata, env_fns=funcs_to_make_envs, context="spawn")

    return env


def build_metaworld_vec_env(
    config: ConfigType,
    benchmark: "metaworld.Benchmark",  # type: ignore[name-defined] # noqa: F821
    mode: str,
    env_id_to_task_map: Optional[Dict[str, "metaworld.Task"]],  # type: ignore[name-defined] # noqa: F821
) -> Tuple[AsyncVectorEnv, Optional[Dict[str, Any]]]:
    from mtenv.envs.metaworld.env import (
        get_list_of_func_to_make_envs as get_list_of_func_to_make_metaworld_envs,
    )
    benchmark_name = config.env.benchmark._target_.replace("metaworld.", "")
    num_tasks = int(benchmark_name.replace("MT", "")) # 1 or 10 or 50 ...
    make_kwargs = {
        "benchmark": benchmark,
        "benchmark_name": benchmark_name, # MT10, MT50
        "env_id_to_task_map": env_id_to_task_map,
        "num_copies_per_env": 1,
        "should_perform_reward_normalization": True,
    }

    funcs_to_make_envs, env_id_to_task_map = get_list_of_func_to_make_metaworld_envs(
        **make_kwargs
    )
    try:
        env_version = benchmark.env_version
    except:
        env_version = 0
    env_metadata = {
        "ids": list(range(num_tasks)), # env indices from 0 to end
        "mode": [mode for _ in range(num_tasks)], # train in MT setting
        "env_version": env_version
    }
    env = MetaWorldVecEnv(
        env_metadata=env_metadata,
        env_fns=funcs_to_make_envs,
        context="spawn",
        shared_memory=False,
        use_onehot=config.env.use_onehot
    )
    return env, env_id_to_task_map

#####################################################################################
############## Return a list of created envs for eval & video recording #############
#####################################################################################
def build_metaworld_env_list(
    config: ConfigType,
    benchmark: "metaworld.Benchmark",  # type: ignore[name-defined] # noqa: F821
    mode: str,
    env_id_to_task_map: Optional[Dict[str, "metaworld.Task"]],  # type: ignore[name-defined] # noqa: F821
) -> Tuple[AsyncVectorEnv, Optional[Dict[str, Any]]]:
    '''
    Build a list of envs for video recording
    '''
    from mtenv.envs.metaworld.env import get_list_of_envs
    benchmark_name = config.env.benchmark._target_.replace("metaworld.", "")
    num_tasks = int(benchmark_name.replace("MT", "")) # 1 or 10 or 50 ...
    make_kwargs = {
        "benchmark": benchmark,
        "benchmark_name": benchmark_name, # MT10, MT50
        "env_id_to_task_map": env_id_to_task_map,
        "num_copies_per_env": 1,
        "should_perform_reward_normalization": True,
    }

    envs_list, env_id_to_task_map = get_list_of_envs(**make_kwargs)
    env_metadata = {
        "ids": list(range(num_tasks)), # env indices from 0 to end
        "mode": [mode for _ in range(num_tasks)], # train in MT setting
    }
    
    
    return envs_list, env_id_to_task_map


