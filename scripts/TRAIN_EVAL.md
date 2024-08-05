# Train and Evaluation
This file provide the CLI we used to produce the data that present in the paper

- To run the Meta-World with KUKA manipulation, add following command (Only valid for MT10_KUKA)
```
env.use_kuka_env=True
```

- To run the LEGION under lifelong learning setting
```
python3 -u main.py \
setup=continuouslearning \
env=metaworld-mt10 \
env.use_onehot=False \
env.use_kuka_env=True \
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
experiment.eval_freq=10000 \
experiment.num_eval_episodes=10 \
experiment.num_train_steps=1000000 \
agent.multitask.num_envs=10 \
experiment.save_video=False \
setup.seed=0 \
replay_buffer.batch_size=512 \
replay_buffer.dpmm_batch_size=3000 \
```

```
python3 -u main.py \
setup=continuouslearning \
env=metaworld-mt10 \
env.use_onehot=False \
env.use_kuka_env=True \
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
experiment.eval_freq=10000 \
experiment.num_eval_episodes=10 \
experiment.num_train_steps=1000000 \
agent.multitask.num_envs=10 \
experiment.save_video=False \
setup.seed=1 \
replay_buffer.batch_size=512 \
replay_buffer.dpmm_batch_size=3000 \
```

```
python3 -u main.py \
setup=continuouslearning \
env=metaworld-mt10 \
env.use_onehot=False \
env.use_kuka_env=True \
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
experiment.eval_freq=10000 \
experiment.num_eval_episodes=10 \
experiment.num_train_steps=1000000 \
agent.multitask.num_envs=10 \
experiment.save_video=False \
setup.seed=2 \
replay_buffer.batch_size=512 \
replay_buffer.dpmm_batch_size=3000 \
```

```
python3 -u main.py \
setup=continuouslearning \
env=metaworld-mt10 \
env.use_onehot=False \
env.use_kuka_env=True \
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
experiment.eval_freq=10000 \
experiment.num_eval_episodes=10 \
experiment.num_train_steps=1000000 \
agent.multitask.num_envs=10 \
experiment.save_video=False \
setup.seed=3 \
replay_buffer.batch_size=512 \
replay_buffer.dpmm_batch_size=3000 \
```

```
python3 -u main.py \
setup=continuouslearning \
env=metaworld-mt10 \
env.use_onehot=False \
env.use_kuka_env=True \
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
experiment.eval_freq=10000 \
experiment.num_eval_episodes=10 \
experiment.num_train_steps=1000000 \
agent.multitask.num_envs=10 \
experiment.save_video=False \
setup.seed=4 \
replay_buffer.batch_size=512 \
replay_buffer.dpmm_batch_size=3000 \
```

```
python3 -u main.py \
setup=continuouslearning \
env=metaworld-mt10 \
env.use_onehot=False \
env.use_kuka_env=True \
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
experiment.eval_freq=10000 \
experiment.num_eval_episodes=10 \
experiment.num_train_steps=1000000 \
agent.multitask.num_envs=10 \
experiment.save_video=False \
setup.seed=5 \
replay_buffer.batch_size=512 \
replay_buffer.dpmm_batch_size=3000 \
```

```
python3 -u main.py \
setup=continuouslearning \
env=metaworld-mt10 \
env.use_onehot=False \
env.use_kuka_env=True \
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
experiment.eval_freq=10000 \
experiment.num_eval_episodes=10 \
experiment.num_train_steps=1000000 \
agent.multitask.num_envs=10 \
experiment.save_video=False \
setup.seed=6 \
replay_buffer.batch_size=512 \
replay_buffer.dpmm_batch_size=3000 \
```

```
python3 -u main.py \
setup=continuouslearning \
env=metaworld-mt10 \
env.use_onehot=False \
env.use_kuka_env=True \
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
experiment.eval_freq=10000 \
experiment.num_eval_episodes=10 \
experiment.num_train_steps=1000000 \
agent.multitask.num_envs=10 \
experiment.save_video=False \
setup.seed=7 \
replay_buffer.batch_size=512 \
replay_buffer.dpmm_batch_size=3000 \
```

- To run the LEGION under multi-task setting
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
experiment.eval_freq=10000 \
experiment.num_eval_episodes=10 \
experiment.num_train_steps=1000000 \
experiment.save_video=False \
setup.seed=0 \
setup.device=cuda:0 \
replay_buffer.batch_size=1280 \
replay_buffer.dpmm_batch_size=3000 
```

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
experiment.eval_freq=10000 \
experiment.num_eval_episodes=10 \
experiment.num_train_steps=1000000 \
experiment.save_video=False \
setup.seed=1 \
setup.device=cuda:0 \
replay_buffer.batch_size=1280 \
replay_buffer.dpmm_batch_size=3000 
```

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
experiment.eval_freq=10000 \
experiment.num_eval_episodes=10 \
experiment.num_train_steps=1000000 \
experiment.save_video=False \
setup.seed=2 \
setup.device=cuda:0 \
replay_buffer.batch_size=1280 \
replay_buffer.dpmm_batch_size=3000 
```

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
experiment.eval_freq=10000 \
experiment.num_eval_episodes=10 \
experiment.num_train_steps=1000000 \
experiment.save_video=False \
setup.seed=3 \
setup.device=cuda:0 \
replay_buffer.batch_size=1280 \
replay_buffer.dpmm_batch_size=3000 
```

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
experiment.eval_freq=10000 \
experiment.num_eval_episodes=10 \
experiment.num_train_steps=1000000 \
experiment.save_video=False \
setup.seed=4 \
setup.device=cuda:0 \
replay_buffer.batch_size=1280 \
replay_buffer.dpmm_batch_size=3000 
```

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
experiment.eval_freq=10000 \
experiment.num_eval_episodes=10 \
experiment.num_train_steps=1000000 \
experiment.save_video=False \
setup.seed=5 \
setup.device=cuda:0 \
replay_buffer.batch_size=1280 \
replay_buffer.dpmm_batch_size=3000 
```

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
experiment.eval_freq=10000 \
experiment.num_eval_episodes=10 \
experiment.num_train_steps=1000000 \
experiment.save_video=False \
setup.seed=6 \
setup.device=cuda:0 \
replay_buffer.batch_size=1280 \
replay_buffer.dpmm_batch_size=3000 
```

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
experiment.eval_freq=10000 \
experiment.num_eval_episodes=10 \
experiment.num_train_steps=1000000 \
experiment.save_video=False \
setup.seed=7 \
setup.device=cuda:0 \
replay_buffer.batch_size=1280 \
replay_buffer.dpmm_batch_size=3000 
```

## Evaluation

All training and evaluation data are saved under `logs/YYYY-MM-DD HH:MM:SS`.

Under each data log folder, you will see:

```
dpmm_model                      -- saved DPMM model and latent variable data
model                           -- saved LEGION ckpts
tb_logger_MM-DD_HH-MM           -- tensorboard logger
video                           -- saved video visualization 
config.json                     -- config setup file
eval.log                        -- log data for evaluation
log.jsonl                       -- log hyperparameters
train.log                       -- log data for training
```




## Baselines

We use `MTRL` and `Continual World` for baseline comparison.

* MTRL supports 8 different multi-task RL algorithms as described [here](https://mtrl.readthedocs.io/en/latest/pages/tutorials/overview.html).

* Continual world supports various continual learning methods inpired from foundation machine learning domain, algorithm described [here] (https://sites.google.com/view/continualworld)
