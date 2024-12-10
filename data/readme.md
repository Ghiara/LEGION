# Data availability

we released the raw data that reported in our paper for detailed check. Additionally, we prepared a series of pretrained legion models for reproduce the success results and released their ckpt weights here.

The files are listed below:

### 1. lifelong -- training results for lifelong learning
  - curriculum -- train the model in a curriculum based task ordering
  - reverse -- train the model in a reverse task ordering
  - random -- train the model in a random task ordering
  - few-shot -- training log data for few-shot performance
  - hetero_arms -- renderings of heterogenous embodiments from Sawyer and KUKA
    - default -- renderings from default task enviroments
    - cross_validation -- renderings using switched embodiment

### 2. mtrl -- multitask based training results
  - latent_samples.npz -- saved inference results in Bayesian non-parametric knowledge space
  - legion_eval_success*.csv -- success rate evaluation records

### 3. weights -- pretrained weights for LEGION lifelong learning
  - ckpt -- pretrained weights for evaluation
    - actor.pt -- SAC actor weights
    - task_encoder.pt -- upstream pretrained language embedding module
    - vae.pt -- upstream task inference modules
  - legion_eval.ipynb -- scripts for weights loading and evaluation