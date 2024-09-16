# Data availability

we released the raw data that reported in our paper for detailed check. Additionally, we prepared a series of pretrained legion models for reproduce the success results and released their ckpt weights here.

The files are listed below:

### 1. lifelong -- training results for lifelong learning
  - curriculum -- our initial LEGION lifelong learning setup
  - reverse -- train the model in an "hard to easy" mode
  - random -- train the model in a random task ordering
  - few-shot -- training log data for few-shot performance

### 2. mtrl -- multitask based training results
  - latent_samples.npz -- saved inference results in Bayesian non-parametric knowledge space
  - legion_eval_success*.csv -- success rate evaluation records

### 3. weights -- pretrained weights for LEGION lifelong learning
  - 01 -- pretrained weights 01
    - actor.pt -- SAC actor weights
    - task_encoder.pt -- upstream pretrained language embedding module
    - vae.pt -- upstream task inference modules
  - 02 -- pretrained weights 02
  - 03 -- pretrained weights 03
  - legion_eval.ipynb -- scripts for weights loading and evaluation