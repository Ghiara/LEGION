### Data availability

we released the raw data that reported in our paper for detailed check. Additionally, we prepared a series of pretrained legion models for reproduce the success results and released their ckpt weights here.

The files are listed below:

- lifelong -- training results for lifelong learning
  - curriculum -- our initial LEGION lifelong learning setup
  - reverse -- train the model in an "hard to easy" mode
  - random -- train the model in a random task ordering
  - few-shot -- training log data for few-shot performance

- mtrl -- multitask based training results

- weights -- best weights for LEGION lifelong learning
  - legion_eval.ipynb -- scripts for weights loading and evaluation