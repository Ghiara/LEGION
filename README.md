[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/facebookresearch/mtrl/blob/main/LICENSE)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Zulip Chat](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://mtenv.zulipchat.com)
[![DOI](https://zenodo.org/badge/619254723.svg)](https://doi.org/10.5281/zenodo.14265088)

# Preserving and Combining Knowledge in Robotic Lifelong Reinforcement Learning

LEGION: A Language Embedding based Generative Incremental Off-policy Reinforcement Learning Framework with Non-parametric Bayes


## Repository Agenda

1. [Introduction](#Introduction)

2. [Installation & Setup](#Installation)

3. [Training and Evaluation](#Train)

4. [Repository Structure](#FileStructure)

5. [Data Availability](#Data)

6. [Acknowledgements](#Acknowledgements)

## Introduction

Humans can continually accumulate knowledge and develop increasingly complex behaviors and skills throughout their lives, which is a capability known as `lifelong learning`. 
Although this lifelong learning capability is considered an essential mechanism that makes up generalized intelligence, recent advancements in artificial intelligence predominantly excel in narrow, specialized domains and generally lack of this lifelong learning capability.
Our study introduces a robotic lifelong reinforcement learning framework that addresses this gap by incorporating a non-parametric Bayesian model into the knowledge space.
Additionally, we enhance the agent's semantic understanding of tasks by integrating language embeddings into the framework.
Our proposed embodied agent can consistently accumulate knowledge from a continuous stream of one-time feeding tasks. 
Furthermore, our agent can tackle challenging real-world long-horizon tasks by combining and reapplying its acquired knowledge from the original tasks stream.
Our findings demonstrate that intelligent embodied agents can exhibit a capability for lifelong learning similar to that of human beings.
The proposed framework advances our understanding of the robotic lifelong learning process and may inspire the development of more broadly applicable intelligence.

### LEGION long horizon task demonstration
[![Movie1](/docs/static/images/movie_cover.png "Long horzion task demonstration")](https://www.cit.tum.de/cit/startseite/)
### LEGION Framework for Training
- Training: Our framework receives language semantic information and environment observations 
          as input to make policy decisions and output action patterns, it trains on only one task at a time. L
          represents the loss functions and is explained in the Method section `Upstream task inference`.
![train](/docs/static/images/framework_train.png "Framework for Lifelong reinforcement Learning")
### LEGION Framework for Deployment
- Deployment: In the real-world demonstration, the agent parameters remain frozen, the agent 
          receives input signal from real-world hardware and outputs corresponding action signals, both `sim2real` and `real2sim` 
          modules process the data to align the gap between the simulation and real world.
![deployment](/docs/static/images/framework_deployment.png "Deployment")

## Installation

To install the repository, follow the steps below:

* Clone this repository: `git clone https://github.com/Ghiara/LEGION.git`.

* Follow the [INSTALL.md](scripts/INSTALL.md) to install the repository and dependencies.

* Note: `mujoco200` with `mujoco-py==2.0.2.8`, `gym=0.20.0`, `protobuf==3.20.0`, `cython<3` works with this project, you can first manually install the denpendices before you install the Metaworld environment.

## Train

To reproduce the results we present in the paper, we provide a [TRAIN_EVAL.md](scripts/TRAIN_EVAL.md) that record the training cli we used. To run the code, please follow our guidelines in the [TRAIN_EVAL.md](scripts/TRAIN_EVAL.md).


## FileStructure

In summary, we use `Hydra` to manage the training process. 
- The configs for all instances can be found under `config` folder. 
- The agent implementation can be found under `mtrl/agent` folder.
- The enviroments can be found at `mtrl/env`.
- The training script is implemented at `mtrl/experiment`.


The detailed structure of this project is shown as follows:

```
LEGION
    |- config                               -- config files folder
    |- metadata                             -- language embedding folder
    |- mtrl                                 -- implementation of our agent
        |- agent
            |- components
                |- actor.py                 -- downstream SAC actor
                |- bnp_model.py             -- Bayesian nonparametric model
                |- critic.py                -- downstream SAC critic
                |- decoder.py               -- upstream decoder for dynamic/semantic rebuild
                |- encoder.py               -- upstream encoder task inference
                |- task_encoder.py          -- upstream encoder for language processing
            ...
            |- sac_dpmm.py                  -- our LEGION agent implementation
            ...
        |- env                              -- environment builder utils
        |- experiment
            ...
            |- continuouslearning.py        -- our implementation of training script
            ...           
    |- scripts
        |- INSTALL.md                       -- package installation guidelines
        |- TRAIN_EVAL.md                    -- training & evaluation clis
    |- source (after followed INSTALL.md)
        |- bnpy                             -- third party Bayesian non-parametric library
        |- Metaworld-KUKA-IIWA-R800         -- third party metaworld environment
        |- mtenv                            -- third party environment manager library
    main.py                                 -- main entry of repository
    README.md                               -- this file
```


## Data

The original training and evaluation data that we presented in the paper are avaiable at [here](data/).


## Acknowledgements

* Project file pre-commit, mypy config, towncrier config, circleci etc are based on same files from [Hydra](https://github.com/facebookresearch/hydra).

* Implementation Inherited from [MTRL](https://mtrl.readthedocs.io/en/latest/index.html) library. 

* Documentation of MTRL repository refer to: [https://mtrl.readthedocs.io](https://mtrl.readthedocs.io).