[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/facebookresearch/mtrl/blob/main/LICENSE)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Zulip Chat](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://mtenv.zulipchat.com)
[![DOI](https://zenodo.org/badge/619254723.svg)](https://doi.org/10.5281/zenodo.14265088)

# Preserving and Combining Knowledge in Robotic Lifelong Reinforcement Learning

<div align="center">

Official implementation of LEGION: A Language Embedding based Generative Incremental Off-policy Reinforcement Learning Framework with Non-parametric Bayes


#### [[Project Website]](https://ghiara.github.io/LEGION/) [[Paper]](https://www.nature.com/articles/s42256-025-00983-2)


[Yuan Meng](https://github.com/Ghiara)<sup>1,\*</sup>, [Zhenshan Bing](https://github.com/zhenshan-bing)<sup>1,2,\*,&dagger;</sup>, [Xiangtong Yao](https://www.ce.cit.tum.de/air/people/xiangtong-yao/)<sup>1,\*</sup>, [Kejia Chen](https://kifabrik.mirmi.tum.de/team/)<sup>1</sup>,

[Kai Huang](https://cse.sysu.edu.cn/content/2466)<sup>3,&dagger;</sup>, [Yang Gao](https://is.nju.edu.cn/gy_en/main.htm)<sup>2,&dagger;</sup>, [Fuchun Sun](https://www.cs.tsinghua.edu.cn/csen/info/1312/4393.htm)<sup>4,&dagger;</sup>, [Alois Knoll](https://www.ce.cit.tum.de/air/people/prof-dr-ing-habil-alois-knoll/)<sup>1</sup>.

</div>

<p align="center">
<small><sup>1</sup>School of Computation, Information and Technology, Technical University of Munich, Germany</small>
<br><small><sup>2</sup>State Key Laboratory for Novel Software Technology, Nanjing University, China</small>
<br><small><sup>3</sup>Key Laboratory of Machine Intelligence and Advanced Computing, Sun Yat-sen University, China</small>
<br><small><sup>4</sup>Department of Computer Science and Technology, Tsinghua University, China</small>
<small><br><sup>*</sup>Indicates Equal Contribution</small>
<small><br><sup>&dagger;</sup>To whom correspondence should be addressed; E-mail: zhenshan.bing@tum.de, huangk36@mail.sysu.edu.cn, gaoy@nju.edu.cn, fcsun@tsinghua.edu.cn</small>
</p>

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
The proposed framework advances our understanding of the robotic lifelong learning process and may inspire the development of more broadly applicable intelligence.

### LEGION long horizon task demonstration

<!-- [![Movie1](/docs/static/images/movie_cover.png "Long horzion task demonstration")](https://www.cit.tum.de/cit/startseite/) -->
[![Movie1](/docs/static/images/movie_cover.png "Long horizon task demonstration")](https://assets-eu.researchsquare.com/files/rs-4353532/v1/49b2a9646c62385bb89ddc0e.mp4)


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

To set up the repository, follow the steps below:

* Clone the repository: `git clone https://github.com/Ghiara/LEGION.git`.

* Please refer to [INSTALL.md](scripts/INSTALL.md) for detailed installation steps and environment setup.

> [!TIP] 
> This project works best with the following versions: `mujoco200`, `mujoco-py==2.0.2.8`, `gym=0.20.0`, `protobuf==3.20.0`, `cython<3`. It's recommended to install these dependencies manually **before** installing the MetaWorld environment to avoid compatibility issues.

## Train

To reproduce the results reported in our paper, we provide a separate file containing the exact training command lines used during our experiments.
> [!IMPORTANT]
> Please follow the instructions in [TRAIN_EVAL.md](scripts/TRAIN_EVAL.md) to run the code properly.



## FileStructure

We use `Hydra` to manage the training process. 
- The configs for all instances can be found under `config` folder. 
- The agent implementation can be found under `mtrl/agent` folder.
- The enviroments can be found at `mtrl/env`.
- The training script is implemented at `mtrl/experiment`.


The detailed structure of this project is shown as follows:

```bash
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


## Citation

To cite this article:

```bibtex
@article{meng2025preserving,
  title={Preserving and combining knowledge in robotic lifelong reinforcement learning},
  author={Meng, Yuan and Bing, Zhenshan and Yao, Xiangtong and Chen, Kejia and Huang, Kai and Gao, Yang and Sun, Fuchun and Knoll, Alois},
  journal={Nature Machine Intelligence},
  pages={1--14},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```