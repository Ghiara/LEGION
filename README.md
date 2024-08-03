[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/facebookresearch/mtrl/blob/main/LICENSE)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Zulip Chat](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://mtenv.zulipchat.com)

# LEGION: A Language Embedding based Generative Incremental Off-policy Reinforcement Learning Framework with Non-parametric Bayes


## Contents

1. [Introduction](#Introduction)

2. [Installation & Setup](#Installation)

3. [Training and Evaluation](#Train)

4. [Repository Structure](#FileStructure)

5. [Acknowledgements](#Acknowledgements)

## Introduction

Humans can continually accumulate knowledge and develop increasingly complex behaviors and skills throughout their lives, which is a capability known as ``lifelong learning''. 
Although this lifelong learning capability is considered an essential mechanism that makes up generalized intelligence, recent advancements in artificial intelligence predominantly excel in narrow, specialized domains and generally lack of this lifelong learning capability.
Our study introduces a robotic lifelong reinforcement learning framework that addresses this gap by incorporating a non-parametric Bayesian model into the knowledge space.
Additionally, we enhance the agent's semantic understanding of tasks by integrating language embeddings into the framework.
Our proposed embodied agent can consistently accumulate knowledge from a continuous stream of one-time feeding tasks. 
Furthermore, our agent can tackle challenging real-world long-horizon tasks by combining and reapplying its acquired knowledge from the original tasks stream.
Our findings demonstrate that intelligent embodied agents can exhibit a capability for lifelong learning similar to that of human beings.
The proposed framework advances our understanding of the robotic lifelong learning process and may inspire the development of more broadly applicable intelligence.

### LEGION long horizon task demonstration
[![Movie1](/imgs/movie_cover.png "Long horzion task demonstration")](https://www.cit.tum.de/cit/startseite/)
### LEGION Framework for Training
![train](/imgs/framework_train.png "Framework for Lifelong reinforcement Learning")
### LEGION Framework for Deployment
![deployment](/imgs/framework_deployment.png "Deployment")

## Installation

To install the repository, follow the steps below:

* Clone this repository: `git clone https://github.com/Ghiara/LEGION.git`.

* Follow the [INSTALL.md](INSTALL.md) to install the repository and dependencies.

* Note: `mujoco200` with `mujoco-py==2.0.2.8`, `gym=0.20.0`, `protobuf==3.20.0`, `cython<3` works with this project, you can first manually install the denpendices before you install the Metaworld environment.

## FileStructure




## Acknowledgements

* Project file pre-commit, mypy config, towncrier config, circleci etc are based on same files from [Hydra](https://github.com/facebookresearch/hydra).

* Implementation Inherited from [MTRL](https://mtrl.readthedocs.io/en/latest/index.html) library. 

* Documentation of MTRL repository refer to: [https://mtrl.readthedocs.io](https://mtrl.readthedocs.io).