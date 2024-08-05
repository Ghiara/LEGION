# Installation of LEGION

## 1. Prerequisites
You can skip this if you have installed all required packages.
### 1.1 The code is installed and tested on Ubuntu 22.04 LTS, make sure your OS is Ubuntu 20.04 or 22.04

### 1.2 We use miniconda to manage the virtual environment, make sure you have installed miniconda following: https://docs.anaconda.com/miniconda/

These four commands quickly and quietly install the latest 64-bit version of the installer and then clean up after themselves. To install a different version or architecture of Miniconda for Linux, change the name of the `.sh` installer in the `wget` command.

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

```

After installing, initialize your newly-installed Miniconda. The following commands initialize for bash and zsh shells:

```bash
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

### 1.3 We use Mujoco200 to build RL einvironment 

To install `Mujoco200`, following: https://www.roboti.us/download.html, download `mujoco200 linux`, unzip the file and place at `your/root/path/to/.mujoco/` with the folder name `mujoco200`. Namely, the full path should be something like `/home/ubuntu/.mujoco/mujoco200`. After that, add following cli (modify according to your own path) in your root `.bashrc` file:
```bash
export  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/your/.mujoco/mujoco200/bin 
```

## 2.Installation of LEGION dependencies.

Here we will introduce how to establish the legion dependencies.

### 2.1 build virtual env:

Open a terminal:
```bash
conda create -n legion python=3.7

conda activate legion

pip install dm_control mujoco-py==2.0.2.8 cython==0.29.33 protobuf==3.20.0 gym==0.20.0
```

### 2.2 Download and install LEGION dependencies:

Install legion repository dependencies

```bash
(optinal) git clone https://github.com/Ghiara/LEGION.git

cd LEGION && conda activate legion && pip install -r requirements/dev.txt
```

Install modified metaworld environments under `source` folder

```bash
cd source

git clone https://github.com/Ghiara/Metaworld-KUKA-IIWA-R800.git

cd Metaworld-KUKA-IIWA-R800 && pip install -e . && cd ..
```

Install Bayesian Nonparametric Models library (bnpy) under `source` folder

```bash
git clone https://github.com/bnpy/bnpy.git

cd bnpy && pip install -e . && cd ..
```


Install `mtenv` modified environment manager

```bash
git clone https://github.com/Ghiara/mtenv.git

cd mtenv && pip install -e . && cd ..
```

Before you run the code, change the `task_encoder_cfg.path_to_load_from` in the `config/agent/components/continuouslearning_multitask.yaml` with your local repository path.

## 3 Error catching
- TypeError: Descriptors cannot not be created directly. If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.If you cannot immediately regenerate your protos, some other possible workarounds are: 1. Downgrade the protobuf package to 3.20.x or lower.2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).

```bash
pip install protobuf==3.20.0
```

- Failed import mujoco-py: 

try another version of mujoco-py, for example:
```bash 
pip install mujoco-py==2.0.2.10
```

- ImportError: /home/.../miniconda3/envs/legion/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /lib/x86_64-linux-gnu/libLLVM-13.so.1)
  
following this page: https://bcourses.berkeley.edu/courses/1478831/pages/glibcxx-missing
```bash
cd /home/to/your/anaconda3/envs/legion/lib
mv libstdc++.so.6 libstdc++.so.6.old
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6
```

