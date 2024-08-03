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

### 1. build virtual env:

Open a terminal:
```bash
conda create -n legion python=3.7
```

### 2. Download and install LEGION dependencies:
```bash
(optinal) git clone https://github.com/Ghiara/LEGION.git

cd LEGION

conda activate legion

pip install -r requirements/dev.txt
```

