#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=mail@co.jp
#SBATCH --gres=gpu:1

#SBATCH -o ./log/%j.txt

export CUDA_HOME=/usr/local/cuda-11.3
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3 main.py Outdoor test