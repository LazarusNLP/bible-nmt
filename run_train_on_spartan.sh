#!/bin/bash
source ~/.bashrc

module load CUDA/12.4.1 cuDNN/9.6.0.74-CUDA-12.4.1

conda init
conda activate david

# Run the training script
bash run_train.sh