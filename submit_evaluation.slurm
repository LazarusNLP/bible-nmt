#!/bin/bash
#SBATCH --job-name=bible-nmt-training
#SBATCH --account=punim0478
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=0-48:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=setiawand@unimelb.edu.au

# Change to the directory where the script is located
cd $SLURM_SUBMIT_DIR

# Source bashrc to get conda
source ~/.bashrc

# Load required modules
module load CUDA/12.4.1 cuDNN/9.6.0.74-CUDA-12.4.1

# Set CUDA environment variables
export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.4.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Initialize conda and activate environment
conda activate david

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

# Run the training script
bash ./run_evaluation.sh

echo "End time: $(date)"