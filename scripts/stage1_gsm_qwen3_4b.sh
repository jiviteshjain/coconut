#!/bin/bash

#SBATCH --partition=general,r3lit

#SBATCH --job-name=coco1_qwen3_4b_gsm  # <-- MOFIDY
#SBATCH --output=/home/jivitesj/projects/safety-1/coconut/lndata/logs/%x.out
#SBATCH --error=/home/jivitesj/projects/safety-1/coconut/lndata/logs/%x.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2 # <-- MODIFY
#SBATCH --mem=200G  # <-- MODIFY

#SBATCH --gres=gpu:2  # <-- MODIFY
#SBATCH --constraint=VRAM_80GB  # <-- MODIFY
# A100_80GB if 80gb is needed
# A6000|L40|L40S|6000Ada if 40gb is needed

#SBATCH --time=2-00:00:00  # <-- MODIFY

#SBATCH --mail-user=jivitesj@andrew.cmu.edu   # Your email address
#SBATCH --mail-type=BEGIN                    # Send email when the job starts
#SBATCH --mail-type=END                      # Send email when the job ends
#SBATCH --mail-type=FAIL                     # Send email if the job fails

set -e # Exit on any error
set -o pipefail # Exit on any error in a pipeline

source /home/jivitesj/batch_jobs.bashrc
export HF_HOME=/scratch/jivitesj/hf_cache
export HF_HUB_CACHE=/scratch/jivitesj/hf_cache/hub
export HF_DATASETS_CACHE=/scratch/jivitesj/hf_cache/datasets
mkdir -p $HF_HOME $HF_HUB_CACHE $HF_DATASETS_CACHE

cd ~/projects/safety-1/coconut
source .venv2/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
# For precise CUDA site (slow, use only to catch the first bad step)
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_NCCL_BLOCKING_WAIT=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
torchrun --master_port 29503 --nnodes 1 --nproc_per_node 2 run.py args/gsm_coconut_qwen.yaml
