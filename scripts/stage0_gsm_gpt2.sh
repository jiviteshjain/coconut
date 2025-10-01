#!/bin/bash

#SBATCH --partition=general,r3lit

#SBATCH --job-name=coco0_gpt2_gsm  # <-- MOFIDY
#SBATCH --output=~/projects/safety-1/coconut/lndata/logs/%x.out
#SBATCH --error=~/projects/safety-1/coconut/lndata/logs/%x.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2  # <-- MODIFY
#SBATCH --mem=80G  # <-- MODIFY

#SBATCH --gres=gpu:2  # <-- MODIFY
#SBATCH --constraint=A100_80GB|H100  # <-- MODIFY
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
cd ~/projects/safety-1/coconut
source .venv/bin/activate

torchrun --nnodes 1 --nproc_per_node 2 run.py args/gsm_cot.yaml
