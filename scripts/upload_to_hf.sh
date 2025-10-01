#!/bin/bash

#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos

#SBATCH --job-name=upload_to_hf  # <-- MOFIDY
#SBATCH --output=/home/jivitesj/projects/safety-1/coconut/lndata/logs/%x.out
#SBATCH --error=/home/jivitesj/projects/safety-1/coconut/lndata/logs/%x.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32 # <-- MODIFY
#SBATCH --mem=300G  # <-- MODIFY

#SBATCH --time=1-00:00:00  # <-- MODIFY

#SBATCH --mail-user=jivitesj@andrew.cmu.edu   # Your email address
#SBATCH --mail-type=BEGIN                    # Send email when the job starts
#SBATCH --mail-type=END                      # Send email when the job ends
#SBATCH --mail-type=FAIL                     # Send email if the job fails

set -e # Exit on any error
set -o pipefail # Exit on any error in a pipeline

source /home/jivitesj/batch_jobs.bashrc
mamba activate interp-1

cd ~/projects/safety-1/coconut/lndata/
hf upload-large-folder jiviteshjn/coconut-checkpoints --repo-type=model checkpoints --num-workers=32