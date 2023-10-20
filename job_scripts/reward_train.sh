#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N reward_train
#$ -cwd                  
#$ -l h_rt=00:35:00 
#$ -l h_vmem=100G
#$ -o logs/train.log
#$ -e logs/train.err
#$ -q gpu
#$ -pe gpu-a100 2
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem
# Initialise the environment modules
. /etc/profile.d/modules.sh
export XDG_CACHE_HOME="/exports/eddie/scratch/s1808795/.cache"

# Load Python
module load cuda
module load python/3.4.3

source /exports/eddie/scratch/s1808795/PEFT-TRL-LLMs/venv/bin/activate

# Run the program
python ~/git/PEFT-TRL-LLMs/python_files/reward_model.py

deactivate
