\#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N sft
#$ -cwd                  
#$ -l h_rt=03:00:00 
#$ -l h_vmem=100G
#$ -o logs/train.log
#$ -e logs/train.err
#$ -q gpu
#$ -pe gpu-a100 1
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem

> $(pwd)/logs/train.err
> $(pwd)/logs/train.log

# Initialise the environment modules
. /etc/profile.d/modules.sh
export XDG_CACHE_HOME="/exports/eddie/scratch/s1808795/.cache"

# Load Python
module load cuda
module load python/3.11.4


# source activate myenv
source /exports/eddie/scratch/s1808795/PEFT-TRL-LLMs/venv/bin/activate
# /exports/eddie/scratch/s1808795/miniconda3/bin/conda init bash
# /exports/eddie/scratch/s1808795/miniconda3/bin/conda activate myenv

# Run the program
python ~/git/PEFT-TRL-LLMs/python_files/sft_example.py

deactivate
