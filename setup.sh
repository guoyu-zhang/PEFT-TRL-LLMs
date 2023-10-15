#!/bin/bash

cd /exports/eddie/scratch/s1808795

# Hardcoded GitHub repo URL
REPO_URL="https://github.com/guoyu-zhang/PEFT-TRL-LLMs.git"

# Clone the repository
git clone $REPO_URL
if [ $? -ne 0 ]; then
    echo "Error cloning the repository!"
    exit 1
fi

# Extract the repo name from the URL for the directory name
REPO_NAME=$(basename -s .git $REPO_URL)

cd $REPO_NAME

# Create a virtual environment named 'venv'
# python -m venv /exports/eddie/scratch/s1808795/git/$REPO_NAME/venv
~/miniconda3/bin/python -m venv venv

# Activate the virtual environment
# source /exports/eddie/scratch/s1808795/git/$REPO_NAME/venv/bin/activate
source venv/bin/activate

# You can install necessary packages here if needed
pip install -r requirements.txt

echo "Virtual environment setup is complete!"

# To deactivate the virtual environment, use the 'deactivate' command
deactivate

