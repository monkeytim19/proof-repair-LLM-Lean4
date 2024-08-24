#!/bin/bash

dirname=$1

# activate virtual environment and add python to PATH
source /vol/bitbucket/tcwong/individual_project/venv/bin/activate # TODO need to be flexible with the working directories and paths
export PYTHONPATH=/vol/bitbucket/tcwong/individual_project/proof-repair

# add elan to PATH
export PATH=$(echo $PATH | tr ':' '\n' | grep -v "/homes/tcwong/.elan/bin" | tr '\n' ':')
export PATH="$PATH:/vol/bitbucket/tcwong/individual_project/leandojo-reprover/.elan/bin"

# run the python script to generate the data
python /vol/bitbucket/tcwong/individual_project/proof-repair/trace_repo_theorems.py -d $dirname

