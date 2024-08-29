#!/bin/bash

dirname=$1

# activate virtual environment and add python to PATH
source /vol/bitbucket/tcwong/individual_project/venv/bin/activate
export PYTHONPATH=/vol/bitbucket/tcwong/individual_project/proof-repair-LLM-Lean4

# add elan to PATH
export PATH=$(echo $PATH | tr ':' '\n' | grep -v "/homes/tcwong/.elan/bin" | tr '\n' ':')
export PATH="$PATH:/vol/bitbucket/tcwong/individual_project/leandojo-reprover/.elan/bin"

# run the python script to generate the data
python /vol/bitbucket/tcwong/individual_project/proof-repair-LLM-Lean4/pipeline/tracer/trace.py -d $dirname

