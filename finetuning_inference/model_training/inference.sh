#!/bin/bash

MODEL_NAME=""
INFERENCE_SPLIT="test"
DATA_SPLIT=""
CKPT_PATH=""
NOHUP=false

while getopts m:i:d:c:gn flag; do
    case "${flag}" in
        m) MODEL_NAME=${OPTARG};;  
        i) INFERENCE_SPLIT=${OPTARG};;  
        d) DATA_SPLIT=${OPTARG};;
        c) CKPT_PATH=${OPTARG};;
        n) NOHUP=true;;
        *) 
            echo "Invalid option: -$flag" 
            exit 1 
            ;;
    esac
done

# check for -m flag
if [[ -z "$MODEL_NAME" ]]; then
    echo "-m flag must be present to specify the name of the model."
    exit 1
fi

# check for -i flag
if [[ "$INFERENCE_SPLIT" != "train" && "$INFERENCE_SPLIT" != "valid" && "$INFERENCE_SPLIT" != "test" ]]; then
    echo "Invalid option for -i. Use 'train' or 'valid', or 'test'."
    exit 1
fi

# check for -d flag
if [[ "$DATA_SPLIT" != "random" && "$DATA_SPLIT" != "by_file" ]]; then
    echo "Invalid option for -d. Use 'random' or 'by_file'."
    exit 1
fi

# check for -c flag
if [[ -z "$CKPT_PATH" ]]; then
    echo "-c flag must be present to specify the checkpoint of the model to perform inference with."
    exit 1
fi

echo "Performing inference on $INFERENCE_SPLIT data from the $DATA_SPLIT data split using model ${MODEL_NAME}."

if $NOHUP; then
    nohup python model_training/${MODEL_NAME}/main.py -m inference --inference-split $INFERENCE_SPLIT --data-split $DATA_SPLIT --ckpt-path $CKPT_PATH > model_training/${MODEL_NAME}/inference_${DATA_SPLIT}.out 2>&1
else
    python model_training/${MODEL_NAME}/main.py -m inference --inference-split $INFERENCE_SPLIT --data-split $DATA_SPLIT --ckpt-path $CKPT_PATH
fi