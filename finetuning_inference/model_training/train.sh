#!/bin/bash

MODEL_NAME=""
TRAIN_SPLIT="train"
VALID_SPLIT="valid"
DATA_SPLIT=""
API_KEY=""
NOHUP=false

while getopts m:t:v:d:k:n flag; do
    case "${flag}" in
        m) MODEL_NAME=${OPTARG};;  
        t) TRAIN_SPLIT=${OPTARG};;
        v) VALID_SPLIT=${OPTARG};;
        d) DATA_SPLIT=${OPTARG};;
        k) API_KEY=${OPTARG};;
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

# check for -t flag
if [[ "$TRAIN_SPLIT" != "train" && "$TRAIN_SPLIT" != "train-valid" && "$TRAIN_SPLIT" != "all" ]]; then
    echo "Invalid option for -t. Use 'train' or 'train-valid' or 'all'."
    exit 1
fi

# check for -d flag
if [[ "$DATA_SPLIT" != "random" && "$DATA_SPLIT" != "by_file" ]]; then
    echo "Invalid option for -d. Use 'random' or 'by_file'."
    exit 1
fi


# check for -v flag
if [[ "$VALID_SPLIT" != "valid" && "$VALID_SPLIT" != "valid-test" && "$VALID_SPLIT" != "test" && "$VALID_SPLIT" != "none" ]]; then
    echo "Invalid option for -v. Use 'valid' or 'valid-test' or 'test' or 'none'."
    exit 1
fi


echo "Finetuning ${MODEL_NAME} using ${DATA_SPLIT} data split - training with ${TRAIN_SPLIT} data and evaluating with $VALID_SPLIT data."

if $NOHUP; then
    nohup python model_training/${MODEL_NAME}/main.py -m finetune -k $API_KEY --train-split $TRAIN_SPLIT --valid-split $VALID_SPLIT --data-split $DATA_SPLIT > model_training/${MODEL_NAME}/${DATA_SPLIT}_finetune.out 2>&1
    
else
    python model_training/${MODEL_NAME}/main.py -m finetune -k $API_KEY --train-split $TRAIN_SPLIT --valid-split $VALID_SPLIT --data-split $DATA_SPLIT
fi