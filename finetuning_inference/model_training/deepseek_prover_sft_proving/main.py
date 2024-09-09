from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq, BitsAndBytesConfig
from peft import PeftConfig, get_peft_model, LoraConfig, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from datetime import datetime
import json
import numpy as np
import torch
import os
import sys
import argparse

MODEL_NAME = "deepseek_prover_sft_proving"


def tokenize_inputs(datapoints, tokenizer):
    # tokenize the inputs
    inputs = []
    for statement in datapoints['statement']:
        inputs.append(statement)
    tokenized_inputs = tokenizer(inputs, max_length=900, padding="max_length", truncation=True)
    return tokenized_inputs


def generate_predictions(tokenized_inputs, tokenizer, model):
    tokenized_inputs = tokenized_inputs.copy()
    input_ids = torch.tensor(tokenized_inputs['input_ids']).to('cuda')
    attention_mask = torch.tensor(tokenized_inputs['attention_mask']).to('cuda')
    
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, 
                                 attention_mask=attention_mask,
                                 max_length=3072, # max_positional_embedding length in model
                                 num_beams=1,
                                 use_cache=True,
                            )

    tokenized_inputs["prediction"] = tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=True)
    return tokenized_inputs


def extract_decoder_predictions(datapoints_with_predictions):
    extracted_predictions = []
    for statement, predicted_proof in zip(datapoints_with_predictions['statement'], datapoints_with_predictions['prediction']):
        extracted_predictions.append(predicted_proof[len(statement):])

    datapoints_with_predictions["prediction"] = extracted_predictions
    return datapoints_with_predictions


def inference(inference_split, data_split, checkpoint_path):
    base_model = "deepseek-ai/DeepSeek-Prover-V1.5-SFT"
    # get the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.relpath(script_dir, start=os.getcwd())
    tokenizer_dir = os.path.join(model_dir, f"new_tokenizer_{data_split}")

    if checkpoint_path == base_model:
        model = AutoModelForCausalLM.from_pretrained(base_model, 
                                                     attn_implementation="flash_attention_2", 
                                                     torch_dtype=torch.bfloat16,
                                                     use_cache=True,
                                                     device_map="cuda")
        tokenizer = AutoTokenizer.from_pretrained(base_model, clean_up_tokenization_spaces=False)


    else:
        checkpoint_dir = os.path.join(model_dir, checkpoint_path)
        model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_dir,
                                                         attn_implementation="flash_attention_2", 
                                                         torch_dtype=torch.bfloat16,
                                                         use_cache=True,
                                                         device_map="cuda")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, clean_up_tokenization_spaces=False)


    if tokenizer.pad_token is None:
        num_added_toks = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.truncation_side ="left"
    model.resize_token_embeddings(len(tokenizer))

    
    model = model.eval()
    if inference_split == 'train':
        datapath = f'proof_repair_data/{data_split}/train.csv'
    elif inference_split == 'valid':
        datapath = f'proof_repair_data/{data_split}/valid.csv'
    else:
        datapath = f'proof_repair_data/{data_split}/test.csv'

    data = load_dataset('csv', data_files={'test': datapath})

    tokenize_fn = lambda x: tokenize_inputs(x, tokenizer)
    tokenized_data = data.map(tokenize_fn, batched=True)

    generate_fn = lambda x: generate_predictions(x, tokenizer, model)
    model = torch.compile(model)
    torch.cuda.empty_cache()
    predictions = tokenized_data['test'].map(generate_fn, batched=True, batch_size=32)
    predictions = predictions.map(extract_decoder_predictions, batched=True, batch_size=64)['prediction']
    
    json_file = os.path.join(model_dir, f"{inference_split}_{data_split}_prediction.json")
    with open(json_file, "w") as file:
        json.dump(predictions, file)
    
    output_data = data['test'].add_column("predicted_proof", predictions)
    output_path = os.path.join(model_dir, f"{inference_split}_{data_split}_prediction.csv")
    output_data.to_csv(output_path)

        

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Finetune or perform inference on model.')

    parser.add_argument('-m', '--mode', choices=['finetune', 'inference'], required=True,
                        help='Select either to "finetune" or perform "inference" with the model on proof repair data.')

    parser.add_argument('-k', '--key', 
                        required='finetune' in parser.parse_known_args()[0].mode,
                        help='Weights and Biases API key for logging.')

    parser.add_argument('--train-split', 
                        choices=['train', 'train-valid', 'all'], 
                        default='train',
                        required='finetune' in parser.parse_known_args()[0].mode,
                        help='Select data split for the model to be finetuned on.')

    parser.add_argument('--valid-split', 
                        choices=['valid', 'test', 'valid-test', 'none'],
                        default='valid',
                        required='finetune' in parser.parse_known_args()[0].mode,
                        help='Select data split for the model to be evaluated during finetuning.')

    
    parser.add_argument('--inference-split', 
                        choices=['train', 'valid', 'test', 'all'], 
                        required='inference' in parser.parse_known_args()[0].mode,
                        help='Select data split for the model to be perform inference on.')

    
    parser.add_argument('--ckpt-path', 
                        required='inference' in parser.parse_known_args()[0].mode,
                        help='The path of the checkpoint, relative to the model directory.')

    parser.add_argument('--data-split', 
                        required=True,
                        choices=["random", "by_file"],
                        help="The type of data split used.")

    
    # Parse the command-line arguments
    args = parser.parse_args()


    # Parse the command-line arguments
    if args.mode == 'finetune':
        print("Finetuning starting.")
        if args.key == "":
            finetune(train_split=args.train_split, valid_split=args.valid_split)
        else:
            os.environ["WANDB_API_KEY"] = args.key
            os.environ["WANDB_PROJECT"] = "decoder"
            os.environ["WANDB_LOG_MODEL"] = "checkpoint"
            finetune(train_split=args.train_split, 
                     valid_split=args.valid_split, 
                     data_split=args.data_split,
                     report_to="wandb")
        print("Finetuning finished.")
    elif args.mode == 'inference':
        print(f"Inference starting on {args.inference_split} data using model from checkpoint {args.ckpt_path}.")
        inference(inference_split=args.inference_split, data_split=args.data_split, checkpoint_path=args.ckpt_path)
        print("Inference finished.")

