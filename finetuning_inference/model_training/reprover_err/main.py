from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, AutoPeftModelForSeq2SeqLM
from datetime import datetime
import json
import torch
import os
import argparse

MODEL_NAME = "reprover_err"


def tokenize(datapoints, tokenizer):
    # tokenize the inputs
    inputs = []
    for statement, failed_proof, error_msg in zip(datapoints['statement'], datapoints['failed_proof'], datapoints['error_msg']):
        inputs.append(error_msg+"<|SOS|>"+statement+"<|SOP|>"+failed_proof)
    tokenized_inputs = tokenizer(inputs, max_length=3072, padding="max_length", truncation=True)

    # tokenize the targets
    targets = [proof for proof in datapoints['proof']]
    tokenized_targets = tokenizer(targets, max_length=2560, padding="max_length", truncation=True)
    tokenized_inputs['labels'] = tokenized_targets['input_ids']
    return tokenized_inputs


def tokenize_inputs(datapoints, tokenizer):
    # tokenize the inputs
    inputs = []
    for statement, failed_proof, error_msg in zip(datapoints['statement'], datapoints['failed_proof'], datapoints['error_msg']):
        inputs.append(error_msg+"<|SOS|>"+statement+"<|SOP|>"+failed_proof)
    tokenized_inputs = tokenizer(inputs, max_length=3072, padding="max_length", truncation=True)
    return tokenized_inputs


def generate_predictions(tokenized_inputs, tokenizer, model):

    tokenized_inputs = tokenized_inputs.copy()
    input_ids = torch.tensor(tokenized_inputs['input_ids']).to('cuda')
    attention_mask = torch.tensor(tokenized_inputs['attention_mask']).to('cuda')
    
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, 
                                 attention_mask=attention_mask, 
                                 max_new_tokens=2560, 
                                 num_beams=4,
                                 use_cache=True,
                                 early_stopping=True,
                                )

    tokenized_inputs["prediction"] = tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=True)
    return tokenized_inputs


def finetune(train_split, valid_split, data_split, report_to="none"):

    
    tokenizer = AutoTokenizer.from_pretrained("kaiyuy/leandojo-lean4-tacgen-byt5-small", clean_up_tokenization_spaces=False)
    model = AutoModelForSeq2SeqLM.from_pretrained("kaiyuy/leandojo-lean4-tacgen-byt5-small", 
                                                  use_cache=False, 
                                                  torch_dtype=torch.bfloat16)

    lora_config = LoraConfig(
        inference_mode=False,
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="SEQ_2_SEQ_LM",
        use_rslora=True,
        use_dora=True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.relpath(script_dir, start=os.getcwd())
    tokenizer_dir = os.path.join(model_dir, f"new_tokenizer_{data_split}")

    tokenizer.add_tokens(["<|SOP|>"], special_tokens=True)
    tokenizer.add_tokens(["<|SOS|>"], special_tokens=True)
    tokenizer.truncation_side = "left"
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.save_pretrained(tokenizer_dir)
    
    model.train()
    
    if train_split == 'train':
        train_datapath = f'proof_repair_data/{data_split}/train.csv'
    elif train_split == 'train-valid':
        train_datapath = f'proof_repair_data/{data_split}/train_valid.csv'
    else:
        train_datapath = f'proof_repair_data/proof_repair_dataset.csv'


    if valid_split == 'valid':
        valid_datapath = f'proof_repair_data/{data_split}/valid.csv'
    elif valid_split == 'test':
        valid_datapath = f'proof_repair_data/{data_split}/test.csv'
    elif valid_split == 'valid-test':
        valid_datapath = f'proof_repair_data/{data_split}/valid_test.csv'
    else:
        valid_datapath = None
        
    if valid_datapath is None:
        data_files = {'train': train_datapath} 
    else:
        data_files = {'train': train_datapath, 'valid': valid_datapath} 

    data = load_dataset('csv', data_files=data_files)
    
    
    tokenize_fn = lambda x: tokenize(x, tokenizer)
    tokenized_data = data.map(tokenize_fn, batched=True, remove_columns=data['train'].column_names)
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, model=model)

    # specify training args
    training_args = Seq2SeqTrainingArguments(
        run_name=f"{MODEL_NAME}-{data_split}-{datetime.now().strftime('%m-%d-%H-%M')}",
        output_dir=f"model_training/{MODEL_NAME}/checkpoints-{data_split}-{datetime.now().strftime('%m-%d-%H-%M')}",
        learning_rate=5e-4,        
        lr_scheduler_type="cosine",
        optim="adamw_bnb_8bit",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=25,
        save_steps=25,
        logging_steps=5,
        save_total_limit=1,
        num_train_epochs=8,
        group_by_length=True,
        bf16=True,
        bf16_full_eval=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=report_to,
        torch_compile=True,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data.get('valid', None),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    torch.cuda.empty_cache()
    trainer.train()


def inference(inference_split, data_split, checkpoint_path):
    # get the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.relpath(script_dir, start=os.getcwd())
    checkpoint_dir = os.path.join(model_dir, checkpoint_path)
    tokenizer_dir = os.path.join(model_dir, f"new_tokenizer_{data_split}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, clean_up_tokenization_spaces=False)

    model = AutoPeftModelForSeq2SeqLM.from_pretrained(checkpoint_dir, 
                                                      use_cache=True,
                                                      torch_dtype=torch.bfloat16,
                                                     ).to("cuda")
    tokenizer.truncation_side = "left"
    model = model.eval()
    if inference_split == 'train':
        datapath = f'proof_repair_data/{data_split}/train.csv'
    elif inference_split == 'valid':
        datapath = f'proof_repair_data/{data_split}/valid.csv'
    else:
        datapath = f'proof_repair_data/{data_split}/test.csv'

    data = load_dataset('csv', data_files={'test': datapath})

    tokenize_fn = lambda x: tokenize_inputs(x, tokenizer)
    tokenized_data = data.map(tokenize_fn, batched=True, remove_columns=data['test'].column_names)

    generate_fn = lambda x: generate_predictions(x, tokenizer, model)
    model = torch.compile(model)
    torch.cuda.empty_cache()
    predictions = tokenized_data['test'].map(generate_fn, batched=True, batch_size=96)['prediction']
    
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
            os.environ["WANDB_PROJECT"] = "seq2seq"
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

