# proof-repair-LLM-Lean4

**Description**: This repository contains all the software artifacts from the MSc in Artificial Intelligence Individual Project on using LLMs for performing proof repair in Lean 4, including any modules that were developed as a product of the project and records of experiments performed to yield the results in the final report.


## Table of Contents

1. [Prerequsites](#prerequsites)
2. [Pipeline Usage](#pipeline)
3. [LLM Fine-tuning & Inference](#model)
4. [Experiment Results](#results)


## Prerequsites

To access and run any scripts provided in this repository, please follow the following steps to set-up:

1. Clone the repository:
    ```bash
    git clone https://github.com/username/repository-name.git
    ```

2. Navigate into the project directory:
    ```bash
    cd proof-repair-LLM-Lean4
    ```

3. If using the pipeline, install dependencies:
    ```bash
    pip install -r pipeline_requirements.txt
    ```

4. If running experiments with LLMs and models, install dependencies:
    ```bash
    pip install -r model_requirements.txt
    ```


## Pipeline

The pipeline consists of 3 main components: ```tracer```, ```scraper```, ```verifier```. Below provides the basic instructions on how to use them.

### Set-up

Before any part of the pipeline can be used, it is necessary to fill in all the necessary details in ```pipeline.config.py```.

### Tracer

The ```tracer``` can retrieve information on theorems from all ```.lean``` files in the directory that is provided to it. To run it:
```bash
python -m pipeline.tracer.trace -d RELATIVE/PATH/FROM/TARGET/REPO/ROOT/TO/DIR
```

### Scraper

The ```scraper``` scrapes data from a Git repository that corresponds to a Lean 4 project, preprocesses it, and prepares it for the use of AI model training. 


To use scrape data with it, take the following steps:

1. Create a ```FOO.txt``` file within the ```FILENAMES_DIR``` directory that was previously specified in ```pipeline.config.py```. Within this file, include all the relative paths from the target repository to the ```.lean``` files or subdirectories that you wish to scrape data from. Separate each entry by a new line. 

2.  Run 
```bash
python -m pipeline.data_collection.collect -f FOO
```

After scraping the raw data, it is possible to aggregate and preprocess it via:
```bash
python -m pipeline.data_collection.preprocess -d BAR.csv
```
The preprocessed data will then be saved as ```BAR.csv``` within the ```DATA_DIR``` directory from ```pipeline.config.py```. To ensure the soundness of the dataset, it is strongly recommended to use the ```verifier``` to ensure that all extracted reference/ground-truth proofs are indeed valid relative to its theorem statement

Finally, it is also possible to perform data-spliting with the module using a fully randomised split by:
```bash
python -m pipeline.data_collection.split -d BAR.csv -v NUM_VALIDATION -t NUM_TEST -r
```
Alternatively, spliting the data based on grouping the datapoints by their source ```.lean``` file can be done by:
```bash
python -m pipeline.data_collection.split -d BAR.csv -v NUM_VALIDATION -t NUM_TEST
```

### Verifier

The ```verifier``` can verify the validity of theorem proofs by replacing the proof attempt to the original ```.lean``` files and asking Lean's kernel to check it. 
Note that the attempted proof must be part of a ```.csv``` file with additional columns ```proof```, ```thm_name```, ```filepath```, ```statement```, ```commit```.

To run it:
```bash
python -m pipeline.verifier.verify -d RELATIVE/PATH/TO/CSV/FILE -c COLUMN_NAME_TO_VERIFY
```
If the -c flag is not specified, then the ```verifier``` will by default verify the column with the heading ```predicted_proof```.

## LLM Fine-tuning and Inference

All the models that have been trained are located in a subdirectory with their model names within the ```model_training``` directory.

To perform any fine-tuning or inference with LLMs, first navigate to the ```finetuning_inference``` directory via:
```bash
cd finetuning_inference
```

Then open the file ```run_commands.txt```. From this file, you can copy the appropriate command and run it in bash. For instance, if you want to perform fine-tuning on the ReProver model using the training data and evaluating it using the validation data from the by_file split, you can run:
```bash
model_training/train.sh -m reprover -n -t train-valid -v test -d random
```

In general, for fine-tuning you can run:
```bash
model_training/train.sh -m MODEL_NAME -k WANDB_API_KEY -n -t TRAIN_SPLIT -v EVAL_SPLIT -d DATA_SPLIT
```
Note, for this to run, there should exsit ```proof_repair_data/DATA_SPLIT/TRAIN_SPLIT.csv``` and ```proof_repair_data/DATA_SPLIT/EVAL_SPLIT.csv```. Also, this requires that the directory ```model_training/MODEL_NAME``` exists with the ```main.py``` file, which should contain the ```finetune``` and ```inference``` functions. 

For performing inference on a model, you can run
```bash
model_training/inference.sh -m MODEL_NAME -i TEST_SPLIT -d DATA_SPLIT -c RELATIVE/PATH/TO/MODEL_CKPT_DIR/FROM/MODEL_NAME
```
Again, this will require that ```proof_repair_data/DATA_SPLIT/TEST_SPLIT.csv``` exists. Also, it will require that ```RELATIVE/PATH/TO/MODEL_CKPT_DIR/FROM/MODEL_NAME``` refers to to a model checkpoint that is used to run the inference. 

If you want to run inference on a pre-trained base model, then replace the ```RELATIVE/PATH/TO/MODEL_CKPT_DIR/FROM/MODEL_NAME``` directly with the path from HuggingFace. For instance, to perform inference on the base ReProver generator model, you can run:

```bash
model_training/inference.sh -m reprover -i test -d by_file -c kaiyuy/leandojo-lean4-tacgen-byt5-small -n
```

## Experiment Results

All experiment results are within the directory ```./experiments```, can each subdirectory contains the test predictions from inference and also the indices of the datapoints in mathlib4-repair that either were successfully or unsucessfully repaired by the model. 

Checkpoints to LLMs that have already been fine-tuned are saved as a collection on HuggingFace and can be accessed via: https://huggingface.co/collections/tcwong/proof-repair-llm-lean4-66de36a17a044c1fbdeaf8de
