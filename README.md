# proof-repair-LLM-Lean4

**Description**: This repository contains all the software artifacts from the MSc in Artificial Intelligence Individual Project on using LLMs for performing proof repair in Lean 4, including any modules that were developed as a product of the project and records of experiments performed to yield the results in the final report.


## Table of Contents

1. [Prerequsites](#prerequsites)
2. [Pipeline Usage](#pipeline)


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

3a. Install dependencies for using the pipeline (it is recommended to do it within a virtual envrionment):
    ```bash
    pip install -r pipeline_requirements.txt
    ```

3b. Install dependencies for running experiments with LLMs
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
python -m pipeline.tracer.trace -d RELATIVE/PATH/FROM/TARGE/REPO/ROOT/TO/DIR
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

