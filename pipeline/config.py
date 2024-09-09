from git import Repo
import os

# set the root path of the working directory
ROOT_DIR = os.getcwd() # TO SET

# set path to reference repository and directory to keep copies of them
REPO_PATH = "temp" # TO SET
REPO_COPY_DIR = os.path.join(ROOT_DIR, "pipeline/repo_copies") # TO SET

# set paths relevant to repository tracing
THEOREM_EXTRACTOR_DIR = os.path.join(ROOT_DIR, "pipeline/tracer/TheoremExtractor")
TRACED_INFO_DIR = os.path.join(ROOT_DIR, "pipeline/tracer/traced_info")

# set paths relevant to data collection
FILENAMES_DIR = os.path.join(ROOT_DIR, "pipeline/scraper/filenames")
RAW_DATA_DIR = os.path.join(ROOT_DIR, "pipeline/scraper/raw_data")
DATA_DIR = os.path.join(ROOT_DIR, "pipeline/scraper/processed_data")

# set path relevant to verification
DATA_INDICES_DIR = os.path.join(ROOT_DIR, "pipeline/utils/condor/verifier/data_indices")

SEED = 2024 # set seed for randomisation

REPO = Repo(REPO_PATH)
REF_COMMIT = REPO.head.commit

DIRECTORIES_TO_SCRAPE = [
        "Algebra",
        "Analysis",
        "Computability",
        "FieldTheory",
        "InformationTheory",
        "LinearAlgebra",
        "MeasureTheory",
        "Order",
        "RingTheory",
        "AlgebraicGeometry",
        "CategoryTheory",
        "Condensed",
        "Geometry",
        "Logic",
        "ModelTheory",    
        "Probability",
        "SetTheory",
        "AlgebraicTopology",
        "Combinatorics",
        "Dynamics",
        "GroupTheory",
        "NumberTheory",
        "RepresentationTheory",
        "Topology",
        "Data",
    ]
DIRECTORIES_TO_SCRAPE = ["Mathlib/"+dir for dir in DIRECTORIES_TO_SCRAPE]