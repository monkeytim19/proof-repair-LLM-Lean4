from git import Repo
import os

# set the root path of the working directory
ROOT_DIR = "/Users/timmonkey/Desktop/Imperial/Summer Term/Individual Project/Repos/proof-repair-LLM-Lean4"

# set path to reference repository and directory to keep copies of them
REPO_PATH = "/Users/timmonkey/Desktop/Imperial/Summer Term/Individual Project/Repos/mathlib4"
REPO_COPY_DIR = "/Users/timmonkey/Desktop/Imperial/Summer Term/Individual Project/Repos/temp"

# set paths relevant to repository tracing
THEOREM_EXTRACTOR_DIR = os.path.join(ROOT_DIR, "pipeline/tracer/TheoremExtractor")
TRACED_INFO_DIR = os.path.join(ROOT_DIR, "pipeline/tracer/traced_info")

# set paths relevant to data collection
FILENAMES_DIR = os.path.join(ROOT_DIR, "pipeline/scraper/filenames")
RAW_DATA_DIR = os.path.join(ROOT_DIR, "pipeline/scraper/raw_data")
DATA_DIR = os.path.join(ROOT_DIR, "pipeline/scraper/processed_data")

# set path relevant to verification
DATA_INDICES_DIR = os.path.join(ROOT_DIR, "pipeline/utils/condor/verifier/data_indices")
OUTCOME_DIR = os.path.join(ROOT_DIR, "pipeline/utils/condor/verifier")


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