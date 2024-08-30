from git import Repo
import os

# set path to reference repository and directory to keep copies of them
REPO_PATH = "/Users/timmonkey/Desktop/Imperial/Summer Term/Individual Project/Repos/mathlib4"
REPO_COPY_DIR = "/Users/timmonkey/Desktop/Imperial/Summer Term/Individual Project/Repos/temp"
# REPO_PATH = "/vol/bitbucket/tcwong/individual_project/mathlib4"
# REPO_COPY_DIR = "/vol/bitbucket/tcwong/individual_project/repo_verifiy"

ROOT_DIR = "/Users/timmonkey/Desktop/Imperial/Summer Term/Individual Project/Repos/proof-repair-LLM-Lean4"

# set paths relevant to repository tracing
THEOREM_EXTRACTOR_DIR = os.path.join(ROOT_DIR, "pipeline/tracer/TheoremExtractor")
TRACED_INFO_DIR = os.path.join(ROOT_DIR, "pipeline/tracer/traced_info")

# set paths relevant to data collection
FILENAMES_DIR = os.path.join(ROOT_DIR, "pipeline/utils/condor/data_collection/filenames")
RAW_DATA_DIR = os.path.join(ROOT_DIR, "pipeline/data_collection/raw_data")
DATA_DIR = os.path.join(ROOT_DIR, "pipeline/data_collection/processed_data")

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