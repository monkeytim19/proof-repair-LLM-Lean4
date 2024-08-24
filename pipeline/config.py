from git import Repo
import os

# set path to reference repository and directory to keep copies of them
REPO_PATH = "/Users/timmonkey/Desktop/Imperial/Summer Term/Individual Project/Repos/mathlib4"
REPO_COPY_DIR = "/Users/timmonkey/Desktop/Imperial/Summer Term/Individual Project/Repos/temp"

# set paths relevant to repository tracing
THEOREM_EXTRACTOR_DIR = os.path.join(os.getcwd(), "tracer/TheoremExtractor")
TRACED_INFO_DIR = os.path.join(os.getcwd(), "tracer/traced_info")

# set paths relevant to data collection
FILENAMES_DIR = os.path.join(os.getcwd(), "data_collection/filenames")
RAW_DATA_DIR = os.path.join(os.getcwd(), "data_collection/raw_data")
DATA_DIR = os.path.join(os.getcwd(), "data_collection/processed_data")

# TRACED_INFO_DIR = '/vol/bitbucket/tcwong/individual_project/proof-repair/traced_info'
# RAW_DATA_DIR = '/vol/bitbucket/tcwong/individual_project/proof-repair/raw_data'
# DATA_DIR = '/vol/bitbucket/tcwong/individual_project/proof-repair/processed_data'

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
    ]
DIRECTORIES_TO_SCRAPE = ["Mathlib/"+dir for dir in DIRECTORIES_TO_SCRAPE]    


# def file_commits(filepath=None):
#     """
#     Returns the list of all commits from all branches in the repository relevant to the desired filepath.
    
#     If no filepath is specified, then the entire repository will be considered.
#     """
#     if filepath is None:
#         return list(REPO.iter_commits(all=True))
#     else:
#         return list(REPO.iter_commits(all=True, paths=filepath))
    

# REF_COMMIT = file_commits()[[f.hexsha for f in file_commits()].index('a261710852a957a7d20d89b962e4b59887549f21')]
