import os
import json
from utils.theorem_extraction import file_num_theorem
from utils.grouping import groups_by_thm_num
from config import RAW_DATA_DIR, DIRECTORIES_TO_SCRAPE


def unfinished_files_groups(n_groups):
    """
    Returns and organizes the files that have not been scraped yet into separate groups by applying 
    the groups_by_thm_num function.
    """
    files = [f for f in os.listdir(RAW_DATA_DIR) if os.path.isfile(os.path.join(RAW_DATA_DIR, f))]
    files = [f.replace("_", "/").replace(".json", ".lean") for f in files]
    subset_file_num_thm = [tup for tup in file_num_theorem(DIRECTORIES_TO_SCRAPE) if tup[0] not in files] 
    return groups_by_thm_num(n_groups, subset_file_num_thm)
    

def save_data_to_json_in_dir(data, json_file_name, dir_path):
    """Saves the dataset to a .json file. in a specific directory."""
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, f"{json_file_name}.json")
    with open(path, "w") as file:
        json.dump(data, file)


def read_json_file(filepath):
    """Loads and reads a .json file."""
    with open(filepath, "r") as file:
        return json.load(file)
    

# def leanfile_replace_slash(path, repl):
#     """
#     Replaces the path of a .lean file by removing the .lean suffix and replaces its '/' with another string.
#     """
#     return path.replace("/", repl)[:-5]


# def is_lean_suffix(path):
#     """Returns whether the suffix to the given path is '.lean' or not, i.e. if it is a .lean file."""
#     return path[-min(5, len(path)):] == ".lean"