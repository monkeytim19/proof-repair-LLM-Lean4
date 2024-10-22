import os
import json
from pipeline.utils.theorem_extraction import file_num_theorem
from pipeline.utils.grouping import groups_by_thm_num
from pipeline.config import RAW_DATA_DIR, DIRECTORIES_TO_SCRAPE


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
    

def write_file(filepath, file_str):
    """Overwrite the file at the specified filepath with new string."""
    with open(filepath, "w") as file:
        file.write(file_str)