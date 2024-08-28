import os
import subprocess
import json
import argparse
from tqdm import tqdm
from datetime import datetime
from pipeline.utils.strings import leanfile_replace_slash, remove_comments, pos_conversion
from pipeline.utils.git_extraction import file_commits, file_str_from_commit, get_all_lean_subfile_paths
from pipeline.utils.files import save_data_to_json_in_dir
from pipeline.utils.theorem_extraction import theorem_statement_proof, substituted_file
from pipeline.utils.lean_repo_copying import create_repository_copy, remove_repository_copy
from pipeline.utils.lean_extraction import get_decl_name
from pipeline.config import TRACED_INFO_DIR, REPO_COPY_DIR, RAW_DATA_DIR, REF_COMMIT, FILENAMES_DIR


def scrape_file_history(dataset, filepath, all_thm_info, repo_copy_path):
    """
    Looks over the (Git commit) history of a .lean file to determine whether or not its past
    proof fails to compile as a proof for the reference theorem.

    If compilation is unsuccessful, it will be added to the dataset as a new datapoint. The given
    filepath should point to a .lean file.
    """
    print(f"{datetime.now()} - Working on {filepath}", flush=True)
    ref_file_str = remove_comments(file_str_from_commit(REF_COMMIT, filepath))
    full_file_path = os.path.join(repo_copy_path, filepath)
    filepath_module = leanfile_replace_slash(filepath, ".")

    # keep the current version of the file for restoration later
    with open(full_file_path, "r") as file:
        original_file_str = file.read()

    # verify if the comment removal causes compilation error
    with open(full_file_path, "w") as file:
        file.write(ref_file_str)
    try:
        subprocess.run(["lake", "build", filepath_module],
                            cwd=repo_copy_path,
                            capture_output=True,
                            text=True,
                            check=True
                        )
    except subprocess.CalledProcessError:
        print(f"COMMENT REMOVAL PROBLEM: {filepath}", flush=True)
        return dataset

    commits_ls = file_commits(filepath)
    for thm_info in all_thm_info:
        # extract declaration name
        thm_full_name = thm_info["name"]
        thm_decl_start_pos = pos_conversion(thm_info["start"])
        thm_decl_end_pos = pos_conversion(thm_info["end"])
        try:
            thm_decl_name = get_decl_name(original_file_str, thm_decl_start_pos, thm_decl_end_pos)
        except:
            print(f"INDEX PROBLEM: {thm_full_name} ({filepath})", flush=True)
            continue

        # ignore declarations that do not exist in the script
        if thm_decl_name is None:
            continue

        ### NOTE try-except block for debugging and to identify theorems that are not parsed
        try:
            thm_body = theorem_statement_proof(ref_file_str, thm_decl_name)
        except:
            print(f"PARSING PROBLEM: {thm_full_name} ({filepath})", flush=True)
            continue

        thm_statement = thm_body["statement"]
        ref_thm_proof = thm_body['proof']

        # ignore theorems with proofs that are empty
        if ref_thm_proof == "":
            continue

        for commit in commits_ls:

            # skip commits that newer than the reference commit
            if commit.committed_date > REF_COMMIT.committed_date:
                continue
                
            # retrieve the old file 
            old_file_str = file_str_from_commit(commit, filepath)

            # skip commit if old file is empty 
            if old_file_str is None:
                continue
            old_file_str = remove_comments(old_file_str)

            # skip history if the theorem statement has differed
            if thm_statement not in old_file_str:
                break
        
            old_thm_proof = theorem_statement_proof(old_file_str, thm_decl_name)["proof"]
            
            # skip commit if there has been no change to proof
            if ref_thm_proof == old_thm_proof:
                continue
            
            subbed_file_str = substituted_file(ref_file_str, old_thm_proof, thm_statement)

            # overwrite file with changes 
            with open(full_file_path, "w") as file:
                file.write(subbed_file_str)

            # attempt compiling file and append to dataset if it fails
            try:
                subprocess.run(["lake", "build", filepath_module],
                                    cwd=repo_copy_path,
                                    capture_output=True,
                                    text=True,
                                    check=True
                                )
            except subprocess.CalledProcessError as e:
                dataset["filepath"].append(filepath)
                dataset["thm_name"].append(thm_full_name)
                dataset["decl_name"].append(thm_decl_name)
                dataset["commit"].append(commit.hexsha)
                dataset["failed_proof"].append(old_thm_proof)
                dataset["error_msg"].append(e.stdout)

    # restore file to existing condition
    with open(full_file_path, "w") as file:
        file.write(original_file_str)

    return dataset


def construct_dataset(filename):
    """
    Constructs the proof repair dataset by attempting to compile an old version of a proof to a theorem.
    """
    print(f"Begin dataset construction for repo at commit {REF_COMMIT.hexsha}", flush=True)
    group_filepath = os.path.join(FILENAMES_DIR, f"{filename}.txt")
    with open(group_filepath, "r") as group_file:
        dir_paths = group_file.read().splitlines()

    # create a copy of the original repository 
    os.makedirs(REPO_COPY_DIR, exist_ok=True)
    repo_copy_path = os.path.join(REPO_COPY_DIR, filename)
    if not os.path.exists(repo_copy_path):
        create_repository_copy(repo_copy_path)
    
    lean_filepath_ls = get_all_lean_subfile_paths(REF_COMMIT, dir_paths)
    for lean_filepath in tqdm(lean_filepath_ls):
        
        # retrieve names of traced theorems
        
        thm_info_path = os.path.join(TRACED_INFO_DIR, f"{leanfile_replace_slash(lean_filepath, '_')}.json")
        with open(thm_info_path, "r") as file:
            all_thm_info = json.load(file)

        # generate dataset
        dataset = {
            "filepath": [],
            "thm_name": [],
            "decl_name": [],
            "commit": [],
            "failed_proof": [],
            "error_msg": [],
            }
        
        # dir_full_path = os.path.join(repo_copy_path, dir_path)
        dataset = scrape_file_history(dataset, lean_filepath, all_thm_info, repo_copy_path)
        save_data_to_json_in_dir(dataset, leanfile_replace_slash(lean_filepath, "_"), RAW_DATA_DIR)


    remove_repository_copy(repo_copy_path)
    os.remove(group_filepath)
    print("COMPLETED", flush=True)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Perform the data collection by scraping the Git repository of Lean 4 files.")
    parser.add_argument("-f", "--paths-filename", 
                        type=str,
                        required=True,
                        help="Name of the file that contians all the paths to collect the data from.")
    
    filename = parser.parse_args().paths_filename
    construct_dataset(filename)