import os
import argparse
from pipeline.utils.strings import leanfile_replace_slash, is_lean_suffix
from pipeline.utils.git_extraction import get_all_blobs
from pipeline.utils.lean_repo_copying import add_lean_exe_to_lakefile, create_repository_copy
from pipeline.utils.files import save_data_to_json_in_dir
from pipeline.utils.lean_extraction import get_theorem_names
from config import REPO_COPY_DIR, REF_COMMIT, THEOREM_EXTRACTOR_DIR, TRACED_INFO_DIR


def add_theorem_extractor_to_repo(repo_copy_path):
    """
    Creates a copy of the reference Lean project repository and adds the Theorem Extractor Lean library to it.
    """
    # create copy of source lean repo
    create_repository_copy(repo_copy_path)

    # add lake_exe to repo copy
    whole_file_lean_exe_path = os.path.join(THEOREM_EXTRACTOR_DIR, "WholeFile/lean_exe.txt")
    individual_thm_lean_exe_path = os.path.join(THEOREM_EXTRACTOR_DIR, "IndividualTheorem/lean_exe.txt")
    lean_exe_paths = [whole_file_lean_exe_path, individual_thm_lean_exe_path]
    add_lean_exe_to_lakefile(lean_exe_paths, repo_copy_path)


def trace_directory(dir_path, repo_copy_path):
    """
    Traces a directory of .lean files within a lean project with the TheoremExtractor module added to its lakefile.lean file.

    Returns a dictionary that contains all the names, selection ranges of the theorems that are defined in the .lean
    files declared within the specified directory.

    Args:
        dir_path (str): The relative path of the directory that needs to be traced within the repository of the Lean project.
        repo_copy_path (str): The path of the directory of the copied Lean repo/project.
    """
    for blob in get_all_blobs(REF_COMMIT.tree[dir_path]):
        module_path = blob.path
        # remove the suffix and make appropriate changes to the string
        if not is_lean_suffix(module_path):
            continue
        module_path = leanfile_replace_slash(module_path, ".")

        # retrieve theorem names and save to file
        thm_names_info = get_theorem_names(module_path, repo_copy_path)
        thm_names = []
        for name_info in thm_names_info:
            thm_names_dict = {}
            thm_names_dict["name"], thm_names_dict["start"], thm_names_dict["end"] = name_info.split("&")
            thm_names.append(thm_names_dict)
            
        # save the extracted theorem names to .json file
        save_data_to_json_in_dir(thm_names, module_path.replace(".", "_"), TRACED_INFO_DIR)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Trace a directory within a Lean 4 repository to collect information about theorems within it.")
    parser.add_argument("-d", "--dir-path", 
                        type=str,
                        required=True,
                        help="Relative path of the directory to be traced, relative to the directory of the Lean 4 repository.")

    dir_path = parser.parse_args().dir_path
    os.makedirs(REPO_COPY_DIR, exist_ok=True)
    repo_copy_path = os.path.join(REPO_COPY_DIR, dir_path)

    if not os.path.exists(repo_copy_path):
        add_theorem_extractor_to_repo(repo_copy_path)
    trace_directory(dir_path, repo_copy_path)