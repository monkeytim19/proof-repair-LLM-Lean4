import subprocess
import os
from pipeline.config import REPO_PATH, THEOREM_EXTRACTOR_DIR


def create_repository_copy(repo_copy_path, retrieve_cache=True):
    """Creates a copy of the reference repository and retrieves its cache for faster Lean 4 compilation."""
    print("STARTING: Lean repo copy creation", flush=True)
    try:
        subprocess.run(args=['cp', '-r', REPO_PATH, repo_copy_path], check=True) 
        print("DONE: Lean repo copy creation", flush=True)
    except:
        print("Failed to create Lean repo copy", flush=True)

    # retrieve cache that will be used for building lean later
    if retrieve_cache:
        print("STARTING: retrieve cache", flush=True)
        subprocess.run(
            args=["lake", "exe", "cache", "get!"], 
            cwd=repo_copy_path, 
            check=True 
        )
        print("DONE: retrieved cache", flush=True)


def add_lean_exe_to_lakefile(lean_exe_paths, lean_repo_path):
    """Adds the necessary .lean files as executables to the repository's lakefile.lean."""
    for path in lean_exe_paths:
        lakefile_path = os.path.join(lean_repo_path, 'lakefile.lean')
        # add to lakefile as new lean executable
        with open(path, 'r') as start_file, open(lakefile_path, 'a') as lakefile:
            text = start_file.read()
            lakefile.write('\n\n' + text)
        
    # add the library as a new directory in the repo copy
    try:
        new_path = os.path.join(lean_repo_path, "TheoremExtractor")
        subprocess.run(args=["cp", "-r", THEOREM_EXTRACTOR_DIR, new_path], check=True) 
    except:
        print("Failed to copy TheoremExtractor directory into the repo copy", flush=True)


def remove_repository_copy(repo_copy_path):
    """Removes the copy of the reference repository created."""
    print("STARTING: Lean repo copy removal", flush=True)
    try:
        subprocess.run(args=["rm", "-rf", repo_copy_path], check=True) 
        print("DONE: Lean repo copy removal", flush=True)
    except:
        print("Failed to remove Lean repo copy", flush=True)