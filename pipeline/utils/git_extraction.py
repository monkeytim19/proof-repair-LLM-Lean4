from git import Tree
from utils.strings import is_lean_suffix
from config import REPO


def file_commits(filepath=None):
    """
    Returns the list of all commits from all branches in the repository relevant to the desired filepath.
    
    If no filepath is specified, then the entire repository will be considered.
    """
    if filepath is None:
        return list(REPO.iter_commits(all=True))
    else:
        return list(REPO.iter_commits(all=True, paths=filepath))


def get_all_blobs(tree):
    """Returns all blobs in a git.Tree object."""
    all_blobs = tree.blobs
    for subtree in tree.trees:
        all_blobs += get_all_blobs(subtree)
    return all_blobs


def get_all_lean_subfile_paths(commit, dir_paths_ls):
    """Returns a list of all paths to .lean files within a directory from the given list of directory paths."""
    lean_file_paths = []
    for dir_path in dir_paths_ls:

        # check if directory exists within the repo
        try:
            tree = commit.tree[dir_path]
        except KeyError:
            print(f"{dir_path} not found in repo.", flush=True)
            continue
        
        # check if object is a directory or file
        if isinstance(tree, Tree):
            lean_file_paths += [blob.path for blob in get_all_blobs(tree) if is_lean_suffix(blob.path)]
        else:
            if is_lean_suffix(tree.path):
                lean_file_paths.append(tree.path)
    return lean_file_paths
    

def blob_file_str(blob):
    """Returns the entire string of the file/git.Blob object."""
    return blob.data_stream.read().decode()


def file_str_from_commit(commit, file_path):
    """Returns the file from a specific file path and commit."""
    try:
        blob = commit.tree[file_path]
        return blob_file_str(blob)
    except KeyError:
        return