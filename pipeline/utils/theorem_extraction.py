import re
from git import Tree
from utils.git_extraction import blob_file_str, file_str_from_commit, get_all_blobs
from utils.strings import remove_comments, theorem_body_regex, theorem_proof_regex, theorem_name_regex
from config import REF_COMMIT


def theorem_names_from_blob(blob):
    """Returns all the theorem names within a single file/git.Blob object."""
    # find all matches in file
    file_str = remove_comments(blob_file_str(blob))
    theorem_names = list(theorem_name_regex().findall(file_str))
    # return list of theorem names from the matches
    return theorem_names
    

def theorem_body(file_str, theorem_name):
    """Returns the body of the theorem of interest from the specified file of text."""
    thm_body_regex = theorem_body_regex(theorem_name)
    thm_body = re.search(thm_body_regex, file_str, re.DOTALL).group()
    return thm_body


def theorem_statement_proof(file_str, theorem_name):
    """
    Returns the statement and proof of the theorem of interest as a dictionary. 
    """
    body = theorem_body(file_str, theorem_name).splitlines()
    statement = [body.pop(0)]
    proof = body

    # identify the line where the theorem statement ends
    two_spaces_regex = r"^\s{2}(?!\s)"
    while body:
        line = body[0]
        if not re.match(two_spaces_regex, line):
            statement.append(body.pop(0))
        else:
            break 

    # identify the exact point where the theorem proof begins
    idx = statement[-1].rfind(":=")
    if idx >= 0:
        proof = [statement[-1][idx:]] + proof
        statement[-1] = statement[-1][:idx]

    return {"statement": "\n".join(statement), "proof": "\n".join(proof)}


def all_theorem_names(commit, dir_path):
    """
    Returns the list of all theorem names that have been defined within a file in the 
    specified directory path during the specific commit. 
    """
    # designate the specific directory of mathlib4
    tree = commit.tree[dir_path]

    # retrieve all the files/blobs within the directory
    if isinstance(tree, Tree):
        all_blobs = get_all_blobs(tree)
    else:
        all_blobs = [tree]

    # retrieve all the names of the theorems from all the files/blobs
    all_theorem_names = {blob.path: theorem_names_from_blob(blob) for blob in all_blobs}
    return all_theorem_names


def file_theorem_info(commit, file_path, thm_names=None):
    """
    Returns a dictionary that contains the statement, proof information of 
    all theorems within the specified file. 
    """
    if thm_names is None:
        file_theorem_names = all_theorem_names(commit, file_path)[file_path]
    else:
        file_theorem_names = thm_names

    # extract all the text from specific file and remove all comments/annotations
    file_str = file_str_from_commit(commit, file_path)
    file_str = remove_comments(file_str)

    thm_info = {}
    for name in file_theorem_names:
        thm_info[(file_path, name)] = theorem_statement_proof(file_str, name)

    # return a dictionary of (name, body) pairs of theorems in the file
    return thm_info


def substituted_file(to_replace_file_str, old_thm_proof, thm_statement):
    """
    Return the most recent file that has the proof of the 
    theorem of interest replaced with its version from an older commit.
    """
    thm_proof_regex = theorem_proof_regex(thm_statement)
    replaced_file_str = re.sub(thm_proof_regex, old_thm_proof, to_replace_file_str, flags=re.DOTALL)
    return replaced_file_str


def file_num_theorem(directories):
    """
    Returns a list of the files that is to be scraped and the number of theorems in each of them respectively.
    """
    files_num_thm = []
    for sub_dir in directories:
        for blob in get_all_blobs(REF_COMMIT.tree[sub_dir]):
            num_thm = len(theorem_names_from_blob(blob))
            if num_thm > 0:
                files_num_thm.append((blob.path, num_thm))
    return files_num_thm