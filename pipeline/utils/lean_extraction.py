import subprocess   


def print_theorem_expr(thm_name, module_path, repo_copy_path):
    """Returns the statement and proof of the theorems declared in the specified module and the status of the extraction."""
    try:
        result = subprocess.run(["lake", "build", module_path],
                            cwd=repo_copy_path,
                            capture_output=True,
                            text=True,
                            check=True
                        )
    # check if there is error in the extraction
    except subprocess.CalledProcessError as e:
        if "unknown constant" in e.stdout:
            print(f"NOT PRESENT IN FILE: {thm_name} from {module_path}")
            status = "missing"
        else:
            print(f"EXTRACTION ERROR: {thm_name} from {module_path}")
            status = "extraction_error"
        return "", status
    
    # split the result into statement and proof
    start_idx = result.stdout.find("theorem")
    result_str = result.stdout[start_idx:]
    result_ls = result_str.splitlines()
    result_ls.pop()
    result_str = "\n".join(result_ls)

    # check if part of the extracted statement or proof has been omitted
    if "⋯" in result_str:
        print(f"STATEMENT/PROOF EXCEEDED MAX LENGTH: {thm_name} from {module_path}")
        status = "exceed_len"
        return "", status

    return result_str, ""
            

def get_theorem_names(module_path, repo_copy_path):
    """Retreives the list of the names of all theorems declared in the specified module."""
    # execute the TheroemExtractor library
    try:
        print(f"Begin to retrieve theorem name from {module_path}", flush=True)
        result = subprocess.run(["lake", "exe", "extract_file_info", module_path],
                            cwd=repo_copy_path,
                            capture_output=True,
                            text=True,
                            check=True
                        )
    except subprocess.CalledProcessError:
        print(f"Failed to retrieve theorem names from {module_path}")

    return result.stdout.splitlines()


def get_theorem_position(thm_name, module_path, repo_copy_path, selection_range=False):
    """
    Extracts the position or the selection range of the theorem from the .lean file it is in."""
    # execute the TheroemExtractor library
    flag = "-selection-range" if selection_range else "-position"
    try:
        result = subprocess.run(["lake", "exe", "extract_single_thm", flag, thm_name, module_path],
                            cwd=repo_copy_path,
                            capture_output=True,
                            text=True,
                            check=True
                        )
    except subprocess.CalledProcessError:
        print(f"Failed to retrieve theorem position from {module_path} for {thm_name}")

    positions = result.stdout.split("&")
    positions[-1] = positions[-1][:-1] if positions[-1][-1:] == "\n" else positions[-1]
    return {"start": positions[0], "end": positions[-1]}


def get_decl_name(file_str, start_pos, end_pos, wanted_decl_keywords=["theorem", "lemma"]):
    """Returns the local name of the decleration given within the .lean file and namespaces."""
    row = start_pos[0] - 1
    start_col, end_col = start_pos[-1], end_pos[-1]
    
    # only return the name for declaration that is theorem or lemma
    decl_keyword = file_str.splitlines()[row][:start_col-1]
    if decl_keyword in wanted_decl_keywords:
        return file_str.splitlines()[row][start_col: end_col]
    return