import pandas as pd
import os
import re
import argparse
from pipeline.config import DATA_DIR, RAW_DATA_DIR, REF_COMMIT
from pipeline.utils.theorem_extraction import file_theorem_info
from pipeline.verifier.verify import verify_proof


def remove_error_msg_redundancies(row):
    """Removes the unnecessary parts of the error message."""
    msg_by_lines = row["error_msg"].splitlines()

    for idx, line in enumerate(msg_by_lines):
        # find the index of the line starting with 'trace: '
        if line.strip().startswith("trace:"):
            start_idx = idx + 1
    
        # Find the index of the line 'Some builds logged failures:'
        if line == "error: Lean exited with code 1":
            end_idx = idx
            break
    
    # extract lines between the two points
    extracted_lines = "\n".join(msg_by_lines[start_idx: end_idx])

    # remove redundant text on the path of file
    regex_pattern = r"\S*"
    regex_pattern += row["filepath"]
    regex_pattern += r"\S*"
    
    # Use re.sub to replace matching substrings with an empty string
    extracted_lines = re.sub(regex_pattern, "", extracted_lines)
    return extracted_lines


def get_ground_truth_info(df):
    """
    Obtain the dictionary of the ground truth information on the theorems in the given dataset.

    This contains the correct theorem statement and proof from the reference commit.
    """
    thm_info_dict = {}
    for filepath in df["filepath"].unique():
        thm_names_ls = df["decl_name"][df["filepath"] == filepath]
        fp_dict = file_theorem_info(REF_COMMIT, filepath, thm_names_ls)
        thm_info_dict = thm_info_dict | fp_dict
    return thm_info_dict


def query_ground_truth(row, ground_truth_table):
    """
    Wrapper function to help index and determine the ground truth for a given row in the dataset.
    """
    return pd.Series(ground_truth_table[(row["filepath"], row["decl_name"])])


def load_raw_dataset():
    """Loads in all the raw data and concatenates them into a single pd.DataFrame object."""
    # load in json files
    json_files = [f for f in os.listdir(RAW_DATA_DIR)]
    dataframes = []
    for json_file in json_files:
        file_path = os.path.join(RAW_DATA_DIR, json_file)
        df = pd.read_json(file_path)
        dataframes.append(df)

    # Concatenate all dataframes into a single dataframe
    return pd.concat(dataframes, ignore_index=True)


def main(datafile, to_verify):
    """Pre-process the dataset and save it to desired path."""
    df = load_raw_dataset()
    raw_n = len(df)
    print(f"Beginning pre-processing with {raw_n} raw data points.")

    # remove any rows that may contain empty values
    df = df[df["error_msg"] != ""]
    df = df[df["failed_proof"] != ""]

    # append the ground truth information to the dataset
    ground_truth_table = get_ground_truth_info(df)
    df[["statement", "proof"]] = df.apply(lambda row: query_ground_truth(row, ground_truth_table), axis=1)

    # remove additional rows that contain empty values
    df = df[df["proof"] != ""]
    df = df[df["statement"] != ""]
    print(f"Removed {raw_n - len(df)} data points from containing empty values.")

    # remove duplicate rows
    raw_n = len(df)
    df = df.drop_duplicates(subset=["filepath", "thm_name", "failed_proof"])
    print(f"Removed {raw_n - len(df)} data points from duplicacy.")

    # perform verification on reference proofs
    if to_verify:
        print("STARTING: Verifying proofs.")
        verify_counts = verify_proof(df, proof_col="proof", verbose=False, repo_copy_name="preprocessing_verify")
        df = df[df.index.isin(verify_counts["success"])]
        print(f"Removed {len(verify_counts['failure'])} invalid data points after verification.")

    # remove redundancies in error message
    df["error_msg"] = df.apply(remove_error_msg_redundancies, axis=1)

    # re-order dataset by filepath and theorem name
    df = df.sort_values(by=["filepath", "thm_name", "commit"]).reset_index(drop=True)

    print(f"{len(df)} data points remaining after pre-processing.")
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, datafile)
    df.to_csv(filepath, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the raw data from the data collection and concatenate them into a single dataset.")

    parser.add_argument("-d", "--data-file", type=str, default="proof_repair_dataset.csv", help="Name of the .csv file containing the dataset. Defaulted to 'proof_repair_dataset.csv'.")
    parser.add_argument("-v", "--verify", action="store_true", help="Performs verification on the extracted reference proofs as part of the preprocessing. (Warning: may take a long time)")

    datafile = parser.parse_args().data_file
    to_verify = parser.parse_args().verify
    main(datafile, to_verify)