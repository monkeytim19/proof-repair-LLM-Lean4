import pandas as pd
import os
import json
import argparse
import subprocess
from datetime import datetime
from pipeline.config import REPO_COPY_DIR
from pipeline.utils.strings import leanfile_replace_slash, remove_comments
from pipeline.utils.lean_repo_copying import create_repository_copy, remove_repository_copy
from pipeline.utils.theorem_extraction import substituted_file


def print_progress(curr_count, total_count, datapoint, proof_col):
    """Prints the progress and detail of the current proof being verified."""
    print(f"---{curr_count}/{total_count}---", flush=True)
    print(f"For theorem {datapoint['thm_name']} in {datapoint['filepath']}:\n{datapoint['statement']}\n\n", flush=True)
    print(f"Original proof:\n{datapoint['proof']})\n\n", flush=True)
    print(f"Verifying proof (Commit):\n{datapoint[proof_col]}\n", flush=True)


def success_rate(counts):
    """Computes the overall success rate given the counts of separate sucesses and failures."""
    success_counts = sum(counts["success"].values())
    failure_counts = sum(counts["failure"].values())
    return success_counts / (failure_counts + success_counts)


def verify_proof(proof_df, proof_col, verbose, repo_copy_name): 
    """
    Verifies whether each of the proofs provided is a valid proof to their corresponding theorem
    and Lean 4 environment. Returns a dictionary that includes the indices of the theorems from the dataset,
    grouped by a successful or failed proof attempt.
    
    A proof is verified by replacing the existing proof and checking if the overall .lean file compiles
    under the same environment (i.e. same namespaces, imports, declarations, etc.)
    """
    if not os.path.exists(REPO_COPY_DIR):
        os.makedirs(REPO_COPY_DIR, exist_ok=True)
    repo_copy_path = os.path.join(REPO_COPY_DIR, repo_copy_name)
    if not os.path.exists(repo_copy_path):
        create_repository_copy(repo_copy_path)

    verification_outcomes = {"success": [], "failure": []}     

    if verbose:
        curr_count, total_count = 0, len(proof_df)
    
    for leanfile_path, leanfile_df in proof_df.groupby("filepath"):
        if verbose:
            print(f"Working on {leanfile_path} - {datetime.now()}.", flush=True)
        copy_leanfile_path = os.path.join(repo_copy_path, leanfile_path)

        with open(copy_leanfile_path, 'r') as leanfile:
            ref_file_str = leanfile.read()

        for idx, datapoint in leanfile_df.iterrows():

            if verbose:
                curr_count += 1
                print_progress(curr_count, total_count, datapoint, proof_col)

            test_file_str = remove_comments(ref_file_str)
            test_file_str = substituted_file(test_file_str, datapoint[proof_col], datapoint["statement"])

            # write the file with the model repaired proof to the file
            with open(copy_leanfile_path, 'w') as leanfile:
                leanfile.write(test_file_str)

            # attempt to compile the code in lean
            filepath_module = leanfile_replace_slash(leanfile_path, ".")

            # attempt to compile file
            try:
                subprocess.run(["lake", "build", filepath_module],
                                    cwd=repo_copy_path,
                                    capture_output=True,
                                    text=True,
                                    check=True
                                )
            except subprocess.CalledProcessError:
                outcome = "failure"
            else:
                outcome = "success"
            verification_outcomes[outcome].append(idx)

            if verbose:
                print(f"Attempt {outcome}.\n", flush=True)

        # restore the file
        with open(copy_leanfile_path, "w") as leanfile:
            leanfile.write(ref_file_str)

    remove_repository_copy(repo_copy_path)
    return verification_outcomes

    
if __name__ == "__main__":

    # e.g. nohup python verify.py -d proof_verification/test_prediction.csv -v > verify.out 2>&1
    parser = argparse.ArgumentParser(description="Verify the proof attempts and compile with Lean 4.")

    parser.add_argument("-d", "--data-path", 
                        type=str,
                        required=True,
                        help="Path to the .csv file that holds the data to the theorems and the proofs.")
    
    parser.add_argument("-c", "--verify-proof-column", 
                        type=str,
                        default="predicted_proof",
                        help="Name of the column in the .csv dataset that holds the proofs to be verified.")
    
    parser.add_argument("-i", "--index-data",
                        type=str,
                        help="Path to a .json file that contains a list of indices of the dataset such that only this subset of the dataset is verified.")

    parser.add_argument("-s", "--save-results", 
                        action="store_true",
                        help="Save the outcome for each theorems from the verification with their index in the dataset.")
    
    parser.add_argument("-n", "--save-name", 
                        type=str,
                        help="Name for the .json files that records the dataset indices of the theorems by their verification outcome.")
    
    parser.add_argument("-r", "--run-num", 
                        type=str,
                        help="Run number for the verification runs.")
    
    parser.add_argument("-v", "--verbose", 
                        action="store_true",
                        help="Display the progress of the proof verification and compilation process.")

    args = parser.parse_args()

    if args.save_results and not args.save_name:
        parser.error("The '-n' argument is required when '-s' is provided.")

    proofs_df = pd.read_csv(args.data_path)
    
    if args.index_data:
        # index_file_path = os.path.join(os.getcwd(), args.index_data)
        with open(args.index_data, "r") as file:
            wanted_indices = json.load(file)
        proofs_df = proofs_df[proofs_df.index.isin(wanted_indices)]
    
    # check if the column used for verifying the proofs are all strings
    # if not proofs_df[args.verify_proof_column].apply(lambda x: isinstance(x, str)).all():
    #     parser.error("The value provided to the '-c' argument points to a column in the dataset that does not contain all strings and is not a valid column to be used for proof verification.")

    print(f"STARTING: verification of proofs from {args.verify_proof_column} column in dataset from {args.data_path} - {datetime.now()}")
    repo_copy_name = f"verification_{args.run_num}" if args.run_num is not None else "verification"
    verify_counts = verify_proof(proofs_df, args.verify_proof_column, args.verbose, repo_copy_name)
    success_counts, failure_counts = len(verify_counts["success"]), len(verify_counts["failure"])
    print(f"Among {len(proofs_df)} proof attempts, there were {success_counts} sucessful and {failure_counts} failed attempts at proving their respect theorems.", flush=True)
    print(f"The rate of successful proof = {success_counts/(success_counts+failure_counts)}.", flush=True)

    # save the results
    if args.save_results:
        for outcome in verify_counts.keys():
            outcome_dir_path = os.path.join(os.getcwd(), f"pipeline/verifier/{outcome}")
            os.makedirs(outcome_dir_path, exist_ok=True)
            save_filepath = os.path.join(outcome_dir_path, args.save_name)
            with open(save_filepath, 'w') as file:
                json.dump(verify_counts[outcome], file)