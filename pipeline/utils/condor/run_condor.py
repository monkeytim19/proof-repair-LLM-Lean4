import subprocess
import os
import argparse
from pipeline.utils.theorem_extraction import file_num_theorem
from pipeline.utils.strings import is_lean_suffix
from pipline.utils.files import groups_by_thm_num
from pipeline.config import FILENAMES_DIR, DIRECTORIES_TO_SCRAPE

def trace_directory_condor(trace_dirs):
    """Submit a job to Condor for each directory that is given to trace its .lean file information."""
    batch_files_dir = os.path.join(os.getcwd(), "utils/condor/tracer")
    for dir in trace_dirs:
        # submit job to condor
        try:
            subprocess.run(
                args=["condor_submit", "-a", f"Dirname={dir}", "condor_file.cmd"], 
                cwd=batch_files_dir, 
                check=True,
                )
        except subprocess.CalledProcessError as e:
            print(e.stderr)
            print(e.stdout)


def scrape_dataset_condor(num_jobs):
    """
    Submit jobs to Condor to scrape the data from the designated Lean 4 repository. 
    
    The jobs will divide up all the .lean files in the repository based on the number of
    theorems in each file, such that each job should scrape the data for similar number of theorems. 
    (Note: this is not equivalent to similar computational load, as it is possible that different
    theorems may have different processing times depending on their Git history.)
    """
    batch_files_dir = os.path.join(os.getcwd(), "utils/condor/data_collection")
    groups = groups_by_thm_num(num_jobs, file_num_theorem(DIRECTORIES_TO_SCRAPE))

    for group in groups:
        # extract first item in the group to name files with
        if is_lean_suffix(group[0]):
            proxy_filename = group[0][:-5]
        else:
            proxy_filename = group[0]
        first_filename = first_filename.replace("/", "_")

        # create text file to record the file in each group
        os.makedirs(FILENAMES_DIR, exist_ok=True)
        group_filepath = os.path.join(FILENAMES_DIR, f"{proxy_filename}.txt")
        with open(group_filepath, "w+") as group_file:
            group_file.write("\n".join(group))

        # submit job to condor
        try:
            subprocess.run(
                args=["condor_submit", "-a", f"Filename={proxy_filename}", "condor_file.cmd"], 
                cwd=batch_files_dir, 
                check=True,
                )
        except subprocess.CalledProcessError as e:
            print(e.stderr)
            print(e.stdout)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the data collection or repository tracing process using Condor to batch jobs.")

    parser.add_argument("-m", "--mode",
                        type=str,
                        choices=["collect-data", "trace-directory"],
                        required=True,
                        help="Mode to batch the Condor jobs for.")

    parser.add_argument("-n", "--num-jobs", 
                        type=int,
                        default=50,
                        help="Number of parallel jobs to be batched to Condor.")
    
    args = parser.parse_args()

    if args.mode == "collect-data":
        if args.num_jobs > 0:
            scrape_dataset_condor(args.num_jobs)
        else:
            print("Invalid input for '-n'. Must be positive.")
    else:
        trace_directory_condor(DIRECTORIES_TO_SCRAPE)
