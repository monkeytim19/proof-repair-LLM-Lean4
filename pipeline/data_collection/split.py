import pandas as pd
import os
import argparse
import random
from pipeline.utils.grouping import group_keys_by_value
from pipeline.config import SEED, DATA_DIR


def filter_dataframe(df, tuple_ls, columns):
    """
    Filters a DataFrame based on a list of tuples that contains values for specified columns.
    """
    pairs_df = pd.DataFrame(tuple_ls, columns=columns)
    return pd.merge(df, pairs_df, on=columns)


def split_data(df, num_valid, num_test):
    """
    Splits the dataframe into train, validation, and test sets based on the desired number
    of validation and test samples.

    The spliting involves randomly shuffling the data but will group the instances from the same
    theorem together into the same set to avoid overlap in data between splits (as it is probable 
    that the same theorems may have similar problems and breakages).
    """
    thm_id_cols = ["filepath", "thm_name"]
    thm_name_df = df.groupby(thm_id_cols)
    unique_thm_counts = thm_name_df.size()

    # shuffle the pairs randomly
    key_value_pairs = list(zip(unique_thm_counts.index, unique_thm_counts.values))
    random.seed(SEED)
    random.shuffle(key_value_pairs)
    unique_thms, thm_counts = zip(*key_value_pairs)
    unique_thms, thm_counts = list(unique_thms), list(thm_counts)

    test_thms, unique_thms, thm_counts = group_keys_by_value(num_test, unique_thms, thm_counts)
    valid_thms, train_thms, _ = group_keys_by_value(num_valid, unique_thms, thm_counts)
    
    train_df = filter_dataframe(df, train_thms, thm_id_cols)
    valid_df = filter_dataframe(df, valid_thms, thm_id_cols)
    test_df = filter_dataframe(df, test_thms, thm_id_cols)

    if len(train_df) + len(valid_df) + len(test_df) != len(df):
        print("Problem with data spliting - require further investigation into the spliting algorithm.")

    return train_df, valid_df, test_df


def save_splits(train_df, valid_df, test_df, data_dir):
    """Save the data splits as separate .csv files in the designated directory for processed data."""
    # concatenate the dataframes to create larger subsets of the dataset
    train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)
    valid_test_df = pd.concat([valid_df, test_df], ignore_index=True)

    train_path = os.path.join(data_dir, "train.csv")
    valid_path = os.path.join(data_dir, "valid.csv")
    test_path =  os.path.join(data_dir, "test.csv")
    train_valid_path = os.path.join(data_dir, "train_valid.csv")
    valid_test_path = os.path.join(data_dir, "valid_test.csv")

    # save to .csv files
    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)
    train_valid_df.to_csv(train_valid_path, index=False)
    valid_test_df.to_csv(valid_test_path, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Split the dataset into train, valid, and test splits.")
    parser.add_argument("-d", "--data-file", type=str, required=True, help="Name of the .csv file containing the dataset.")
    parser.add_argument("-v", "--num-valid", type=int, default=2000, help="Number of data points in the validation set.")
    parser.add_argument("-t", "--num-test", type=int, default=2000, help="Number of data points in the test set.")
    args = parser.parse_args()
    
    num_valid, num_test = args.num_valid, args.num_test
        
    data_path = os.path.join(DATA_DIR, args.data_file)
    df = pd.read_csv(data_path)
    if num_valid > 0 and num_test > 0 and len(df)-num_valid-num_test > 0:
        train_df, valid_df, test_df = split_data(df, num_valid, num_test)
        save_splits(train_df, valid_df, test_df, DATA_DIR)
        print(f"Number of training data = {len(train_df)}.")
        print(f"Number of validation data = {len(valid_df)}.")
        print(f"Number of test data = {len(test_df)}.")
    else:
        print("Invalid number of data points for the validation and test set.")