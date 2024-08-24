import numpy as np


def groups_by_thm_num(n_groups, file_num_thm):
    """
    Organizes the files into a specific number of groups such that the sum of the number of theorems
    defined in all the files in each group is roughly the same.
    
    This uses a greedy number partitioning method. 
    """
    file_num_thm = sorted(file_num_thm, key=lambda x: x[1], reverse=True)
    # initialize empty groups
    groups = [[] for _ in range(n_groups)]
    group_sizes = [0] * n_groups

    # assign items to groups
    for item, size in file_num_thm:
        # find the group with the smallest current total size
        min_index = min(range(n_groups), key=lambda x: group_sizes[x])
        # assign the item to that group
        groups[min_index].append(item)
        group_sizes[min_index] += size
    
    return groups


def group_keys_by_value(target_n, keys, values):
    """
    Returns a subset of keys such that the sum of their corresponding values is as close as possible to a given target value.

    The list of keys and values inputted to the function will be mutated.
    """
    curr_n = 0
    target_keys = []
    while curr_n < target_n:
        if curr_n + max(values) < target_n:
            idx = np.argmax(values)
        elif curr_n + min(values) >= target_n:
            idx = np.argmin(values)
        else:
            abs_diffs = np.abs([x + curr_n - target_n for x in values])
            idx = np.argmin(abs_diffs)
        target_keys.append(keys.pop(idx))
        curr_n += values.pop(idx)
    return target_keys, keys, values