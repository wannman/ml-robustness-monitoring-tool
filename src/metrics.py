import numpy as np

#


# MCE needs to be changed - may take other parameter input
def mean_corruption_error(base_accuracy: float, perturbed_accuracies: list) -> float:
    return 0.0


def relative_corruption_error(base_accuracy: float, perturbed_accuracies: list) -> float:
    return 0.0

def robustness_score(base_accuracy: float, perturbed_accuracies: list) -> float:
    return perturbed_accuracies/base_accuracy



def effective_robustness(base_accuracy: float, perturbed_accuracies: list) -> float:
    return 1.1

def performance_drop_rate(base_accuracy: float, perturbed_accuracies: list) -> float:
    return (base_accuracy-perturbed_accuracies)/base_accuracy


# returns all scores at ones, where applicable
def our_metric(base_accuracy: float, perturbed_accuracies: list) -> list:
    mds_list = []
    diff_list = []

    for i in range(len(perturbed_accuracies)): 
        # Calculate differences
        # No diff or MDS for first value
        if i == 0:
            mds_list.append(None)
            continue

        # positive difference between this value and previous value
        current_diff = perturbed_accuracies[i-1]-perturbed_accuracies[i]
        
        # First diff calculated, but no MDS score here
        if i == 1:
            mds_list.append(None)
            diff_list.append(current_diff)
        else:
            average_diff = sum(diff_list)/len(diff_list)
            mds_list.append(current_diff/average_diff if average_diff > 0 else None)
            # add current difference to list of differences
            diff_list.append(current_diff)
        
    return mds_list
        
 

