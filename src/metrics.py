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


def simple_mds(base_accuracy: float, perturbed_accuracies: list) -> list:
    smds_list = []

    for i in range(len(perturbed_accuracies)): 
        #  Undefined at first perturbed value
        if i == 0:
            smds_list.append(None)
            continue

        # acc(n)-acc(n-1) (not same as original MDS)
        current_diff = perturbed_accuracies[i]-perturbed_accuracies[i-1]

        # acc(n-1)-acc(baseline)
        prev_to_baseline_diff = perturbed_accuracies[i-1]-base_accuracy

        smds_list.append(current_diff/prev_to_baseline_diff)
        
    return smds_list

def max_mds(base_accuracy: float, perturbed_accuracies: list) -> list:
    max_mds_list = []

    for i in range(len(perturbed_accuracies)): 
        #  Undefined at first perturbed value
        if i == 0:
            max_mds_list.append(None)
            continue

        # acc(n-1)-acc(n) (same as MDS, not same as SMDS)
        current_diff = perturbed_accuracies[i-1]-perturbed_accuracies[i]

        max_mds_list.append(max(0, (current_diff/base_accuracy)))
        
    return max_mds_list

# returns all scores at once, where applicable
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

def new_mds(base_accuracy: float, perturbed_accuracies: list, epsilon: float = 0.05) -> list:
    
    mds_values = []

    for i in range(len(perturbed_accuracies)):
        if i == 0:
            mds_values.append(None)
            continue

        current = perturbed_accuracies[i]
        prev = perturbed_accuracies[i - 1]

        if current < prev:
            prev_avg = sum(perturbed_accuracies[:i]) / i
            denom = max(base_accuracy - prev_avg, epsilon)
            mds_score  = (prev - current) / denom
        else:
            mds_score = 0.0

        mds_values.append(mds_score)

    return mds_values
        
 

