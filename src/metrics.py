
# MCE needs to be changed - may take other parameter input
def mean_corruption_error(perturbed_accuracies: list) -> list:
        
    mce_values = []
    error_rates = []
    
    for i, acc in enumerate(perturbed_accuracies, start=1):
            error = 1.0 - acc
            error_rates.append(error)
            mce = sum(error_rates) / i
            mce_values.append(mce)

    return mce_values
    
     

def robustness_score(base_accuracy: float, perturbed_accuracies: list) -> float:
    return perturbed_accuracies/base_accuracy

def performance_drop_rate(base_accuracy: float, perturbed_accuracies: list) -> float:
    return (base_accuracy-perturbed_accuracies)/base_accuracy

def mean_divergence_score(base_accuracy: float, perturbed_accuracies: list, epsilon: float = 0.05) -> list:
    
    mds_values = [None]  # MDS is undefined for n = 1

    for n in range(1, len(perturbed_accuracies)):
        acc_prev = perturbed_accuracies[n - 1]
        acc_curr = perturbed_accuracies[n]

        if acc_curr < acc_prev:
            avg_prev = sum(perturbed_accuracies[:n]) / n
            denom = max(base_accuracy - avg_prev, epsilon)
            mds = (acc_prev - acc_curr) / denom
        else:
            mds = 0.0

        mds_values.append(mds)

    return mds_values
        
 

