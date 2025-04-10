import numpy as np
from sklearn.base import BaseEstimator, accuracy_score
from perturbation import apply_perturbation
from vectorizer import vectorize_data 
from metrics import (
    mean_corruption_error,
    relative_corruption_error,
    robustness_score,
    effective_robustness,
    our_metric
) 

def evaluate_robustness(
        model: BaseEstimator, 
        vectorizer: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        metrics: list[str],
        perturbation_levels: list[float]
    ):

    results = {"perturbation level": [], "accuracy": []}

    for level in perturbation_levels:
        X_perturbed_data = apply_perturbation(X, level)
        X_perturbed_vect, _ = vectorize_data(X_perturbed_data, X, vectorizer, True)

        y_pred = model.predict(X_perturbed_vect)
        accuracy = accuracy_score(y, y_pred)

        results["perturbation level"].append(level)
        results["accuracy"].append(accuracy)


    base_accuracy = results["accuracy"][0]
    metrics_summary = {}

    if "mce" in metrics:
        metrics_summary["mce"] = mean_corruption_error(base_accuracy, results["accuracy"])
        
    if "rce" in metrics:
        metrics_summary["rce"] = relative_corruption_error(base_accuracy, results["accuracy"])

    if "robustness_score" in metrics:
        metrics_summary["robustness_score"] = robustness_score(base_accuracy,results["accuracy"])

    if "effective_robustness" in metrics:
        metrics_summary["effective_robustness"] = effective_robustness(base_accuracy,results["accuracy"])

    if "our_metric" in metrics:
        metrics_summary["our_metric"] = our_metric(base_accuracy, results["accuracy"])

    if "base_accuracy" in metrics:
        metrics_summary["accuracy"] = base_accuracy

    return results, metrics_summary 





        
