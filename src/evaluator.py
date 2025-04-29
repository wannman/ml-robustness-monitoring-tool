from typing import Optional
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
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
        X: list[np.ndarray],
        y: np.ndarray,
        perturbation_levels: list[float],
        metrics: list[str],
        file_path: Optional[Path] = None,
    ):

    results = {"perturbation level": [], "accuracy": []}

    for level in perturbation_levels:
        # Handle the case where file_path is None
        if file_path:
            load_path = file_path / f"perturbed_data_{level:.2f}.pkl"
        else:
            load_path = None

        # Apply perturbation
        X_perturbed_data = apply_perturbation(X, level, load_path=load_path)

        # print(f"\nPerturbation level: {level}")
        # for original, changed in zip(X, X_perturbed_data):
        #     print(f"ORIGINAL: {original}")
        #     print(f"PERTURBED: {changed}")

        # Vectorize the perturbed data
        X_perturbed_vect = vectorize_data(vectorizer, X_perturbed_data)

        # Predict and calculate accuracy
        y_pred = model.predict(X_perturbed_vect)
        accuracy = accuracy_score(y, y_pred)

        # Store results
        results["perturbation level"].append(level)
        results["accuracy"].append(accuracy)

    base_accuracy = results["accuracy"][0]
    metrics_summary = {}

    if "mce" in metrics:
        metrics_summary["mce"] = mean_corruption_error(base_accuracy, results["accuracy"])
        
    if "rce" in metrics:
        metrics_summary["rce"] = relative_corruption_error(base_accuracy, results["accuracy"])

    if "robustness_score" in metrics:
        metrics_summary["robustness_score"] = robustness_score(base_accuracy, results["accuracy"])

    if "effective_robustness" in metrics:
        metrics_summary["effective_robustness"] = effective_robustness(base_accuracy, results["accuracy"])

    if "our_metric" in metrics:
        metrics_summary["our_metric"] = our_metric(base_accuracy, results["accuracy"])

    if "base_accuracy" in metrics:
        metrics_summary["accuracy"] = base_accuracy

    return results, metrics_summary






