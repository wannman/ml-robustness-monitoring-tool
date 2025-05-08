from typing import Optional
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from perturbation import apply_perturbation
from vectorizer import vectorize_data
from metrics import (
    mean_corruption_error,
    performance_drop_rate,
    relative_corruption_error,
    robustness_score,
    effective_robustness,
    our_metric,
    simple_mds,
    max_mds
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

    results = {"perturbation level": [], 
               "accuracy": [], 
               "RS": [],
               "mCE": [],
               "PDR": [],
               "MDS": [],
               "SimpleMDS": [],
               "MaxMDS": []
               }

    
    # Accuracy data used for calculating metrics
    accuracy_data = []
    

    for level in perturbation_levels:
        # Handle the case where file_path is None
        if file_path:
            load_path = file_path / f"perturbed_data_{level:.2f}.pkl"
        else:
            load_path = None

        # Apply perturbation #FUNKAR FÖR 0?
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

        # Store accuracy of each perturbation 
        
        if level == 0.0:
            base_accuracy = accuracy
            results["RS"].append(None)
            results["mCE"].append(None)
            results["PDR"].append(None)
            results["MDS"].append(None)
            results["SimpleMDS"].append(None)
            results["MaxMDS"].append(None)
        else:
            accuracy_data.append(accuracy)

        
    #base_accuracy = results["accuracy"][0]
    metrics_summary = {}



    
    # Add metric data to results
    # For MDS, we send all accuracy data and get a list of results at all possible perturbation levels
    # which we append to results
    mds_scores = []
    smds_scores = []
    max_mds_scores = []
    mds_scores = our_metric(base_accuracy, accuracy_data)
    smds_scores = simple_mds(base_accuracy, accuracy_data)
    max_mds_scores = max_mds(base_accuracy, accuracy_data)
    for i in range(len(accuracy_data)):
        results["RS"].append(robustness_score(base_accuracy, accuracy_data[i]) if "robustness_score" in metrics else None)
        results["mCE"].append(mean_corruption_error(base_accuracy, accuracy_data[i]) if "mce" in metrics else None)        
        results["PDR"].append(performance_drop_rate(base_accuracy, accuracy_data[i]) if "pdr" in metrics else None)
        # results["MDS"].append(our_metric(base_accuracy, accuracy_data[i]) if "our_metric" in metrics else None) 
        # Add MDS from calculated list instead
        results["MDS"].append(mds_scores[i] if "our_metric" in metrics else None)
        results["SimpleMDS"].append(smds_scores[i] if "simple_mds" in metrics else None)
        results["MaxMDS"].append(max_mds_scores[i] if "max_mds" in metrics else None)                
    


    # Summaries (incorporate above?)
    # if "mce" in metrics:
    #     metrics_summary["mce"] = mean_corruption_error(base_accuracy, results["accuracy"])
        
        
    # if "rce" in metrics:
    #     metrics_summary["rce"] = relative_corruption_error(base_accuracy, results["accuracy"])

    # if "robustness_score" in metrics:
    #     metrics_summary["robustness_score"] = robustness_score(base_accuracy, results["accuracy"])
        

    # if "effective_robustness" in metrics:
    #     metrics_summary["effective_robustness"] = effective_robustness(base_accuracy, results["accuracy"])

    # if "our_metric" in metrics:
    #     metrics_summary["our_metric"] = our_metric(base_accuracy, results["accuracy"])

    # if "base_accuracy" in metrics:
    #     metrics_summary["accuracy"] = base_accuracy
    metrics_summary = None

    return results, metrics_summary



