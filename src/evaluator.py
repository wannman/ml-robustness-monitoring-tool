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
    robustness_score,
    mean_divergence_score
)

def evaluate_robustness(
    model: BaseEstimator,
    vectorizer: BaseEstimator,
    X: list[np.ndarray],
    y: np.ndarray,
    perturbation_levels: list[float],
    metrics: list[str],
    file_path: Optional[Path] = None,
    augmentation_type: str = "wordnet"
) -> dict:

    decimals = 3

    # Init results
    results = {
        "perturbation level": [],
        "accuracy": [],
        "RS": [],
        "mCE": [],
        "PDR": [],
        "MDS": [],
    }

    base_accuracy = None
    accuracy_at_levels = []

    # Perturbation and Evaluation Loop
    for level in perturbation_levels:
        load_path = file_path / f"perturbed_data_{level:.2f}.pkl" if file_path else None

        X_perturbed = apply_perturbation(
            X,
            level=level,
            augmentation_type=augmentation_type,
            load_path=load_path
        )

        X_perturbed_vect = vectorize_data(vectorizer, X_perturbed)
        y_pred = model.predict(X_perturbed_vect)
        acc = accuracy_score(y, y_pred)

        results["perturbation level"].append(level)
        results["accuracy"].append(round(acc, decimals))

        if level == 0.0:
            base_accuracy = acc
            for key in ("RS", "mCE", "PDR", "MDS"):
                results[key].append(None)
        else:
            accuracy_at_levels.append(acc)

    if base_accuracy is None:
        raise ValueError("Baselevel accuracy (level=0.0) was not found. Ensure 0.0 is in perturbation_levels.")

    mce_values = mean_corruption_error(accuracy_at_levels) if "mce" in metrics else []
    mds_values = mean_divergence_score(base_accuracy, accuracy_at_levels) if "mds" in metrics else []

    for i, acc in enumerate(accuracy_at_levels):
    # RS (Robustness Score)
        rs = robustness_score(base_accuracy, acc) if "rs" in metrics else None
        results["RS"].append(round(rs, decimals) if rs is not None else None)

        # mCE (Mean Corruption Error)
        mce = mce_values[i] if "mce" in metrics else None
        results["mCE"].append(round(mce, decimals) if mce is not None else None)

        # PDR (Performance Drop Rate)
        pdr = performance_drop_rate(base_accuracy, acc) if "pdr" in metrics else None
        results["PDR"].append(round(pdr, decimals) if pdr is not None else None)

        # MDS (Mean Divergence Score — precomputed)
        mds = mds_values[i] if "mds" in metrics else None
        results["MDS"].append(round(mds, decimals) if mds is not None else None)

    return results
