import os
import pickle
import random
from typing import Optional
import numpy as np
from textattack.augmentation import WordNetAugmenter, CharSwapAugmenter, EasyDataAugmenter, CLAREAugmenter, Augmenter
from textattack.transformations import WordSwapNeighboringCharacterSwap

def apply_perturbation(
    X: np.ndarray,
    level: float = 0.3,
    save_path: Optional[str] = None,
    load_path: Optional[str] = None
) -> np.ndarray:

    # Load previously saved perturbed data (if exists)
    if load_path and os.path.exists(load_path):
        with open(load_path, "rb") as f:
            return pickle.load(f)

    # Otherwise, perform perturbation
    pct_words_to_swap = 1.0
    random.seed(42)
    #augmenter = EasyDataAugmenter(pct_words_to_swap=pct_words_to_swap, transformations_per_example=1)
    augmenter = WordNetAugmenter(pct_words_to_swap=pct_words_to_swap, transformations_per_example=1)
    #augmenter = CharSwapAugmenter(pct_words_to_swap=pct_words_to_swap, transformations_per_example=1)
    #augmenter = Augmenter(transformation=WordSwapNeighboringCharacterSwap(), pct_words_to_swap=pct_words_to_swap, transformations_per_example=1)

    num_lines = len(X)
    num_to_augment = int(num_lines * level)
    data_to_augment = random.sample(range(num_lines), num_to_augment)

    perturbed_X = X.copy()

    for idx in data_to_augment:
        perturbed_X[idx] = augmenter.augment(X[idx])[0]

    # Save the result to disk if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(perturbed_X, f)

    return perturbed_X