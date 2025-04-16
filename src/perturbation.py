import os
import pickle
import random
from typing import Optional
import numpy as np
from textattack.augmentation import EasyDataAugmenter

def apply_perturbation(
    X: list[np.ndarray],
    level: float = 0.3,
    save_path: Optional[str] = None,
    load_path: Optional[str] = None
) -> list[np.ndarray]:

    # Load previously saved perturbed data (if exists)
    if load_path and os.path.exists(load_path):
        with open(load_path, "rb") as f:
            return pickle.load(f)

    # Otherwise, perform perturbation
    pct_words_to_swap = 0.6
    random.seed(42)
    augmenter = EasyDataAugmenter(pct_words_to_swap=pct_words_to_swap, transformations_per_example=1)

    titles, descriptions = X
    num_lines = len(titles)
    num_to_augment = int(num_lines * level)
    data_to_augment = random.sample(range(num_lines), num_to_augment)

    perturbed_titles = titles.copy()
    perturbed_descriptions = descriptions.copy()

    for idx in data_to_augment:
        perturbed_titles[idx] = augmenter.augment(titles[idx])[0]
        perturbed_descriptions[idx] = augmenter.augment(descriptions[idx])[0]

    result = [perturbed_titles, perturbed_descriptions]

    # Save the result to disk if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(result, f)

    return result


