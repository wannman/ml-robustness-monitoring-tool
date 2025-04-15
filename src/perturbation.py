import random
import numpy as np
from textattack.augmentation import EasyDataAugmenter

def apply_perturbation(X: list[np.ndarray], level: float = 0.3) -> list[np.ndarray]:

    # Set a seed for reproducibility
    random.seed(42)  
    augmenter = EasyDataAugmenter(pct_words_to_swap=0.3, transformations_per_example=1)

    titles, descriptions = X
    num_lines = len(titles)
    
    # Calculate the number of lines to augment
    num_to_augment = int(num_lines * level)  

    data_to_augment = random.sample(range(num_lines), num_to_augment)

    perturbed_titles = titles.copy()
    perturbed_descriptions = descriptions.copy()

    for idx in data_to_augment: 
        perturbed_titles[idx] = augmenter.augment(titles[idx])[0]
        perturbed_descriptions[idx] = augmenter.augment(descriptions[idx])[0]

    return [perturbed_titles, perturbed_descriptions]



