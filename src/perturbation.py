from typing import Optional
from numpy import ndarray
from textattack.augmentation import Augmenter, EasyDataAugmenter

def apply_perturbation(X: ndarray, level: float = 0.2, augmenter: Optional[Augmenter] = None) -> list[str]:
    
    if augmenter is None:
        augmenter = EasyDataAugmenter(pct_words_to_swap=level)

    perturbed_data = []

    for text in X:
        augmented = augmenter.augment(text)
        perturbed_data.append(augmented[0] if augmented else text)

    return perturbed_data



