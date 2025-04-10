from numpy import ndarray
from textattack.augmentation import EasyDataAugmenter

def apply_perturbation(X: ndarray, level: float = 0.3) -> list[str]:
    
    augmenter = EasyDataAugmenter(pct_words_to_swap=level)
    perturbed_data = []

    for text in X:
        augmented = augmenter.augment(text)
        perturbed_data.append(augmented[0] if augmented else text)

    return perturbed_data



