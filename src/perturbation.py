import random
from numpy import ndarray
from textattack.augmentation import WordNetAugmenter, EasyDataAugmenter, CharSwapAugmenter

def apply_perturbation(X: ndarray, level: float = 0.3) -> list[str]:
    random.seed(42)  # Set a seed for reproducibility

    wordnet_augmenter = WordNetAugmenter(pct_words_to_swap=level, transformations_per_example=1)
    charswap_augmenter = CharSwapAugmenter(pct_words_to_swap=level, transformations_per_example=1)

    perturbed_data = []

    for text in X:
        augmented = wordnet_augmenter.augment(text)
        augmented_text = augmented[0] if augmented else text

        augmented = charswap_augmenter.augment(augmented_text)
        perturbed_data.append(augmented[0] if augmented else augmented_text)

    return perturbed_data



