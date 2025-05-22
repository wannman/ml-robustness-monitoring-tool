# Robustness Evaluation of Text Classification Models

This project evaluates the robustness of text classification models under adversarial perturbations. It uses the [AG News](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) dataset and various textual augmentations to simulate real-world perturbations

---

## Goal

- Train baseline classifiers on clean text data.
- Apply controlled perturbations using `textattack` (e.g., WordNet, deletion, character swap).
- Evaluate model robustness across multiple metrics, including Mean Divergence Score, a novel metric designed to indicate inconsistent degradation compared to average performance. 
- Store and reuse perturbed data to speed up experimentation.
- Visualize results

---

## Pipeline Overview

1. **Import Dependencies**  
   Setup system paths and import libraries.

2. **Load Dataset**  
   Combine "Title" and "Description" into a single input feature. Labels are extracted from "Class Index".

3. **Vectorize Text Data**  
   Transform raw text into TF-IDF vectors.

4. **Train Models**  
   Train Logistic Regression, Naive Bayes, and Random Forest classifiers.

5. **Apply Perturbations (Optional)**  
   Use `textattack` augmenters to create perturbed versions of test data. Perturbed data is saved for reuse.

6. **Evaluate Robustness**  
   Evaluate model accuracy under increasing perturbation levels. Compute multiple robustness metrics:
   - Accuracy
   - Robustness Score (RS)
   - Mean Corruption Error (mCE)
   - Performance Drop Rate (PDR)
   - Mean Divergence Score (MDS)

7. **Save Results to CSV File (Optional)**
   Exports the evaluation results to a CSV file

8. **Load Evaluation Results**  
   Loads previously saved evaluation results from CSV files for
   visualization, comparison, aggregation

9. **Visualization of Results**

---

## Dependencies

Install required libraries via pip:

```bash
pip install -r requirements.txt

