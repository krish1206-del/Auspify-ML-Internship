# Task 3 – Netflix Audience Rating Classification

**Auspify Technologies | Machine Learning Internship**

A multi-class classification project that predicts the audience rating category of a Netflix title from its content metadata. This project is based on Task 3 in the Auspify Machine Learning Task List.

## Objective
Build a model to predict Netflix audience-rating categories using content attributes. The workflow includes rating-category analysis, training-data preparation, classification model training, hyperparameter tuning, and evaluation.

## Dataset
`Dataset.csv` contains 8,790 Netflix title records and 10 columns. The target is `rating`. The model uses the following predictors:

- `director`, `country`, and `listed_in` combined as text metadata and vectorized with TF-IDF
- `type` and `duration`, one-hot encoded
- `release_year`, standardized

`show_id`, `title`, and `rating` are not used as model input features. The rating is the target only.

## Important target-label handling
The dataset has 14 original rating labels, including a few very rare labels. To make stratified evaluation more stable, categories with fewer than 50 records are grouped into `Other (rare)`. In this dataset, the grouped labels are `G`, `TV-Y7-FV`, `NC-17`, and `UR`. The trained model therefore predicts 11 grouped categories. The notebook shows both the original and grouped rating distributions and explains this choice.

## Models
- Most-frequent baseline
- Logistic Regression (unweighted and class-weighted)
- Decision Tree
- Random Forest
- Tuned Logistic Regression and Tuned Decision Tree using 3-fold stratified cross-validation

Hyperparameter tuning uses macro-F1 because the classes are imbalanced. The selected final model is chosen using cross-validation macro-F1, not by inspecting the test set.

## Evaluation
The held-out test set contains 1,758 records. Metrics include accuracy, balanced accuracy, macro-F1, weighted-F1, a class-wise classification report, and a confusion matrix. See `outputs/model_comparison.csv`, `outputs/run_summary.json`, and the notebook for the full results.

The final model in the current run achieved approximately:

| Metric | Result |
|---|---:|
| Accuracy | 47.50% |
| Balanced accuracy | 43.37% |
| Macro-F1 | 0.385 |
| Weighted-F1 | 0.490 |

The unweighted Logistic Regression variant achieved higher raw accuracy (53.81%) but lower balanced accuracy. This illustrates why accuracy alone can be misleading for imbalanced multi-class classification.

## Project structure

```text
Task_3_Netflix_Audience_Rating_Classification/
├── Task_3.ipynb
├── Dataset.csv
├── task3_rating_classifier.py
├── requirements.txt
├── README.md
├── Task_3_Summary.md
├── screenshots/
│   ├── 01_dataset_preview.png
│   ├── 02_original_rating_distribution.png
│   ├── 03_grouped_rating_distribution.png
│   ├── 04_model_comparison.png
│   ├── 05_confusion_matrix.png
│   ├── 06_classwise_f1_scores.png
│   └── 07_Task_3_Results_Snapshot.png
└── outputs/
    ├── best_rating_classifier.joblib
    ├── classification_report.csv
    ├── confusion_matrix.csv
    ├── model_comparison.csv
    ├── rating_category_counts.csv
    ├── run_summary.json
    └── sample_test_predictions.csv
```

## How to run

1. Extract this folder.
2. Open a terminal in this folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Open `Task_3.ipynb` in Jupyter Notebook, JupyterLab, Google Colab, or VS Code and run all cells.

   Or execute the standalone Python script:

   ```bash
   python task3_rating_classifier.py
   ```

The notebook and Python script use relative paths, so keep `Dataset.csv` in the same folder as the notebook/script.

## Limitations
- The dataset contains catalogue metadata, not audience feedback or the internal rules used by Netflix to assign ratings.
- Rare original labels are grouped for evaluation; the model does not distinguish those labels individually.
- Some rating categories are difficult to predict from the available metadata. Consult macro-F1, balanced accuracy, and the class-wise report instead of relying only on accuracy.
- The results describe this dataset and fixed random split; they should not be interpreted as production performance.

## Author
Krish Nilesh Gurav  
Machine Learning Intern – Auspify Technologies
