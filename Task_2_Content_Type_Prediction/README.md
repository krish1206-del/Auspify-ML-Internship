# Task 2 – Netflix Content Type Prediction Model

**Auspify Technologies | Machine Learning Internship**

## Objective

Build a supervised machine-learning model that predicts whether a Netflix catalogue title is a **Movie** or a **TV Show** using available content metadata.

## Dataset

- File: `Dataset.csv`
- Records: **8,790**
- Columns supplied: 10
- Target: `type`
- Class distribution: 6,126 Movies and 2,664 TV Shows

## Project workflow

1. Load and inspect the dataset.
2. Check missing values and duplicate rows.
3. Select relevant input features and define `type` as the target.
4. Prepare numerical, categorical, and text features.
5. Train Logistic Regression, Decision Tree, and Random Forest classifiers.
6. Compare the models with a majority-class baseline.
7. Evaluate the selected model with accuracy, macro precision, macro recall, macro F1, a classification report, and a confusion matrix.
8. Perform a feature-ablation check to show how genre metadata affects the result.

## Feature selection

**Target:** `type` (`Movie` or `TV Show`)

**Predictors:**
- Numerical/date-derived: `release_year`, `date_added_year`, `date_added_month`
- Categorical: `rating`
- Text/multi-value metadata: `director`, `country`, `listed_in`

**Excluded:**
- `show_id`: catalogue identifier
- `title`: high-cardinality title label, not a general content attribute
- `duration`: excluded to prevent target leakage because values such as `90 min` and `2 Seasons` closely disclose the content type

Date fields are parsed and model preprocessing is performed inside Scikit-learn pipelines. Text metadata is vectorized with TF-IDF, and `rating` is one-hot encoded.

## Model results

The experiment uses a reproducible **stratified 80/20 train/test split** (`random_state=42`): 7,032 training rows and 1,758 test rows.

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Logistic Regression | 99.94% | 99.93% |
| Decision Tree | 99.49% | 99.39% |
| Random Forest | 98.52% | 98.26% |
| Majority-class baseline | 69.68% | 41.07% |

**Selected model:** Logistic Regression (highest macro F1 among the tested classifiers).

### Important interpretation

The `listed_in` metadata includes TV-specific genre labels. To make this influence clear, Logistic Regression was also tested on the same split without `listed_in`:

- With all selected metadata: **99.94% accuracy**
- Without `listed_in`: **94.82% accuracy**

The near-ceiling full-feature score is therefore partly explained by genre labels that are strongly associated with the target. Treat these metrics as performance on this Netflix catalogue and split, not as a guarantee of performance on every streaming platform or future dataset.

## Repository contents

```text
Task_2_Content_Type_Prediction/
├── Task_2.ipynb
├── content_type_prediction.py
├── Dataset.csv
├── requirements.txt
├── README.md
├── Task_2_Summary.md
├── screenshots/
├── charts/
└── outputs/
```

## How to run

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the notebook `Task_2.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.

   Or run the project from the terminal:

   ```bash
   python content_type_prediction.py --dataset Dataset.csv --output .
   ```

The script writes evaluation tables and a saved pipeline into `outputs/`, and generates figures in `screenshots/` and `charts/`.

## Main tools

Python, Pandas, NumPy, Scikit-learn, Matplotlib, TF-IDF, One-Hot Encoding, Logistic Regression, Decision Tree, and Random Forest.
