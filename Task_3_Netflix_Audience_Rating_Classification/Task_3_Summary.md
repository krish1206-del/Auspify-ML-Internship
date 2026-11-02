# Task 3 – Project Summary

## Project
Netflix Audience Rating Classification

## Aim
Predict the audience rating category of Netflix titles using content attributes as a multi-class classification problem.

## Dataset
- Total records: 8,790
- Original rating labels: 14
- Modelling labels: 11 (categories with fewer than 50 examples were grouped as `Other (rare)`)
- Training set: 7,032 records
- Test set: 1,758 records

Rare labels grouped: `G`, `TV-Y7-FV`, `NC-17`, and `UR`.

## Features used
TF-IDF text features from director, country, and genre metadata; one-hot encoded content type and duration; standardized release year. The title identifier, title name, and target rating were not supplied as predictor features.

## Models compared
Most-frequent baseline, Logistic Regression (unweighted and class-weighted), Decision Tree, Random Forest, tuned Logistic Regression, and tuned Decision Tree.

## Final model and test results
The selected model is **Tuned Logistic Regression**, selected using 3-fold stratified cross-validation macro-F1.

| Metric | Result |
|---|---:|
| Accuracy | 47.50% |
| Balanced accuracy | 43.37% |
| Macro-F1 | 0.385 |
| Weighted-F1 | 0.490 |
| Cross-validation macro-F1 | 0.391 |

An unweighted Logistic Regression model reached 53.81% accuracy, but its balanced accuracy was 38.07%. The class-weighted tuned model was selected because it better balances performance across rating categories.

## Conclusion
The project demonstrates a complete multi-class classification workflow, including data preparation, TF-IDF feature extraction, categorical encoding, Decision Tree and Random Forest comparisons, hyperparameter tuning, and model evaluation. The results show that predicting audience-rating categories from catalogue metadata is challenging, especially for less common categories. For this reason, balanced accuracy and macro-F1 are important alongside overall accuracy.

## Files
The notebook contains the full workflow and saved outputs. `task3_rating_classifier.py` provides a standalone script for reproducing the model training and evaluation.
