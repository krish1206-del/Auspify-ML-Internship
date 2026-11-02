# Task 2 – Project Summary

## Aim

Predict whether a Netflix title is a Movie or a TV Show using supervised machine learning and catalogue metadata.

## Dataset and preparation

The supplied dataset contains 8,790 records. The target column is `type`. The selected predictors are release year, date-added year and month, rating, director, country, and listed genres. Numerical fields are imputed and scaled, rating is one-hot encoded, and text/multi-value metadata is transformed with TF-IDF.

The catalogue ID and title are excluded. Duration is also excluded because values such as minutes and seasons can reveal the target directly.

## Models compared

- Logistic Regression
- Decision Tree
- Random Forest
- Majority-class baseline

The data was split using a stratified 80/20 holdout: 7,032 training records and 1,758 test records.

## Results

Logistic Regression was the strongest tested model:

- Accuracy: **99.94%**
- Macro precision: **99.96%**
- Macro recall: **99.91%**
- Macro F1-score: **99.93%**

Decision Tree achieved 99.49% accuracy and Random Forest achieved 98.52% accuracy. The majority-class baseline achieved 69.68% accuracy.

## Additional validation and interpretation

A feature-ablation experiment was run on the same train/test rows. Removing `listed_in` (genre metadata) reduced Logistic Regression accuracy from 99.94% to 94.82%. This is important because some genre labels explicitly describe TV categories. The high full-feature score is therefore influenced by strong information in the supplied Netflix metadata.

The metrics describe performance on this dataset and split; they should not be interpreted as guaranteed performance on other catalogues or future content.

## Conclusion

The project demonstrates a complete classification workflow: data inspection, feature selection, categorical and text encoding, model training, model comparison, and evaluation. Logistic Regression was selected based on macro F1-score.
