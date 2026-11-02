# Task 1 — Project Summary

## Title
Netflix Content Recommendation System

## Aim
To build a content-based machine learning recommendation system that suggests Netflix titles similar to a selected movie or TV show using genres, categories, and other available content attributes.

## Dataset
The supplied `Dataset.csv` contains 8,790 rows and 10 columns. It has no null values recorded by Pandas. After trimming title whitespace and removing repeated title entries, 8,786 unique title records remain. The model uses `type`, `director`, `country`, `release_year`, `rating`, `duration`, and `listed_in`. The `title` column is used to identify the selected item, not as a similarity feature.

## Algorithm
1. Data cleaning and duplicate-title removal.
2. Metadata feature preparation with weighted tokens.
3. TF-IDF vectorization.
4. Cosine similarity calculation.
5. Ranking and display of the top-N similar titles.

## Tools and libraries
Python, Pandas, NumPy, Scikit-learn, Matplotlib, and Jupyter Notebook.

## Sample result
For **The Conjuring**, the system recommends **Insidious** (0.9654) and **The Conjuring 2** (0.8657) as the two closest matches in the supplied catalog. The score is a cosine-similarity measure, not a probability.

## Evaluation
An intrinsic evaluation was performed on a reproducible random sample of 100 titles, with the top 10 recommendations inspected for each query.

| Metric | Result |
|---|---:|
| Queries evaluated | 100 |
| Mean top-10 cosine similarity | 0.5684 |
| Mean top-10 genre Jaccard overlap | 0.8175 |
| Same content type share in top 10 | 99.8% |
| Same director share when known | 36.9% |

The dataset does not provide user watch history, clicks, or ground-truth recommendation labels. Therefore, these metrics measure metadata consistency and do not represent user satisfaction or recommendation accuracy.

## Conclusion
The project successfully builds a content-based Netflix recommendation system using TF-IDF and cosine similarity. It identifies titles with similar genres and metadata and returns ranked suggestions for a selected title. The evaluation indicates strong genre overlap and near-identical content-type matching within the sampled recommendations. The system can be improved further by adding plot descriptions, cast details, or real user feedback when such data is available.
