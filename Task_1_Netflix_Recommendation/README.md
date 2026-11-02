# Task 1 — Netflix Content Recommendation System

**Internship:** Machine Learning Intern, Auspify Technologies  
**Intern:** Krish Nilesh Gurav  
**Task:** Task 1 (Easy) — Netflix Content Recommendation System

## Objective

Build a content-based recommendation system that suggests Netflix titles similar to a selected title using genres, categories, and other available content attributes.

## Project files

```text
Task_1_Netflix_Recommendation/
├── Task_1.ipynb
├── Dataset.csv
├── netflix_recommender.py
├── requirements.txt
├── README.md
├── Task_1_Summary.md
├── screenshots/
│   ├── Task_1_Results_Snapshot.png
│   ├── content_type_distribution.png
│   ├── top_genres.png
│   ├── conjuring_recommendation_scores.png
│   └── evaluation_similarity_distribution.png
├── charts/
│   ├── Task_1_Results_Snapshot.png
│   ├── content_type_distribution.png
│   ├── top_genres.png
│   ├── conjuring_recommendation_scores.png
│   └── evaluation_similarity_distribution.png
└── outputs/
    ├── dataset_profile.csv
    ├── top_genres.csv
    ├── sample_recommendations.csv
    ├── evaluation_summary.csv
    └── evaluation_details_100_queries.csv
```

## Dataset

The supplied `Dataset.csv` contains 8,790 records and 10 columns. After trimming title whitespace and removing duplicate title entries, 8,786 unique title records remain. The notebook uses metadata available in the file, including `type`, `director`, `country`, `release_year`, `rating`, `duration`, and `listed_in`.

## Methodology

1. Load and inspect the Netflix dataset.
2. Clean title values and remove duplicate titles.
3. Prepare weighted metadata features, giving genres the strongest weight.
4. Convert features into numerical vectors using TF-IDF.
5. Calculate cosine similarity between a selected title and the rest of the catalog.
6. Return the top-N most similar titles.
7. Evaluate metadata consistency on a reproducible sample of 100 titles.

## Technologies

Python, Pandas, NumPy, Scikit-learn, Matplotlib, and Jupyter Notebook.

## Run locally

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
jupyter notebook
```

Open `Task_1.ipynb` and run all cells from top to bottom. Keep `Dataset.csv` and `netflix_recommender.py` in the same folder as the notebook. The notebook creates or refreshes the `charts/` and `outputs/` folders when executed.

To run the recommender from the command line instead:

```bash
python netflix_recommender.py --data Dataset.csv --title "The Conjuring" --top 10
```

## Example result

For the title **The Conjuring**, the highest-ranked recommendations include:

| Rank | Recommended title | Similarity score |
|---:|---|---:|
| 1 | Insidious | 0.9654 |
| 2 | The Conjuring 2 | 0.8657 |
| 3 | Kanika | 0.5628 |
| 4 | Green Room | 0.4398 |
| 5 | Knock Knock | 0.4334 |

Similarity scores range from 0 to 1. They indicate similarity between the metadata vectors; they are not probabilities, audience ratings, or measures of user satisfaction.

## Evaluation and limitations

The notebook evaluates 100 sampled titles using mean top-10 cosine similarity, genre Jaccard overlap, same-content-type share, and director overlap when director information is available. The dataset has no user clicks, viewing histories, or ground-truth recommendation pairs, so the project does not claim user-level recommendation accuracy, precision, or recall. These results are an intrinsic metadata-consistency check only.

See `Task_1_Summary.md` for the concise project summary and evaluation results. Visual result files are included in both `charts/` (generated notebook charts) and `screenshots/` (visual evidence images).
