"""Content-based Netflix title recommender for Auspify ML Internship Task 1.

The model uses only metadata present in Dataset.csv. It does not use viewing
history, ratings from users, or Netflix's internal recommendation data.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


REQUIRED_COLUMNS = {
    "show_id", "type", "title", "director", "country",
    "release_year", "rating", "duration", "listed_in"
}


def _is_missing(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in {"", "not given", "nan", "none", "unknown"}


def _slug(value: object) -> str:
    """Normalize a category into a single readable feature token."""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _split_values(value: object) -> list[str]:
    if _is_missing(value):
        return []
    return [part.strip() for part in str(value).split(",") if not _is_missing(part)]


def _duration_bucket(row: pd.Series) -> str | None:
    raw = row.get("duration")
    if _is_missing(raw):
        return None
    match = re.search(r"(\d+)", str(raw))
    if not match:
        return None
    amount = int(match.group(1))
    if str(row.get("type", "")).strip().lower() == "movie":
        if amount < 90:
            return "runtime_under_90_min"
        if amount < 120:
            return "runtime_90_to_119_min"
        return "runtime_120_plus_min"
    if amount == 1:
        return "seasons_1"
    if amount <= 3:
        return "seasons_2_to_3"
    return "seasons_4_plus"


def create_content_features(data: pd.DataFrame) -> pd.Series:
    """Build weighted, structured text from each title's available metadata.

    Genres receive the highest weight, followed by director and content type.
    No plot summaries or user-behavior information exist in the supplied file.
    """
    feature_rows: list[str] = []
    for _, row in data.iterrows():
        tokens: list[str] = []

        # Genres are the strongest signal for content similarity.
        for genre in _split_values(row.get("listed_in")):
            tokens.extend([f"genre_{_slug(genre)}"] * 3)

        # Director and format are useful secondary signals.
        for director in _split_values(row.get("director")):
            tokens.extend([f"director_{_slug(director)}"] * 2)

        content_type = row.get("type")
        if not _is_missing(content_type):
            tokens.extend([f"type_{_slug(content_type)}"] * 2)

        for country in _split_values(row.get("country")):
            tokens.append(f"country_{_slug(country)}")

        rating = row.get("rating")
        if not _is_missing(rating):
            tokens.append(f"rating_{_slug(rating)}")

        release_year = row.get("release_year")
        if not _is_missing(release_year):
            try:
                decade = (int(release_year) // 10) * 10
                tokens.append(f"decade_{decade}s")
            except (TypeError, ValueError):
                pass

        duration_feature = _duration_bucket(row)
        if duration_feature:
            tokens.append(duration_feature)

        feature_rows.append(" ".join(tokens))

    return pd.Series(feature_rows, index=data.index, name="content_features")


class NetflixRecommender:
    """TF-IDF + cosine-similarity content recommendation model."""

    def __init__(self, data: pd.DataFrame | str | Path):
        if isinstance(data, (str, Path)):
            self.raw_data = pd.read_csv(data)
        else:
            self.raw_data = data.copy()

        missing_columns = REQUIRED_COLUMNS.difference(self.raw_data.columns)
        if missing_columns:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

        self.data = self._clean_data(self.raw_data)
        self.data = self.data.reset_index(drop=True)
        self.data["content_features"] = create_content_features(self.data)

        self.vectorizer = TfidfVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
            sublinear_tf=True,
            norm="l2",
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data["content_features"])
        self._title_lookup = {
            str(title).casefold(): idx for idx, title in enumerate(self.data["title"])
        }

    @staticmethod
    def _clean_data(data: pd.DataFrame) -> pd.DataFrame:
        cleaned = data.copy()
        cleaned["title"] = cleaned["title"].astype(str).str.strip()
        cleaned = cleaned[cleaned["title"].ne("")]
        cleaned = cleaned.drop_duplicates(subset=["title"], keep="first").copy()

        # Normalize placeholder values. Preserve true missingness in the source,
        # while keeping model inputs safe and consistent.
        for column in ["director", "country", "rating", "duration", "listed_in"]:
            cleaned[column] = cleaned[column].where(
                ~cleaned[column].astype(str).str.strip().str.lower().isin(
                    {"not given", "nan", "none", "unknown", ""}
                ),
                np.nan,
            )
        cleaned["type"] = cleaned["type"].fillna("Unknown").astype(str).str.strip()
        cleaned["release_year"] = pd.to_numeric(cleaned["release_year"], errors="coerce")
        return cleaned

    def _resolve_title(self, title: str) -> int:
        query = str(title).strip().casefold()
        if query in self._title_lookup:
            return self._title_lookup[query]

        partial = [
            (idx, str(candidate))
            for idx, candidate in enumerate(self.data["title"])
            if query in str(candidate).casefold()
        ]
        if len(partial) == 1:
            return partial[0][0]
        if len(partial) > 1:
            matches = [name for _, name in partial[:10]]
            raise ValueError(
                f"More than one title matches '{title}'. Try an exact title. Matches: {matches}"
            )

        # Provide helpful close matches for spelling errors or partial queries.
        from difflib import get_close_matches
        suggestions = get_close_matches(str(title), self.data["title"].tolist(), n=5, cutoff=0.35)
        detail = f" Similar titles: {suggestions}" if suggestions else ""
        raise ValueError(f"Title '{title}' was not found in the dataset.{detail}")

    def recommend(self, title: str, top_n: int = 10) -> pd.DataFrame:
        """Return the top-N most similar titles, excluding the selected title."""
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        query_idx = self._resolve_title(title)
        similarity_scores = (self.tfidf_matrix[query_idx] @ self.tfidf_matrix.T).toarray().ravel()
        similarity_scores[query_idx] = -1.0
        top_indices = np.argsort(similarity_scores)[::-1][:top_n]

        columns = ["title", "type", "release_year", "listed_in", "director", "country"]
        result = self.data.iloc[top_indices][columns].copy()
        result.insert(0, "rank", np.arange(1, len(result) + 1))
        result["similarity_score"] = similarity_scores[top_indices]
        result["similarity_score"] = result["similarity_score"].round(4)
        return result.reset_index(drop=True)

    def evaluate(self, sample_size: int = 100, top_n: int = 10, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Intrinsic relevance check over a reproducible sample of catalog titles.

        Since the dataset has no user clicks, watch history, or relevance labels,
        this is a metadata-consistency evaluation—not predictive accuracy.
        """
        if sample_size < 1 or top_n < 1:
            raise ValueError("sample_size and top_n must be at least 1")

        rng = np.random.default_rng(random_state)
        n = len(self.data)
        sample_indices = np.sort(rng.choice(n, size=min(sample_size, n), replace=False))
        similarities = (self.tfidf_matrix[sample_indices] @ self.tfidf_matrix.T).toarray()

        rows: list[dict[str, float | str]] = []
        for position, query_idx in enumerate(sample_indices):
            scores = similarities[position].copy()
            scores[query_idx] = -1.0
            neighbors = np.argsort(scores)[::-1][:top_n]
            query = self.data.iloc[query_idx]
            query_genres = set(_split_values(query.get("listed_in")))
            query_directors = set(_split_values(query.get("director")))

            genre_overlaps: list[float] = []
            same_type: list[float] = []
            same_director: list[float] = []
            top_scores: list[float] = []
            for neighbor_idx in neighbors:
                candidate = self.data.iloc[neighbor_idx]
                candidate_genres = set(_split_values(candidate.get("listed_in")))
                union = query_genres | candidate_genres
                intersection = query_genres & candidate_genres
                genre_overlaps.append(len(intersection) / len(union) if union else 0.0)
                same_type.append(float(str(query["type"]) == str(candidate["type"])))
                candidate_directors = set(_split_values(candidate.get("director")))
                if query_directors and candidate_directors:
                    same_director.append(float(bool(query_directors & candidate_directors)))
                top_scores.append(float(scores[neighbor_idx]))

            rows.append({
                "query_title": str(query["title"]),
                "mean_top_n_similarity": float(np.mean(top_scores)),
                "mean_genre_jaccard": float(np.mean(genre_overlaps)),
                "same_type_share": float(np.mean(same_type)),
                "same_director_share_when_known": float(np.mean(same_director)) if same_director else np.nan,
            })

        details = pd.DataFrame(rows)
        summary = pd.DataFrame([
            {"metric": "Queries evaluated", "value": len(details), "interpretation": "Number of catalog titles sampled"},
            {"metric": f"Mean top-{top_n} cosine similarity", "value": details["mean_top_n_similarity"].mean(), "interpretation": "Average TF-IDF similarity of recommendations"},
            {"metric": f"Mean top-{top_n} genre Jaccard overlap", "value": details["mean_genre_jaccard"].mean(), "interpretation": "Average genre overlap; 1.0 means identical genre sets"},
            {"metric": f"Same content type share in top-{top_n}", "value": details["same_type_share"].mean(), "interpretation": "Fraction of recommendations matching Movie/TV Show type"},
            {"metric": "Same director share when director is known", "value": details["same_director_share_when_known"].mean(), "interpretation": "Fraction sharing at least one listed director; excludes unknown directors"},
        ])
        return summary, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend Netflix titles using content metadata.")
    parser.add_argument("--data", default="Dataset.csv", help="Path to Dataset.csv")
    parser.add_argument("--title", required=True, help="Netflix title to find similar recommendations for")
    parser.add_argument("--top", type=int, default=10, help="Number of recommendations (default: 10)")
    parser.add_argument("--output", default=None, help="Optional path to save recommendations as CSV")
    args = parser.parse_args()

    model = NetflixRecommender(args.data)
    results = model.recommend(args.title, args.top)
    print(f"\nRecommendations for: {args.title}\n")
    print(results.to_string(index=False))
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nSaved recommendations to: {args.output}")


if __name__ == "__main__":
    main()
