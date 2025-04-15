import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import random
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from prompts import CATEGORY_SUGGESTION_PROMPT, TEXT_CLASSIFICATION_PROMPT
from scipy.sparse import csr_matrix

from .base import BaseClassifier


class TFIDFClassifier(BaseClassifier):
    """Classifier using TF-IDF and clustering for fast classification"""

    def __init__(self) -> None:
        super().__init__()
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(
            max_features=1000, stop_words="english", ngram_range=(1, 2)
        )
        self.model: Optional[KMeans] = None
        self.feature_names: Optional[np.ndarray] = None
        self.categories: Optional[List[str]] = None
        self.centroids: Optional[np.ndarray] = None

    def classify(self, texts: List[str], categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Classify texts using TF-IDF and clustering"""
        # Vectorize the texts
        X: csr_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # Auto-detect categories if not provided
        if not categories:
            num_clusters: int = min(5, len(texts))  # Don't create more clusters than texts
            self.categories = self._generate_default_categories(texts, num_clusters)
        else:
            self.categories = categories
            num_clusters = len(categories)

        # Cluster the texts
        self.model = KMeans(n_clusters=num_clusters, random_state=42)
        clusters: np.ndarray = self.model.fit_predict(X)
        self.centroids = self.model.cluster_centers_

        # Calculate distances to centroids for confidence
        distances: np.ndarray = self._calculate_distances(X)

        # Prepare results
        results: List[Dict[str, Any]] = []
        for i, text in enumerate(texts):
            cluster_idx: int = clusters[i]

            # Calculate confidence (inverse of distance, normalized)
            confidence: float = self._calculate_confidence(distances[i])

            # Create explanation
            explanation: str = self._generate_explanation(X[i], cluster_idx)

            results.append(
                {
                    "category": self.categories[cluster_idx],
                    "confidence": confidence,
                    "explanation": explanation,
                }
            )

        return results

    def _calculate_distances(self, X: csr_matrix) -> np.ndarray:
        """Calculate distances from each point to each centroid"""
        return np.sqrt(
            (
                (X.toarray()[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2
            ).sum(axis=2)
        )

    def _calculate_confidence(self, distances: np.ndarray) -> float:
        """Convert distances to confidence scores (0-100)"""
        min_dist: float = np.min(distances)
        max_dist: float = np.max(distances)

        # Normalize and invert (smaller distance = higher confidence)
        if max_dist == min_dist:
            return 70  # Default mid-range confidence when all distances are equal

        normalized_dist: np.ndarray = (distances - min_dist) / (max_dist - min_dist)
        min_normalized: float = np.min(normalized_dist)

        # Invert and scale to 50-100 range (TF-IDF is never 100% confident)
        confidence: float = 100 - (min_normalized * 50)
        return round(confidence, 1)

    def _generate_explanation(self, text_vector: csr_matrix, cluster_idx: int) -> str:
        """Generate an explanation for the classification"""
        # Get the most important features for this cluster
        centroid: np.ndarray = self.centroids[cluster_idx]

        # Get indices of top features for this text
        text_array: np.ndarray = text_vector.toarray()[0]
        top_indices: np.ndarray = text_array.argsort()[-5:][::-1]

        # Get the feature names for these indices
        top_features: List[str] = [self.feature_names[i] for i in top_indices if text_array[i] > 0]

        if not top_features:
            return "No significant features identified for this classification."

        explanation: str = f"Classification based on key terms: {', '.join(top_features)}"
        return explanation

