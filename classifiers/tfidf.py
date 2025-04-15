
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

from base import BaseClassifier


class TFIDFClassifier(BaseClassifier):
    """Classifier using TF-IDF and clustering for fast classification"""

    def __init__(self):
        super().__init__()
        self.vectorizer = TfidfVectorizer(
            max_features=1000, stop_words="english", ngram_range=(1, 2)
        )
        self.model = None
        self.feature_names = None
        self.categories = None
        self.centroids = None

    def classify(self, texts, categories=None):
        """Classify texts using TF-IDF and clustering"""
        # Vectorize the texts
        X = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # Auto-detect categories if not provided
        if not categories:
            num_clusters = min(5, len(texts))  # Don't create more clusters than texts
            self.categories = self._generate_default_categories(texts, num_clusters)
        else:
            self.categories = categories
            num_clusters = len(categories)

        # Cluster the texts
        self.model = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = self.model.fit_predict(X)
        self.centroids = self.model.cluster_centers_

        # Calculate distances to centroids for confidence
        distances = self._calculate_distances(X)

        # Prepare results
        results = []
        for i, text in enumerate(texts):
            cluster_idx = clusters[i]

            # Calculate confidence (inverse of distance, normalized)
            confidence = self._calculate_confidence(distances[i])

            # Create explanation
            explanation = self._generate_explanation(X[i], cluster_idx)

            results.append(
                {
                    "category": self.categories[cluster_idx],
                    "confidence": confidence,
                    "explanation": explanation,
                }
            )

        return results

    def _calculate_distances(self, X):
        """Calculate distances from each point to each centroid"""
        return np.sqrt(
            (
                (X.toarray()[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2
            ).sum(axis=2)
        )

    def _calculate_confidence(self, distances):
        """Convert distances to confidence scores (0-100)"""
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Normalize and invert (smaller distance = higher confidence)
        if max_dist == min_dist:
            return 70  # Default mid-range confidence when all distances are equal

        normalized_dist = (distances - min_dist) / (max_dist - min_dist)
        min_normalized = np.min(normalized_dist)

        # Invert and scale to 50-100 range (TF-IDF is never 100% confident)
        confidence = 100 - (min_normalized * 50)
        return round(confidence, 1)

    def _generate_explanation(self, text_vector, cluster_idx):
        """Generate an explanation for the classification"""
        # Get the most important features for this cluster
        centroid = self.centroids[cluster_idx]

        # Get indices of top features for this text
        text_array = text_vector.toarray()[0]
        top_indices = text_array.argsort()[-5:][::-1]

        # Get the feature names for these indices
        top_features = [self.feature_names[i] for i in top_indices if text_array[i] > 0]

        if not top_features:
            return "No significant features identified for this classification."

        explanation = f"Classification based on key terms: {', '.join(top_features)}"
        return explanation

