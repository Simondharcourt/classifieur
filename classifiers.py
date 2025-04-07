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

class BaseClassifier:
    """Base class for text classifiers"""
    def __init__(self):
        pass
    
    def classify(self, texts, categories=None):
        """
        Classify a list of texts into categories
        
        Args:
            texts (list): List of text strings to classify
            categories (list, optional): List of category names. If None, categories will be auto-detected
            
        Returns:
            list: List of classification results with categories, confidence scores, and explanations
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _generate_default_categories(self, texts, num_clusters=5):
        """
        Generate default categories based on text clustering
        
        Args:
            texts (list): List of text strings
            num_clusters (int): Number of clusters to generate
            
        Returns:
            list: List of category names
        """
        # Simple implementation - in real system this would be more sophisticated
        default_categories = [f"Category {i+1}" for i in range(num_clusters)]
        return default_categories


class TFIDFClassifier(BaseClassifier):
    """Classifier using TF-IDF and clustering for fast classification"""
    
    def __init__(self):
        super().__init__()
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
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
            
            results.append({
                "category": self.categories[cluster_idx],
                "confidence": confidence,
                "explanation": explanation
            })
        
        return results
    
    def _calculate_distances(self, X):
        """Calculate distances from each point to each centroid"""
        return np.sqrt(((X.toarray()[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
    
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


class LLMClassifier(BaseClassifier):
    """Classifier using a Large Language Model for more accurate but slower classification"""
    
    def __init__(self, client, model="gpt-3.5-turbo"):
        super().__init__()
        self.client = client
        self.model = model
    
    def classify(self, texts: List[str], categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Classify texts using an LLM with parallel processing"""
        if not categories:
            # First, use LLM to generate appropriate categories
            categories = self._suggest_categories(texts)
        
        # Process texts in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks
            future_to_text = {
                executor.submit(self._classify_text, text, categories): text 
                for text in texts
            }
            
            # Collect results as they complete
            results = []
            for future in as_completed(future_to_text):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing text: {str(e)}")
                    results.append({
                        "category": categories[0],
                        "confidence": 50,
                        "explanation": f"Error during classification: {str(e)}"
                    })
        
        return results
    
    def _suggest_categories(self, texts: List[str], sample_size: int = 20) -> List[str]:
        """Use LLM to suggest appropriate categories for the dataset"""
        # Take a sample of texts to avoid token limitations
        if len(texts) > sample_size:
            sample_texts = random.sample(texts, sample_size)
        else:
            sample_texts = texts
        
        prompt = CATEGORY_SUGGESTION_PROMPT.format("\n---\n".join(sample_texts))
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=100
            )
            
            # Parse response to get categories
            categories_text = response.choices[0].message.content.strip()
            categories = [cat.strip() for cat in categories_text.split(",")]
            
            return categories
        except Exception as e:
            # Fallback to default categories on error
            print(f"Error suggesting categories: {str(e)}")
            return self._generate_default_categories(texts)
    
    def _classify_text(self, text: str, categories: List[str]) -> Dict[str, Any]:
        """Use LLM to classify a single text"""
        prompt = TEXT_CLASSIFICATION_PROMPT.format(
            categories=", ".join(categories),
            text=text
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200
            )
            
            # Parse JSON response
            response_text = response.choices[0].message.content.strip()
        
            result = json.loads(response_text)
            # Ensure all required fields are present
            if not all(k in result for k in ["category", "confidence", "explanation"]):
                raise ValueError("Missing required fields in LLM response")
            
            # Validate category is in the list
            if result["category"] not in categories:
                result["category"] = categories[0]  # Default to first category if invalid
            
            # Validate confidence is a number between 0 and 100
            try:
                result["confidence"] = float(result["confidence"])
                if not 0 <= result["confidence"] <= 100:
                    result["confidence"] = 50
            except:
                result["confidence"] = 50
            
            return result
        except json.JSONDecodeError:
            # Fall back to simple parsing if JSON fails
            category = categories[0]  # Default
            for cat in categories:
                if cat.lower() in response_text.lower():
                    category = cat
                    break
            
            return {
                "category": category,
                "confidence": 50,
                "explanation": f"Classification based on language model analysis. (Note: Structured response parsing failed)"
            }



