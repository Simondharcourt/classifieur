
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



class LLMClassifier(BaseClassifier):
    """Classifier using a Large Language Model for more accurate but slower classification"""

    def __init__(self, client, model="gpt-3.5-turbo"):
        super().__init__()
        self.client = client
        self.model = model

    def classify(
        self, texts: List[str], categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Classify texts using an LLM with parallel processing"""
        if not categories:
            # First, use LLM to generate appropriate categories
            categories = self._suggest_categories(texts)

        # Process texts in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks with their original indices
            future_to_index = {
                executor.submit(self._classify_text, text, categories): idx
                for idx, text in enumerate(texts)
            }

            # Initialize results list with None values
            results = [None] * len(texts)

            # Collect results as they complete
            for future in as_completed(future_to_index):
                original_idx = future_to_index[future]
                try:
                    result = future.result()
                    results[original_idx] = result
                except Exception as e:
                    print(f"Error processing text: {str(e)}")
                    results[original_idx] = {
                        "category": categories[0],
                        "confidence": 50,
                        "explanation": f"Error during classification: {str(e)}",
                    }

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
                max_tokens=100,
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
            categories=", ".join(categories), text=text
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )

            # Parse JSON response
            response_text = response.choices[0].message.content.strip()

            result = json.loads(response_text)
            # Ensure all required fields are present
            if not all(k in result for k in ["category", "confidence", "explanation"]):
                raise ValueError("Missing required fields in LLM response")

            # Validate category is in the list
            if result["category"] not in categories:
                result["category"] = categories[
                    0
                ]  # Default to first category if invalid

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
                "explanation": f"Classification based on language model analysis. (Note: Structured response parsing failed)",
            }
