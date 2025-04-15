import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import random
import json
import asyncio
from typing import List, Dict, Any, Optional, Union
import sys
import os
from litellm import OpenAI

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import CATEGORY_SUGGESTION_PROMPT, TEXT_CLASSIFICATION_PROMPT

from .base import BaseClassifier


class LLMClassifier(BaseClassifier):
    """Classifier using a Large Language Model for more accurate but slower classification"""

    def __init__(self, client: OpenAI, model: str = "gpt-3.5-turbo") -> None:
        super().__init__()
        self.client: OpenAI = client
        self.model: str = model

    async def _classify_text_async(self, text: str, categories: List[str]) -> Dict[str, Any]:
        """Async version of text classification"""
        prompt: str = TEXT_CLASSIFICATION_PROMPT.format(
            categories=", ".join(categories),
            text=text
        )

        try:
            # Use the synchronous client method but run it in a thread pool
            loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
            response: Any = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=200,
                )
            )

            # Parse JSON response
            response_text: str = response.choices[0].message.content.strip()
            result: Dict[str, Any] = json.loads(response_text)

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
            category: str = categories[0]  # Default
            for cat in categories:
                if cat.lower() in response_text.lower():
                    category = cat
                    break

            return {
                "category": category,
                "confidence": 50,
                "explanation": f"Classification based on language model analysis. (Note: Structured response parsing failed)",
            }
        except Exception as e:
            return {
                "category": categories[0],
                "confidence": 50,
                "explanation": f"Error during classification: {str(e)}",
            }

    async def _suggest_categories_async(self, texts: List[str], sample_size: int = 20) -> List[str]:
        """Async version of category suggestion"""
        # Take a sample of texts to avoid token limitations
        if len(texts) > sample_size:
            sample_texts: List[str] = random.sample(texts, sample_size)
        else:
            sample_texts: List[str] = texts

        prompt: str = CATEGORY_SUGGESTION_PROMPT.format("\n---\n".join(sample_texts))

        try:
            # Use the synchronous client method but run it in a thread pool
            loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
            response: Any = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=100,
                )
            )

            # Parse response to get categories
            categories_text: str = response.choices[0].message.content.strip()
            categories: List[str] = [cat.strip() for cat in categories_text.split(",")]

            return categories
        except Exception as e:
            # Fallback to default categories on error
            print(f"Error suggesting categories: {str(e)}")
            return self._generate_default_categories(texts)

    async def classify_async(
        self, texts: List[str], categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Async method to classify texts"""
        if not categories:
            categories = await self._suggest_categories_async(texts)

        # Create tasks for all texts
        tasks: List[asyncio.Task] = [self._classify_text_async(text, categories) for text in texts]
        
        # Gather all results
        results: List[Dict[str, Any]] = await asyncio.gather(*tasks)
        return results

    def classify(
        self, texts: List[str], categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for backwards compatibility"""
        return asyncio.run(self.classify_async(texts, categories))
