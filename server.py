from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
from classifiers.llm import LLMClassifier
from litellm import completion
import asyncio

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the LLM classifier
classifier = LLMClassifier(client=completion, model="gpt-3.5-turbo")

class TextInput(BaseModel):
    text: str
    categories: Optional[List[str]] = None

class ClassificationResponse(BaseModel):
    category: str
    confidence: float
    explanation: str

class CategorySuggestionResponse(BaseModel):
    categories: List[str]

@app.post("/classify", response_model=ClassificationResponse)
async def classify_text(text_input: TextInput):
    try:
        # Use async classification
        results = await classifier.classify_async(
            [text_input.text],
            text_input.categories
        )
        result = results[0]  # Get first result since we're classifying one text
        
        return ClassificationResponse(
            category=result["category"],
            confidence=result["confidence"],
            explanation=result["explanation"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/suggest-categories", response_model=CategorySuggestionResponse)
async def suggest_categories(texts: List[str]):
    try:
        categories = await classifier._suggest_categories_async(texts)
        return CategorySuggestionResponse(categories=categories)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 