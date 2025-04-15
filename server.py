from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import json
from classifiers.llm import LLMClassifier
from litellm import completion
import asyncio
from client import get_client, initialize_client
import os
from dotenv import load_dotenv
import pandas as pd
from utils import validate_results
from process import improve_classification

# Load environment variables
load_dotenv()

app: FastAPI = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize client with API key from environment
api_key: Optional[str] = os.environ.get("OPENAI_API_KEY")
if api_key:
    success: bool
    message: str
    success, message = initialize_client(api_key)
    if not success:
        raise RuntimeError(f"Failed to initialize OpenAI client: {message}")

client = get_client()
if not client:
    raise RuntimeError("OpenAI client not initialized. Please set OPENAI_API_KEY environment variable.")

# Initialize the LLM classifier
classifier: LLMClassifier = LLMClassifier(client=client, model="gpt-3.5-turbo")

class TextInput(BaseModel):
    text: str
    categories: Optional[List[str]] = None

class BatchTextInput(BaseModel):
    texts: List[str]
    categories: Optional[List[str]] = None

class ClassificationResponse(BaseModel):
    category: str
    confidence: float
    explanation: str

class BatchClassificationResponse(BaseModel):
    results: List[ClassificationResponse]

class CategorySuggestionResponse(BaseModel):
    categories: List[str]

class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    max_tokens: int
    temperature: float

class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    api_key_configured: bool

class ValidationSample(BaseModel):
    text: str
    assigned_category: str
    confidence: float

class ValidationRequest(BaseModel):
    samples: List[ValidationSample]
    current_categories: List[str]
    text_columns: List[str]

class ValidationResponse(BaseModel):
    validation_report: str
    accuracy_score: Optional[float] = None
    misclassifications: Optional[List[Dict[str, Any]]] = None
    suggested_improvements: Optional[List[str]] = None

class ImprovementRequest(BaseModel):
    df: Dict[str, Any]  # JSON representation of the DataFrame
    validation_report: str
    text_columns: List[str]
    categories: str
    classifier_type: str
    show_explanations: bool
    file_path: str

class ImprovementResponse(BaseModel):
    improved_df: Dict[str, Any]  # JSON representation of the improved DataFrame
    new_validation_report: str
    success: bool
    updated_categories: List[str]

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check the health status of the API"""
    return HealthResponse(
        status="healthy",
        model_ready=client is not None,
        api_key_configured=api_key is not None
    )

@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info() -> ModelInfoResponse:
    """Get information about the current model configuration"""
    return ModelInfoResponse(
        model_name=classifier.model,
        model_version="1.0",
        max_tokens=200,
        temperature=0
    )

@app.post("/classify", response_model=ClassificationResponse)
async def classify_text(text_input: TextInput) -> ClassificationResponse:
    try:
        # Use async classification
        results: List[Dict[str, Any]] = await classifier.classify_async(
            [text_input.text],
            text_input.categories
        )
        result: Dict[str, Any] = results[0]  # Get first result since we're classifying one text
        
        return ClassificationResponse(
            category=result["category"],
            confidence=result["confidence"],
            explanation=result["explanation"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify-batch", response_model=BatchClassificationResponse)
async def classify_batch(batch_input: BatchTextInput) -> BatchClassificationResponse:
    """Classify multiple texts in a single request"""
    try:
        results: List[Dict[str, Any]] = await classifier.classify_async(
            batch_input.texts,
            batch_input.categories
        )
        
        return BatchClassificationResponse(
            results=[
                ClassificationResponse(
                    category=r["category"],
                    confidence=r["confidence"],
                    explanation=r["explanation"]
                ) for r in results
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/suggest-categories", response_model=CategorySuggestionResponse)
async def suggest_categories(texts: List[str]) -> CategorySuggestionResponse:
    try:
        categories: List[str] = await classifier._suggest_categories_async(texts)
        return CategorySuggestionResponse(categories=categories)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate", response_model=ValidationResponse)
async def validate_classifications(validation_request: ValidationRequest) -> ValidationResponse:
    """Validate classification results and provide improvement suggestions"""
    try:
        # Convert samples to DataFrame
        df = pd.DataFrame([
            {
                "text": sample.text,
                "Category": sample.assigned_category,
                "Confidence": sample.confidence
            }
            for sample in validation_request.samples
        ])

        # Use the validate_results function from utils
        validation_report: str = validate_results(df, validation_request.text_columns, client)

        # Parse the validation report to extract structured information
        accuracy_score: Optional[float] = None
        misclassifications: Optional[List[Dict[str, Any]]] = None
        suggested_improvements: Optional[List[str]] = None

        # Extract accuracy score if present
        if "accuracy" in validation_report.lower():
            try:
                accuracy_str = validation_report.lower().split("accuracy")[1].split("%")[0].strip()
                accuracy_score = float(accuracy_str) / 100
            except:
                pass

        # Extract misclassifications
        misclassifications = [
            {"text": sample.text, "current_category": sample.assigned_category}
            for sample in validation_request.samples
            if sample.confidence < 70
        ]

        # Extract suggested improvements
        suggested_improvements = [
            "Review low confidence classifications",
            "Consider adding more training examples",
            "Refine category definitions"
        ]

        return ValidationResponse(
            validation_report=validation_report,
            accuracy_score=accuracy_score,
            misclassifications=misclassifications,
            suggested_improvements=suggested_improvements
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/improve-classification", response_model=ImprovementResponse)
async def improve_classification_endpoint(request: ImprovementRequest) -> ImprovementResponse:
    """Improve classification based on validation report"""
    try:
        # Convert JSON DataFrame back to pandas DataFrame
        df = pd.DataFrame.from_dict(request.df)

        # Call the improve_classification function
        improved_df, new_validation, success, updated_categories = await improve_classification(
            df=df,
            validation_report=request.validation_report,
            text_columns=request.text_columns,
            categories=request.categories,
            classifier_type=request.classifier_type,
            show_explanations=request.show_explanations,
            file=request.file_path
        )

        # Convert improved DataFrame to JSON
        improved_df_json = improved_df.to_dict() if improved_df is not None else None

        return ImprovementResponse(
            improved_df=improved_df_json,
            new_validation_report=new_validation,
            success=success,
            updated_categories=updated_categories
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True) 