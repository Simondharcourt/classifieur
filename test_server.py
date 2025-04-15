import requests
import json
from typing import List, Dict, Any, Optional

BASE_URL: str = "http://localhost:8000"

def test_health_check() -> None:
    """Test the health check endpoint"""
    response: requests.Response = requests.get(f"{BASE_URL}/health")
    print("\nHealth check response:")
    print(json.dumps(response.json(), indent=2))

def test_model_info() -> None:
    """Test the model info endpoint"""
    response: requests.Response = requests.get(f"{BASE_URL}/model-info")
    print("\nModel info response:")
    print(json.dumps(response.json(), indent=2))

def test_classify_text() -> None:
    # Load emails from CSV file
    import csv
    
    emails: List[Dict[str, str]] = []
    with open("examples/emails.csv", "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            emails.append(row)
    
    # Test with default categories using email content
    for email in emails[:5]:
        response: requests.Response = requests.post(
            f"{BASE_URL}/classify",
            json={"text": email["contenu"]}
        )
        print(f"Classification of email '{email['sujet']}' with default categories:")
        print(json.dumps(response.json(), indent=2))

def test_classify_batch() -> None:
    """Test the batch classification endpoint"""
    # Load emails from CSV file
    import csv
    
    emails: List[Dict[str, str]] = []
    with open("examples/emails.csv", "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            emails.append(row)
    
    # Use the first 5 emails for batch classification
    texts: List[str] = [email["contenu"] for email in emails[:5]]
    response: requests.Response = requests.post(
        f"{BASE_URL}/classify-batch",
        json={"texts": texts}
    )
    print("\nBatch classification results:")
    print(json.dumps(response.json(), indent=2))

def test_suggest_categories() -> None:
    # Load reviews from CSV file
    import csv
    
    texts: List[str] = []
    with open("examples/reviews.csv", "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            texts.append(row["text"])
    
    # Use the first few reviews for testing
    texts = texts[:5]
    response: requests.Response = requests.post(
        f"{BASE_URL}/suggest-categories",
        json=texts
    )
    print("\nSuggested categories:")
    print(json.dumps(response.json(), indent=2))

def test_validate_classifications() -> None:
    """Test the validation endpoint"""
    # Load emails from CSV file
    import csv
    
    emails: List[Dict[str, str]] = []
    with open("examples/emails.csv", "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            emails.append(row)
    
    # Create validation samples from the first 5 emails
    samples: List[Dict[str, Any]] = []
    for email in emails[:5]:
        # First classify the email
        classify_response: requests.Response = requests.post(
            f"{BASE_URL}/classify",
            json={"text": email["contenu"]}
        )
        classification: Dict[str, Any] = classify_response.json()
        
        # Create a validation sample
        samples.append({
            "text": email["contenu"],
            "assigned_category": classification["category"],
            "confidence": classification["confidence"]
        })
    
    # Get current categories
    categories_response: requests.Response = requests.post(
        f"{BASE_URL}/suggest-categories",
        json=[email["contenu"] for email in emails[:5]]
    )
    response_data: Dict[str, Any] = categories_response.json()
    current_categories: List[str] = response_data["categories"]  # Extract categories from the response
    
    # Send validation request
    validation_request: Dict[str, Any] = {
        "samples": samples,
        "current_categories": current_categories,
        "text_columns": ["text"]
    }
    
    response: requests.Response = requests.post(
        f"{BASE_URL}/validate",
        json=validation_request
    )
    print("\nValidation results:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    print("Testing FastAPI server endpoints...")
    test_health_check()
    test_model_info()
    test_classify_text()
    test_classify_batch()
    test_suggest_categories()
    test_validate_classifications() 