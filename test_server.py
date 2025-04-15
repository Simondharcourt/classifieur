import requests
import json

BASE_URL = "http://localhost:8000"

def test_classify_text():
    # Test with default categories
    response = requests.post(
        f"{BASE_URL}/classify",
        json={"text": "This is a sample text about technology and innovation."}
    )
    print("Classification with default categories:")
    print(json.dumps(response.json(), indent=2))

    # Test with custom categories
    response = requests.post(
        f"{BASE_URL}/classify",
        json={
            "text": "This is a sample text about technology and innovation.",
            "categories": ["Technology", "Business", "Science", "Sports"]
        }
    )
    print("\nClassification with custom categories:")
    print(json.dumps(response.json(), indent=2))

def test_suggest_categories():
    texts = [
        "This is a text about artificial intelligence and machine learning.",
        "A new breakthrough in quantum computing has been announced.",
        "The latest smartphone features innovative camera technology."
    ]
    
    response = requests.post(
        f"{BASE_URL}/suggest-categories",
        json=texts
    )
    print("\nSuggested categories:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    print("Testing FastAPI server endpoints...")
    test_classify_text()
    test_suggest_categories() 