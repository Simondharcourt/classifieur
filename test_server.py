import requests
import json
from typing import List, Dict, Any, Optional

BASE_URL: str = "http://localhost:8000"

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

if __name__ == "__main__":
    print("Testing FastAPI server endpoints...")
    test_classify_text()
    test_suggest_categories() 