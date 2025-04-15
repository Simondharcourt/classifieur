import requests
import json

BASE_URL = "http://localhost:8000"

def test_classify_text():
    # Load emails from CSV file
    import csv
    
    emails = []
    with open("examples/emails.csv", "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            emails.append(row)
    
    # Test with default categories using email content
    email = emails[0]  # First email
    response = requests.post(
        f"{BASE_URL}/classify",
        json={"text": email["contenu"]}
    )
    print(f"Classification of email '{email['sujet']}' with default categories:")
    print(json.dumps(response.json(), indent=2))

    # Test with custom categories using another email
    email = emails[2]  # Third email
    response = requests.post(
        f"{BASE_URL}/classify",
        json={
            "text": email["contenu"],
            "categories": ["Urgent", "Technique", "Commercial", "Personnel"]
        }
    )
    print(f"\nClassification of email '{email['sujet']}' with custom categories:")
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