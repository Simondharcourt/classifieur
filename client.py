from litellm import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize client as None
client = None

def get_client():
    """Get the OpenAI client instance"""
    global client
    return client

def initialize_client(api_key=None):
    """Initialize the OpenAI client with an API key"""
    global client
    
    # Use provided API key or get from environment
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        return False, "No API key provided"
    
    try:
        client = OpenAI(api_key=api_key)
        # Test the connection with a simple request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5,
        )
        return True, "API Key updated and verified successfully"
    except Exception as e:
        client = None
        return False, f"Failed to initialize client: {str(e)}" 