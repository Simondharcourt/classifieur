from litellm import OpenAI
import os
from dotenv import load_dotenv
from typing import Optional, Tuple, Any

# Load environment variables
load_dotenv()

# Initialize client as None
client: Optional[OpenAI] = None

def get_client() -> Optional[OpenAI]:
    """Get the OpenAI client instance"""
    global client
    return client

def initialize_client(api_key: Optional[str] = None) -> Tuple[bool, str]:
    """Initialize the OpenAI client with an API key"""
    global client
    import logging
    
    # Use provided API key or get from environment
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        logging.error("No API key provided")
        return False, "No API key provided"
    
    try:
        client = OpenAI(api_key=api_key)
        # Test the connection with a simple request
        response: Any = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5,
        )
        logging.info("API Key updated and verified successfully")
        return True, "API Key updated and verified successfully"
    except Exception as e:
        client = None
        error_message: str = f"Failed to initialize client: {str(e)}"
        logging.error(error_message)
        return False, error_message