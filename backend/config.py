import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Access keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")