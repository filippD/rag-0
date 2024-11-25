from dotenv import load_dotenv
import os

load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
