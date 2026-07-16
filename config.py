from dotenv import load_dotenv
import os

load_dotenv() 
api_key = os.getenv("OPENAI_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")