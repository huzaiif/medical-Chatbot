import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    with open("available_models.txt", "w", encoding="utf-8") as f:
        f.write("API Key not found in .env")
else:
    genai.configure(api_key=api_key)
    try:
        models = list(genai.list_models())
        with open("available_models.txt", "w", encoding="utf-8") as f:
            f.write(f"Found {len(models)} models:\n")
            for m in models:
                f.write(f"Name: {m.name}\n")
                f.write(f"Supported generation methods: {m.supported_generation_methods}\n")
                f.write("-" * 20 + "\n")
    except Exception as e:
        with open("available_models.txt", "w", encoding="utf-8") as f:
            f.write(f"Error listing models: {e}")
