from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

ressponse = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents="Jelaskan apa itu RAG secara singkat."
)

print(ressponse.text)