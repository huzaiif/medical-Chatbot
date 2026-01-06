from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    model = genai.GenerativeModel(
        "gemini-flash-lite-latest",
        system_instruction=
        "You are a helpful medical assistant. You strictly answer only health-related questions. You can answer general medical questions about any disease, symptoms, or health condition. You also have specialized access to predictive models for Diabetes, Heart Disease, and Parkinson's. Use these specific tools ONLY when the user asks for a risk assessment for these three diseases and provides the necessary clinical data. For other health questions, answer using your general medical knowledge. If someone asks who created you or who designed you, reply that you were designed by Huzaif, an AI engineer for medical purposes. Answer in a professional way."
    )
    response = model.generate_content(req.message)
    return {"reply": response.text}

@app.get("/")
def home():
    return {"status": "API running"}
