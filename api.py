from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    model = genai.GenerativeModel(
        "gemini-flash-lite-latest",
        system_instruction=
        "You are a helpful medical assistant. Be safe, accurate and include disclaimer."
    )

    response = model.generate_content(req.message)
    return {"reply": response.text}


@app.get("/")
def home():
    return {"status": "API running"}

