"""Support chat service."""
from openai import OpenAI
from src.config import settings

class ChatService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.secret_key)

    def chat(self, message: str) -> str:
        return f"Demo response for: {message}"
