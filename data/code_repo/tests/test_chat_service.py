"""Chat service tests."""
from src.services.chat_service import ChatService

def test_chat_returns_demo_response():
    service = ChatService()
    assert "Demo response" in service.chat("hello")
