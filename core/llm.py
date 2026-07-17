from __future__ import annotations
import json
from typing import List, Dict
from openai import OpenAI
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
        )
        self.model = LLM_MODEL
        self.has_api_key = bool(LLM_API_KEY.strip())

    def chat(self, messages: list[dict], temperature: float = 0.1,
             max_tokens: int = 4096) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def chat_with_tools(self, messages: list[dict], tools: list[dict],
                        temperature: float = 0.1) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=temperature,
            )

            message = response.choices[0].message

            result = {
                "content": message.content or "",
                "tool_calls": []
            }

            if message.tool_calls:
                for tc in message.tool_calls:
                    result["tool_calls"].append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    })

            return result
        except Exception as e:
            return {
                "content": "",
                "error": str(e),
                "tool_calls": []
            }
