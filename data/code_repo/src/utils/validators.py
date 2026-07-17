"""Validation helpers."""
import re
from typing import Optional

def validate_email(email: str) -> bool:
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email))

def validate_password(password: str) -> Optional[str]:
    if len(password) < 8:
        return "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return "Password must include an uppercase letter."
    if not re.search(r"[a-z]", password):
        return "Password must include a lowercase letter."
    if not re.search(r"\d", password):
        return "Password must include a digit."
    return None

def sanitize_input(text: str) -> str:
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"[\"\';]", "", text)
    return text.strip()
