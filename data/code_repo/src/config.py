"""Application configuration."""
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = "mysql://root:password@localhost:3306/cloudengine"
    secret_key: str = "your-secret-key"

settings = Settings()
