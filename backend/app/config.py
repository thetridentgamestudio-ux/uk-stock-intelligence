import os
from functools import lru_cache

from pydantic_settings import BaseSettings

# Project root — always correct regardless of where commands are run from
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ENV_FILE = os.path.join(_PROJECT_ROOT, ".env")
_DB_DEFAULT = f"sqlite:///{os.path.join(_PROJECT_ROOT, 'ukstocks.db')}"
_MODEL_DEFAULT = os.path.join(_PROJECT_ROOT, "models", "xgboost_model.pkl")


class Settings(BaseSettings):
    database_url: str = _DB_DEFAULT
    anthropic_api_key: str = ""
    model_path: str = _MODEL_DEFAULT

    class Config:
        env_file = _ENV_FILE


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
