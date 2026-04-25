from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Product Review Intelligence API"

    redis_url: str = "redis://localhost:6379/0"

    vllm_base_url: str = "http://127.0.0.1:8001/v1"
    vllm_api_key: str = "demo-key"
    summary_model_name: str = "summary-sft"

    app_model_version: str = "summary-sft-candidate"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

settings = Settings()