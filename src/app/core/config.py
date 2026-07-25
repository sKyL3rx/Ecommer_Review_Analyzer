from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Product Review Intelligence API"

    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql+psycopg://ecom:ecom@localhost:5432/ecom_review"

    vllm_base_url: str = "http://127.0.0.1:8001/v1"
    vllm_api_key: str = "demo-key"

    summary_model_name: str = "summary-sft"
    summary_model_version: str = "summary-sft"

    use_fake_summarizer: bool = False
    use_fake_sentiment: bool = False
    auto_create_tables: bool = False

    redis_cache_ttl_seconds: int = 3600

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        protected_namespaces=("settings_",),
    )


settings = Settings()
