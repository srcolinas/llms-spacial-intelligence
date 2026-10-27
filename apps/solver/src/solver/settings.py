import functools
from typing import Literal

import pydantic_settings


class Settings(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(
        env_file=".env",
        env_prefix="SOLVER_",
        extra="ignore",
        frozen=True,
    )

    loglevel: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    """Logging level."""

    concurrency: int = 10
    """Maximum number of questions solved in parallel."""


@functools.cache
def settings() -> Settings:
    return Settings()
