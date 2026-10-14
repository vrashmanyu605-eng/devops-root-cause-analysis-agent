import os
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------
# Loki
# ------------------------------------------------------------------
LOKI_URL: str = os.getenv("LOKI_URL")
LOKI_QUERY: str = os.getenv("LOKI_QUERY")
POLL_INTERVAL: float = float(os.getenv("POLL_INTERVAL", "2"))   # seconds between polls
LOKI_LIMIT: int = int(os.getenv("LOKI_LIMIT", "1000"))          # max log lines per fetch

# ------------------------------------------------------------------
# Batching  (high-throughput tuning)
# ------------------------------------------------------------------
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "50"))            # flush when buffer >= N logs
BATCH_TIMEOUT: float = float(os.getenv("BATCH_TIMEOUT", "5"))  # flush every N seconds regardless
QUEUE_MAXSIZE: int = int(os.getenv("QUEUE_MAXSIZE", "500"))     # back-pressure cap

# ------------------------------------------------------------------
# LiteLLM / Gemini
# ------------------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
LITELLM_MODEL: str = os.getenv("LITELLM_MODEL", "gemini/gemini-2.5-flash")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "8500"))

# ------------------------------------------------------------------
# PostgreSQL
# ------------------------------------------------------------------
PG_HOST: str = os.getenv("PG_DB_HOST")
PG_PORT: int = int(os.getenv("PG_DB_PORT"))
PG_USER: str = os.getenv("PG_DB_USER")
PG_PASSWORD: str = os.getenv("PG_DB_PASSWORD")
PG_DB: str = os.getenv("PG_DB_NAME")
PG_POOL_MIN: int = int(os.getenv("PG_POOL_MIN", "2"))
PG_POOL_MAX: int = int(os.getenv("PG_POOL_MAX", "10"))
