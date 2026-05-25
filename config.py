import os
from dotenv import load_dotenv
load_dotenv()

API_KEY       = os.getenv("ANTHROPIC_API_KEY")
MODEL_FAST    = "claude-haiku-4-5-20251001"
MODEL_MAIN    = "claude-sonnet-4-5"
MAX_TOKENS    = 2048
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 120
TOP_K         = 4
