import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# GitHub Models uses a GitHub token + Azure inference endpoint
# Falls back to direct OpenAI if GITHUB_TOKEN is not set
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

GITHUB_MODELS_ENDPOINT = "https://models.inference.ai.azure.com"

# Primary client (text generation + embeddings)
API_KEY = GITHUB_TOKEN if GITHUB_TOKEN else OPENAI_API_KEY
BASE_URL = GITHUB_MODELS_ENDPOINT if GITHUB_TOKEN else None

# Vision client — GitHub Models doesn't support base64 image data URIs.
# Set OPENAI_API_KEY alongside GITHUB_TOKEN to enable full image description.
# Without it, images are indexed with a metadata-only placeholder.
VISION_API_KEY = OPENAI_API_KEY or GITHUB_TOKEN
VISION_BASE_URL = None if OPENAI_API_KEY else GITHUB_MODELS_ENDPOINT
VISION_ENABLED = bool(OPENAI_API_KEY)  # True only when direct OpenAI key is present

OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_RETRIEVAL_RESULTS = 6
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.35"))  # discard chunks below this
IMAGE_MAX_SIZE = (1024, 1024)

Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
