import os

from dotenv import load_dotenv

load_dotenv()

GOOGLE_CLOUD_PROJECT: str | None = os.getenv("GOOGLE_CLOUD_PROJECT")

SPEECH_LOCATION = os.getenv("SPEECH_LOCATION", "eu")
SPEECH_MODEL = os.getenv("SPEECH_MODEL", "chirp_3")
SPEECH_LANGUAGE = os.getenv("SPEECH_LANGUAGE", "es-ES")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")
GEMINI_LOCATION = os.getenv("GEMINI_LOCATION", "us-central1")

TTS_VOICE = os.getenv("TTS_VOICE", "es-ES-Standard-A")

LLM_DEBOUNCE_MS = int(os.getenv("LLM_DEBOUNCE_MS", "200"))
LLM_DEBOUNCE_MIN_WORDS = int(os.getenv("LLM_DEBOUNCE_MIN_WORDS", "2"))
LLM_FINAL_GRACE_MS = int(os.getenv("LLM_FINAL_GRACE_MS", "400"))
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "20"))

BARGE_IN_THRESHOLD = int(os.getenv("BARGE_IN_THRESHOLD", "300"))

SERVER_LOG_DIR = os.getenv("SERVER_LOG_DIR", "logs")
SERVER_LOG_FILE = os.getenv("SERVER_LOG_FILE", "server.log")
SERVER_LOG_MAX_BYTES = int(os.getenv("SERVER_LOG_MAX_BYTES", str(5 * 1024 * 1024)))
SERVER_LOG_BACKUP_COUNT = int(os.getenv("SERVER_LOG_BACKUP_COUNT", "5"))
SERVER_LOG_LEVEL = os.getenv("SERVER_LOG_LEVEL", "INFO").upper()
