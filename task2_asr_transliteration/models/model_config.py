"""
model_config.py - Central configuration for ASR and transliteration settings.
Edit this file to switch models or adjust pipeline parameters.
"""

# ------------------------------------------------------------------
# ASR (Whisper) configuration
# ------------------------------------------------------------------

# Options: "openai/whisper-small"  (faster, less accurate)
#          "openai/whisper-medium" (slower, more accurate)
ASR_MODEL_ID: str = "openai/whisper-small"

# Audio sampling rate expected by Whisper
SAMPLE_RATE: int = 16_000

# Whisper language hint (ISO 639-1 code)
# Set to None to let Whisper auto-detect the language
LANGUAGE: str = "ta"   # Tamil

# "transcribe" keeps the original language; "translate" converts to English
TASK: str = "transcribe"

# ------------------------------------------------------------------
# Buffer / chunking configuration
# ------------------------------------------------------------------

# Duration of each audio chunk fed to the ASR model (seconds)
CHUNK_DURATION_S: float = 30.0

# Maximum number of chunks held in the buffer at once
BUFFER_MAXSIZE: int = 20

# ------------------------------------------------------------------
# Transliteration configuration
# ------------------------------------------------------------------

# Default source and target scripts for transliteration
# Supported values: "tamil", "devanagari", "telugu", "kannada",
#                   "malayalam", "latin", "itrans", "iast", "slp1"
SOURCE_SCRIPT: str = "tamil"
TARGET_SCRIPT: str = "latin"
