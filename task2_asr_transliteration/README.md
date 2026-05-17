# Indic Translation & ASR System

A two-part research and engineering project:
1. **Task 1** — Evaluation of five English-to-Tamil translation models across translation quality, token metrics, and Indic tokenizer behavior
2. **Task 2** — ASR-based transcription and transliteration system using Whisper and indic-transliteration, containerised with Docker

**Gourilakshmi S **

---

## Project Structure

```
indic-translation-asr-project/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── data/
│   ├── raw/                              # Source dataset (opus-100 en-ta, 500 pairs)
│   ├── processed/translated_outputs/
│   └── results/                          # sacreBLEU scores, token stats, plots
│
├── docs/
│   ├── architecture/                     # System architecture diagrams
│   └── reports/                          # Project summary PDF/DOCX
│
├── task1_translation_evaluation/
│   ├── part_a_batch_translation/         # IndicTrans2 translation + sacreBLEU
│   ├── part_b_token_analysis/            # 5-model token metric comparison
│   └── part_c_indic_token_behavior/      # Raw Tamil Unicode tokenizer analysis
│
└── task2_asr_transliteration/
    ├── app/
    │   ├── main.py                       # Entry point — launches Gradio
    │   ├── asr_pipeline.py               # Whisper ASR wrapper
    │   ├── transliteration.py            # Script conversion helpers
    │   ├── buffer_manager.py             # Queue-based audio buffer
    │   ├── interface.py                  # Gradio UI definition
    │   └── utils.py                      # File I/O & misc helpers
    ├── models/
    │   └── model_config.py               # Central config (model IDs, scripts)
    ├── sample_inputs/
    │   └── sample.wav
    ├── outputs/
    │   ├── transcripts/
    │   └── transliterations/
    ├── tests/
    │   └── test_pipeline.py
    ├── Dockerfile
    ├── docker-compose.yml
    └── requirements.txt
```



---

## Task 2: ASR-Based Transcription and Transliteration System

An end-to-end pipeline that transcribes audio using OpenAI Whisper and converts the output between Indic scripts using indic-transliteration. A Gradio web interface makes the system easy to use, containerised with Docker.

### System Architecture

```
Audio Input
     ↓
Buffer Queue  (queue.Queue — chunked / async)
     ↓
ASR Module    (openai/whisper-small or whisper-medium)
     ↓
Transcript
     ↓
Transliteration Engine  (indic-transliteration)
     ↓
User Interface  (Gradio — port 7860)
```

### Quick Start

**Option A — Local **

```bash
cd task2_asr_transliteration
pip install -r requirements.txt
python app/main.py
# open http://localhost:7860
```

**Option B — Docker**

```bash
docker build -t asr-system .
docker run -p 7860:7860 asr-system
```

**Option C — Docker Compose (recommended)**

```bash
docker compose up --build
```

### Configuration

All tunable settings live in `models/model_config.py`:

| Setting | Default | Description |
|---|---|---|
| `ASR_MODEL_ID` | `openai/whisper-small` | Whisper variant to use |
| `LANGUAGE` | `"ta"` | ISO 639-1 language hint (Tamil) |
| `TASK` | `"transcribe"` | `"transcribe"` or `"translate"` |
| `CHUNK_DURATION_S` | `30.0` | Audio chunk length in seconds |
| `BUFFER_MAXSIZE` | `20` | Max chunks held in buffer |
| `SOURCE_SCRIPT` | `"tamil"` | Default source script |
| `TARGET_SCRIPT` | `"latin"` | Default target script |

### Supported Scripts

`tamil`, `devanagari`, `telugu`, `kannada`, `malayalam`, `latin`, `itrans`, `iast`, `slp1`

### Running Tests

```bash
pytest tests/test_pipeline.py -v
```

### Key Dependencies

| Package | Purpose |
|---|---|
| `transformers` | Whisper model loading and inference |
| `librosa` | Audio loading and resampling |
| `indic-transliteration` | Script conversion |
| `gradio` | Web interface |
| `torch` | Deep learning backend |

---

## License

MIT License — see `LICENSE` for details.
