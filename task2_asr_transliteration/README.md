# Task 2 — ASR-Based Transcription and Transliteration System

An end-to-end pipeline that transcribes audio using **OpenAI Whisper** and
converts the output between Indic scripts using **indic-transliteration**.
A **Gradio** web interface makes the system easy to use, and the whole stack
is containerised with **Docker**.

---

## System Architecture

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
=======
# Evaluation of Indic Translation Models

A research project evaluating five English-to-Tamil translation models across three analytical dimensions: translation quality, token-level comparative metrics, and deep Indic tokenizer behavior analysis.

---

## Project Structure

```
indic-translation-asr-project/
├── data/
│   ├── raw/                        # Source dataset (opus-100 en-ta, 500 pairs)
│   ├── processed/translated_outputs/
│   └── results/                    # sacreBLEU scores, token stats, plots
├── docs/
│   ├── architecture/               # System architecture diagrams
│   └── reports/                    # Project summary PDF
└── task1_translation_evaluation/
    ├── part_a_batch_translation/   # IndicTrans2 translation + sacreBLEU
    ├── part_b_token_analysis/      # 5-model token metric comparison
    └── part_c_indic_token_behavior/# Raw Tamil Unicode tokenizer analysis
>>>>>>> a18e44dee0b50e2b89392d17aefb408aba4d1f24
```

---

<<<<<<< HEAD
## Repository Layout

```
task2_asr_transliteration/
├── app/
│   ├── main.py              # Entry point — launches Gradio
│   ├── asr_pipeline.py      # Whisper ASR wrapper
│   ├── transliteration.py   # Script conversion helpers
│   ├── buffer_manager.py    # Queue-based audio buffer
│   ├── interface.py         # Gradio UI definition
│   └── utils.py             # File I/O & misc helpers
├── models/
│   └── model_config.py      # Central config (model IDs, scripts, …)
├── sample_inputs/
│   └── sample.wav           # Drop test audio files here
├── outputs/
│   ├── transcripts/         # Saved transcription .txt files
│   └── transliterations/    # Saved transliteration .txt files
├── tests/
│   └── test_pipeline.py     # pytest test suite
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start

### Option A — Local (no Docker)

```bash
# 1. Clone and enter the directory
git clone <repo-url>
cd task2_asr_transliteration

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python app/main.py
# → open http://localhost:7860 in your browser
```

### Option B — Docker

```bash
# Build
docker build -t asr-system .

# Run
docker run -p 7860:7860 asr-system
```

### Option C — Docker Compose (recommended)

```bash
docker compose up --build
```

This also mounts `./outputs` and caches the Hugging Face model weights
across restarts.

---

## Configuration

All tunable settings live in **`models/model_config.py`**:

| Setting | Default | Description |
|---|---|---|
| `ASR_MODEL_ID` | `openai/whisper-small` | Whisper variant to use |
| `LANGUAGE` | `"ta"` | ISO 639-1 language hint (Tamil) |
| `TASK` | `"transcribe"` | `"transcribe"` or `"translate"` |
| `CHUNK_DURATION_S` | `30.0` | Audio chunk length (seconds) |
| `BUFFER_MAXSIZE` | `20` | Max chunks held in buffer |
| `SOURCE_SCRIPT` | `"tamil"` | Default source script |
| `TARGET_SCRIPT` | `"latin"` | Default target script |

Switch to `openai/whisper-medium` for better accuracy at the cost of speed.

---

## Supported Scripts

`tamil`, `devanagari`, `telugu`, `kannada`, `malayalam`, `latin`, `itrans`,
`iast`, `slp1`

---

## Running Tests

```bash
pytest tests/test_pipeline.py -v
```

Tests cover the buffer manager, transliteration helpers, and file utilities.
The ASR model itself is not loaded during unit tests to keep them fast.

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `transformers` | Whisper model loading & inference |
| `librosa` | Audio loading and resampling |
| `indic-transliteration` | Script conversion |
| `gradio` | Web interface |
| `torch` | Deep learning backend |
=======
## Task 1: Evaluation of Indic Translation Models

### Models Used

| Model | HuggingFace ID | Role |
|---|---|---|
| IndicTrans2 | `ai4bharat/indictrans2-en-indic-1B` | Primary translation model (Part A) |
| NLLB-200 | `facebook/nllb-200-distilled-600M` | Multilingual baseline |
| Helsinki-dra | `Helsinki-NLP/opus-mt-en-dra` | Dravidian-family model |
| mBART-50 | `facebook/mbart-large-50-many-to-many-mmt` | Explicit ta_IN support |
| MADLAD-400 | `google/madlad400-3b-mt` | 400-language T5-based model |

### Part A — Batch Translation & sacreBLEU
- Translates 500 English sentences to Tamil using IndicTrans2
- Evaluates using `sacrebleu` corpus and sentence BLEU
- Outputs: `translation_outputs.csv`, `sacrebleu_results.csv`, BLEU distribution plot

### Part B — Token-Based Comparative Analysis
- All 5 models translate 15 English sentences (3 complexity tiers)
- Computes: source/target token counts, expansion ratio, avg word length, subword fragmentation, unknown token rate
- Outputs: `token_counts.csv`, `engineered_features.csv`, 4 comparative plots

### Part C — Indic Token Behavior Analysis
- Tokenizers only (no model inference) on 20 raw Tamil sentences
- Measures: avg chars/token, unicode fragmentation rate, subword fragmentation, UNK rate, vocab coverage, memory footprint
- Outputs: `tokenization_comparison.csv`, `tamil_token_patterns.csv`, 6 analysis plots

---

## Installation

```bash
git clone https://github.com/gourilaxmi/indic-translation-asr-project.git
cd indic-translation-asr-project
pip install -r requirements.txt
```

### IndicTransToolkit (required for Part A & B)

```bash
git clone https://github.com/AI4Bharat/IndicTrans2.git
cd IndicTrans2/huggingface_interface
git clone https://github.com/VarunGumma/IndicTransToolkit.git
cd IndicTransToolkit
pip install --editable ./
```

---

## Running the Notebooks

All notebooks are designed for **Google Colab** with Google Drive mounted at:
```
/content/drive/MyDrive/indic-translation-asr-project/
```

Run in order:
1. `task1_translation_evaluation/part_a_batch_translation/part_a_translation_evaluation.ipynb`
2. `task1_translation_evaluation/part_b_token_analysis/part_b_token_analysis.ipynb`
3. `task1_translation_evaluation/part_c_indic_token_behavior/part_c_indic_token_analysis.ipynb`

---

## Key Findings

- **IndicTrans2** achieves the lowest token expansion ratio and highest sacreBLEU — its Indic-trained SentencePiece vocabulary is the primary driver
- **Helsinki-dra** maps all Tamil Unicode to `<unk>` (trained on romanised text only) — unk_rate ≈ 1.0
- **NLLB-200 and MADLAD-400** are strong general multilingual alternatives with meaningful Tamil Unicode coverage
- Tamil's agglutinative morphology causes severe fragmentation in models without native Tamil vocabulary
- Tokenizer vocabulary design predicts translation quality more reliably than model size

---

## Requirements

See `requirements.txt` for the full dependency list.

---

## License

MIT License — see `LICENSE` for details.
>>>>>>> a18e44dee0b50e2b89392d17aefb408aba4d1f24
