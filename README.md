# Evaluation of Indic Translation Models
A research project evaluating five English-to-Tamil translation models across three analytical dimensions: translation quality, token-level comparative metrics, and deep Indic tokenizer behavior analysis. Also includes an ASR-based transcription and transliteration system for Indic languages.

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
├── task1_translation_evaluation/
│   ├── part_a_batch_translation/   # IndicTrans2 translation + sacreBLEU
│   ├── part_b_token_analysis/      # 5-model token metric comparison
│   └── part_c_indic_token_behavior/# Raw Tamil Unicode tokenizer analysis
└── task2_asr_transliteration/
    ├── app/                        # Core application modules
    ├── models/                     # Model configuration
    ├── tests/                      # pytest test suite
    ├── sample_inputs/              # Sample Tamil audio files
    ├── outputs/                    # Saved transcripts and transliterations
    ├── Dockerfile
    ├── docker-compose.yml
    └── requirements.txt
```

---

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

## Task 2: ASR-Based Transcription & Transliteration System
An end-to-end pipeline that transcribes Tamil audio using OpenAI Whisper and converts the output between Indic scripts using indic-transliteration, served via a Gradio web interface and containerized with Docker.

### Models Used
| Component | Model/Library | Role |
|---|---|---|
| ASR | `openai/whisper-small` | Primary transcription model |
| ASR (alt) | `openai/whisper-medium` | Higher accuracy alternative |
| Transliteration | `indic-transliteration` v2.3.x | Script conversion |
| Interface | `gradio` v6.x | Web UI |

### System Architecture
```
Audio Input
     ↓
Buffer Queue  (queue.Queue — chunked/async)
     ↓
ASR Module    (openai/whisper-small)
     ↓
Transcript    (Tamil text)
     ↓
Transliteration Engine  (indic-transliteration)
     ↓
Gradio Interface  (port 7860)
```

### Supported Scripts
`tamil`, `malayalam`, `devanagari`, `telugu`, `kannada`, `latin`, `itrans`, `iast`, `slp1`

### Part A — ASR Transcription
- Transcribes Tamil audio using Whisper with forced Tamil language decoding
- Uses `forced_decoder_ids` to lock language to Tamil and prevent auto-detection errors
- Supports both full-file and chunked buffer pipeline modes

### Part B — Transliteration
- Converts Tamil transcript to target Indic script using indic-transliteration
- Cross-Indic conversions (e.g. Tamil → Malayalam) routed through ITRANS as bridge
- Supports batch transliteration across all 9 scripts

### Part C — Dockerized Deployment
- Fully containerized with Dockerfile and docker-compose.yml
- Queue-based AudioBufferManager handles chunked audio with overflow protection and zero-padding
- Outputs saved as timestamped .txt files in outputs/transcripts/ and outputs/transliterations/

---

## Installation
```bash
git clone https://github.com/gourilaxmi/indic-translation-asr-project.git
cd indic-translation-asr-project
pip install -r requirements.txt
```

### IndicTransToolkit (required for Task 1 Parts A & B)
```bash
git clone https://github.com/AI4Bharat/IndicTrans2.git
cd IndicTrans2/huggingface_interface
git clone https://github.com/VarunGumma/IndicTransToolkit.git
cd IndicTransToolkit
pip install --editable ./
```

---

## Running the Notebooks (Task 1)
All notebooks are designed for **Google Colab** with Google Drive mounted at:
```
/content/drive/MyDrive/indic-translation-asr-project/
```
Run in order:
1. `task1_translation_evaluation/part_a_batch_translation/part_a_translation_evaluation.ipynb`
2. `task1_translation_evaluation/part_b_token_analysis/part_b_token_analysis.ipynb`
3. `task1_translation_evaluation/part_c_indic_token_behavior/part_c_indic_token_analysis.ipynb`

---

## Running Task 2 Locally
```bash
cd task2_asr_transliteration
pip install -r requirements.txt

# Mac/Linux
PYTHONPATH=app:models python app/main.py

# Windows
$env:PYTHONPATH="app;."
python app\main.py
```
Open `http://localhost:7860` in your browser.

### Docker
```bash
cd task2_asr_transliteration
docker build -t asr-system .
docker run -p 7860:7860 asr-system
```

### Tests
```bash
cd task2_asr_transliteration
pytest tests/test_pipeline.py -v
```

---

## Key Findings
### Task 1
- **IndicTrans2** achieves the lowest token expansion ratio and highest sacreBLEU — its Indic-trained SentencePiece vocabulary is the primary driver
- **Helsinki-dra** maps all Tamil Unicode to `<unk>` (trained on romanised text only) — unk_rate ≈ 1.0
- **NLLB-200 and MADLAD-400** are strong general multilingual alternatives with meaningful Tamil Unicode coverage
- Tamil's agglutinative morphology causes severe fragmentation in models without native Tamil vocabulary
- Tokenizer vocabulary design predicts translation quality more reliably than model size

### Task 2
- Whisper's `forced_decoder_ids` with `language=ta` significantly improves Tamil transcription accuracy over auto-detection
- Direct cross-Indic script conversion requires an ITRANS bridge for correct output
- Queue-based buffering with zero-padding enables stable processing of variable-length audio

---

## Requirements
See `requirements.txt` for the full dependency list. Task 2 has its own `requirements.txt` inside `task2_asr_transliteration/`.

---

## License
MIT License — see `LICENSE` for details.
