# Part C: Indic Token Behavior Analysis — Observations

## Objective
This part does not perform translation. It isolates each model's **tokenizer only** and
feeds raw Tamil Unicode sentences directly, measuring how well each tokenizer handles Tamil
script at the character and subword level. No model inference is run.

## Dataset
20 Tamil sentences covering a range of domains: education, politics, science, agriculture,
culture, and healthcare. Sentences vary in length and morphological complexity, including
agglutinative forms (e.g., `மருத்துவமனையில்` = "in the hospital").

## Models / Tokenizers Analysed
| Label | Tokenizer Type | Vocab Size |
|---|---|---|
| IndicTrans2 | SentencePiece (Indic-aware) | 32322 |
| NLLB-200 | SentencePiece (multilingual) | 256204 |
| Helsinki-dra | SentencePiece (Marian) |62952 |
| mBART-50 | SentencePiece (mBART) | 250054 |
| MADLAD-400 | SentencePiece (T5-based) | 256000 |

## Architectural Changes / Fixes Applied

**1. IndicTrans2 tokenizer requires BCP-47 language tag prefix.**
The tokenizer's `_src_tokenize` method splits input on the first two space-delimited tokens
as `src_lang` and `tgt_lang`, then validates them against an internal `LANGUAGE_TAGS` list.
Calling it on plain Tamil text raises:
```
ValueError: not enough values to unpack (expected 3, got 1)
```
Fixed by prepending `tam_Taml tam_Taml` before the Tamil sentence:
```python
if model_name == "IndicTrans2":
    processed_sentence = f"tam_Taml tam_Taml {sentence}"
```
The two language tag tokens are excluded from all metric calculations since they are
artefacts of the format, not genuine tokenization of the Tamil content.

**2. Demo word cell required the same prefix fix.**
The standalone demo cell (`மருத்துவமனையில்`) also calls the IndicTrans2 tokenizer directly.
Fixed by applying the same prefix and slicing off the first two tokens before display:
```python
if cfg["name"] == "IndicTrans2":
    tokens = tokens[2:]
```

**3. Helsinki-dra: Tamil Unicode absent from vocabulary.**
`opus-mt-en-dra` was trained on romanised/transliterated Dravidian text. Its SentencePiece
vocabulary contains no Tamil Unicode characters. Result: all Tamil input collapses to
`_ | <unk>` — the word boundary marker plus a single unknown token regardless of word
length. This is a valid and informative result, not an error.

## Key Observations

**1. IndicTrans2 achieves the lowest token count and highest chars-per-token.**
Its SentencePiece model was trained on Indic scripts with Tamil-specific subword units.
Tamil agglutinative words like `மருத்துவமனையில்` (16 chars) are represented in far fewer
tokens compared to multilingual models, reducing both fragmentation and memory footprint.

**2. Helsinki-dra has a vocab_coverage of ~0 and unk_rate of 1.0 for Tamil Unicode.**
This definitively shows the model was never trained on native Tamil script. Any Tamil text
fed to this tokenizer is completely opaque — every character maps to `<unk>`. This explains
its poor translation quality observed in Part B.

**3. NLLB-200 and MADLAD-400 handle Tamil better than mBART-50.**
Both were explicitly trained with Tamil Unicode data in large multilingual corpora, giving
them meaningful subword coverage. mBART-50 has Tamil support (`ta_IN`) but its SentencePiece
vocabulary is less Tamil-dense than NLLB or MADLAD, resulting in higher fragmentation.

**4. Tamil's agglutinative morphology drives high fragmentation in general models.**
Tamil suffixes encode tense, person, number, gender, and case within a single word. A model
trained primarily on European languages sees these suffix chains as unseen character
sequences and splits them into many 1–2 character fragments, inflating token counts and
Transformer attention costs quadratically.

**5. Unicode combining marks are a fragmentation signal.**
Tamil uses combining characters (matras, vowel signs in the `\u0B80–\u0BFF` range) that
attach to base consonants visually but can be split by tokenizers as separate tokens. The
`unicode_frag_rate` metric captures this — higher values mean the tokenizer is breaking
Tamil akshara units at combining-mark boundaries, which destroys linguistic coherence.

**6. Memory footprint scales directly with token count.**
Since each token is stored as a 32-bit integer (4 bytes), models with higher token counts
per sentence consume proportionally more memory in the Transformer's key-value cache.
IndicTrans2's compact Tamil representation gives it a meaningful efficiency advantage at
scale.

## Conclusion
Tokenizer design is the single most important factor in Tamil NLP quality. Models whose
SentencePiece vocabularies were built with Tamil Unicode data (IndicTrans2, NLLB-200,
MADLAD-400) handle Tamil gracefully with low fragmentation and near-zero unknown rates.
Models without Tamil Unicode in their training data (Helsinki-dra) are fundamentally
incapable of processing native Tamil script regardless of their translation architecture.
This analysis justifies the choice of IndicTrans2 as the primary model in Part A and
explains the quality ranking observed across models in Part B.
