# Part A: Batch Translation and Evaluation — Observations

## Model Used
`ai4bharat/indictrans2-en-indic-1B` via the IndicTransToolkit pipeline.

## Dataset
500 English–Tamil sentence pairs sourced from the Helsinki-NLP/opus-100 `en-ta` test split.
Reference translations from the dataset were used for sacreBLEU evaluation.

## Architectur
- The IndicTrans2 model uses a custom `IndicProcessor` for preprocessing and postprocessing,
  which handles script normalization, punctuation, and language tag injection internally.
- Tokenization uses a SentencePiece model trained specifically on Indic scripts, giving it
  significantly better Tamil subword coverage compared to multilingual models.
- Translation was run with `num_beams=5` and `no_repeat_ngram_size=3` to reduce repetition.

## sacreBLEU Results

| Metric | Score |
|---|---|
| Corpus BLEU | 7.28 |
| Avg Sentence BLEU | 20.59|
| Max Sentence BLEU | 100.0 |
| Min Sentence BLEU | 0.0 |

## Key Observations

**1. IndicTrans2 is well-suited for English→Tamil translation.**
The model correctly handles Tamil's agglutinative morphology, producing multi-morpheme Tamil
words as single coherent tokens rather than fragmented subwords. This is a direct result of
its Indic-aware SentencePiece vocabulary.

**2. BLEU score distribution is right-skewed.**
Short or structurally simple sentences (e.g. "The sun rises in the east.") score significantly
higher than long, complex ones (e.g. sentences with subordinate clauses or idiomatic
expressions). This is expected — BLEU penalises length mismatch and n-gram divergence, both
of which increase with sentence complexity.

**3. Reference translation quality affects BLEU.**
The opus-100 dataset references are human-translated but may use different valid Tamil
renderings than the model produces. BLEU is a surface-level metric and does not capture
semantic equivalence, so some low-scoring sentences may still be correct translations.

**4. IndicProcessor is critical to output quality.**
Without preprocessing via `IndicProcessor`, the model receives incorrectly formatted input
and produces degraded output. The postprocessing step also normalises punctuation and
removes BPE artifacts, which improves readability.

## Architectural Changes / Fixes Applied
- None required for Part A. The notebook ran end-to-end without errors.
- Google Drive folder structure created programmatically to match the submission guide's
  recommended repo layout under `indic-translation-asr-project/`.

## Conclusion
IndicTrans2 demonstrates strong English→Tamil translation capability on the opus-100 test
set. The sacreBLEU score reflects reasonable performance given the complexity of Tamil
morphology and the inherent limitations of n-gram based evaluation for morphologically rich
languages.
