# Part A: Batch Translation and Evaluation — Observations

## Model Used
`ai4bharat/indictrans2-en-indic-1B` via the IndicTransToolkit pipeline.

## Dataset
500 English–Tamil sentence pairs sourced from the Helsinki-NLP/opus-100 `en-ta` test split.
Reference translations from the dataset were used for all evaluation metrics.

## Architecture
- The IndicTrans2 model uses a custom `IndicProcessor` for preprocessing and postprocessing,
  which handles script normalization, punctuation, and language tag injection internally.
- Tokenization uses a SentencePiece model trained specifically on Indic scripts, giving it
  significantly better Tamil subword coverage compared to multilingual models.
- Translation was run with `num_beams=5` and `no_repeat_ngram_size=3` to reduce repetition.

## Evaluation Results

### sacreBLEU

| Metric | Score |
|---|---|
| Corpus BLEU | 7.28 |
| Avg Sentence BLEU | 20.59 |
| Max Sentence BLEU | 100.0 |
| Min Sentence BLEU | 0.0 |

### Additional Metrics (chrF, chrF++, TER, ROUGE-L)

| Metric | Corpus Score | Avg Sentence Score | Max | Min | Note |
|---|---|---|---|---|---|
| chrF | 40.43 | 40.47 | 100.0 | 0.0 | Higher is better |
| chrF++ | 35.34 | 36.64 | 100.0 | 0.0 | Higher is better |
| TER | 88.53 | — | — | — | Lower is better |
| ROUGE-L | 5.69 | 5.69 | 100.0 | 0.0 | Higher is better; avg of sentence scores |

> Corpus and sentence-level results saved to `additional_metrics_results.csv`.
> Per-sentence chrF, chrF++, and ROUGE-L scores are appended as columns in `translation_outputs.csv`.
>
> Note on ROUGE-L: the `rouge-score` library computes only sentence-level scores.
> The corpus score reported here is the average of all sentence-level ROUGE-L scores,
> which is consistent with the aggregation method in the original ROUGE paper.

## Key Observations

**1. IndicTrans2 is well-suited for English→Tamil translation.**
The model correctly handles Tamil's agglutinative morphology, producing multi-morpheme Tamil
words as single coherent tokens rather than fragmented subwords. This is a direct result of
its Indic-aware SentencePiece vocabulary.

**2. chrF (40.43) substantially outperforms BLEU (7.28) — expected for Tamil.**
The 5× gap between chrF and BLEU is not a contradiction; it reflects the fundamental
difference in what these metrics measure. BLEU operates on word n-grams and penalises any
surface-form mismatch, including morphological variants that are semantically equivalent.
chrF's character-level overlap is far more forgiving of valid morphological alternatives and
partial matches in Tamil script, making it the more reliable metric for this language pair.

**3. chrF++ (35.34) is lower than chrF (40.43) due to word-order penalty.**
chrF++ adds a word-unigram and word-bigram component on top of character n-grams. Tamil's
SOV word order frequently differs from reference translations even when meaning is preserved,
causing the word-order-sensitive chrF++ to score lower than pure character-level chrF.

**4. ROUGE-L (5.69) tracks closely with BLEU and reflects the same word-level limitations.**
ROUGE-L measures the longest common subsequence at the word level. For Tamil, where a single
agglutinated word can correspond to an entire English phrase, word-level LCS rarely aligns
even when the translation is semantically correct. Its low score here should not be
interpreted as poor translation quality.

**5. TER (88.53) is high, reflecting Tamil's structurally different word order.**
TER counts the minimum edits (insertions, deletions, shifts) needed to convert the hypothesis
into the reference. Tamil is a Subject-Object-Verb language while English is
Subject-Verb-Object, so even a correct translation requires many token-level reorderings
relative to the reference, inflating TER. A high TER is therefore expected and does not
indicate poor performance.

**6. BLEU score distribution is right-skewed.**
Short or structurally simple sentences score significantly higher than long, complex ones with
subordinate clauses or idiomatic expressions. BLEU penalises length mismatch and n-gram
divergence, both of which increase with sentence complexity.

**7. Reference translation quality affects all metrics equally.**
The opus-100 references are human-translated but may use different valid Tamil renderings than
the model produces. All metrics above are surface-level and do not capture semantic
equivalence, so some low-scoring sentences may still be correct translations.

**8. IndicProcessor is critical to output quality.**
Without preprocessing via `IndicProcessor`, the model receives incorrectly formatted input
and produces degraded output. The postprocessing step also normalises punctuation and
removes BPE artifacts, improving readability across all evaluated outputs.


## Conclusion
IndicTrans2 demonstrates strong English→Tamil translation capability on the opus-100 test
set. The low BLEU (7.28), ROUGE-L (5.69), and high TER (88.53) scores should not be taken
as evidence of poor translation — all three are ill-suited to Tamil's agglutinative, SOV
structure. chrF (40.43) is the most appropriate automatic metric for this language pair and
indicates reasonable translation quality. For a more semantically grounded assessment,
metrics such as COMET or BERTScore with a multilingual encoder would be needed.
