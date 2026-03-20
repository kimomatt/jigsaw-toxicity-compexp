# Concept Library Spec

Use this file as the source of truth for concept definitions, standards, and deviations.

## 1) Scope
- Project: `jigsaw-toxicity-compexp` concept library
- Owner: Matthew Kimotsuki
- Last updated: 2026-03-12
- Version: `concept-spec-v0.1`
- Goal: Build deterministic, dataset-agnostic binary concept annotations for text examples.

## 2) Global Design Rules
- Determinism: fixed ordering, no randomness in concept outputs.
- Binary outputs: all concept values must be in `{0,1}` with `uint8`.
- Alignment: `values[i, :]` must map to `text_ids[i]`.
- Reproducibility: all tunable params recorded in `meta`.

## 3) Preprocessing Contract
- Tier 1 tokenizer backend: `spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])` from `concepts/tier1_words.py`.
- Tier 1 tokenization call path: `nlp.make_doc(text)` is used so only the tokenizer runs; no parser / tagger / NER / lemmatizer pass is applied during Tier 1 extraction.
- Tier 1 lowercasing policy: use spaCy `token.lower_`.
- Tier 1 whitespace handling: whitespace tokens are dropped via `token.is_space`.
- Tier 1 punctuation handling: tokenization is delegated to spaCy; vocabulary filtering excludes the explicit minimal skip list and drops pure-punctuation tokens from Tier 1 vocabulary candidates.
- Tier 2 annotation backend: `spacy.load("en_core_web_sm")` from `concepts/tier2_primitives.py`.
- Tier 2 batching path: `nlp.pipe(texts, batch_size=128)` is used so docs are parsed once per batch and reused across all Tier 2 concepts.
- Tier 2 token attributes: concept rules may use token text, lowercase form, dependency labels, POS/tag information, sentence boundaries, and named entities from spaCy annotations.
- Unicode handling: non-ASCII symbols are mostly excluded by tokenizer; raw-text regex checks still operate on original text.

## 4) Tier Definitions
### Tier 1 (word concepts)
- Intended purpose: Provide transparent lexical presence concepts using a dataset-fitted top-k token vocabulary.
- Vocabulary construction method: tokenize raw text with spaCy English tokenization, compute document frequency on train texts only, remove a minimal explicit skip list, drop pure-punctuation tokens, and keep the top-k tokens after `min_doc_freq` / `max_doc_frac` filtering.
- Fitting split (e.g., train only): train split only.
- Parameters (`k`, `min_df`, `max_df`, stopwords): `top_k`, `min_doc_freq`, `max_doc_frac`, a minimal exclusion list currently `{a, an, of, the, ., ,}`, and a pure-punctuation token filter.
- Standard source/method used:
  - spaCy English tokenization via `en_core_web_sm`
  - frequency-based lexical feature selection in the style of Compexp sentence-level token features
- Deviations from standard:
  - current implementation uses raw spaCy tokens rather than a separate model-vocabulary object
  - current implementation uses document frequency pruning (`min_doc_freq`, `max_doc_frac`) because the emitted concepts are binary word-presence indicators
  - pure punctuation tokens are excluded from Tier 1 because punctuation-style signals are handled more naturally as Tier 2 surface-form concepts
  - special bookkeeping tokens such as `UNK` / `PAD` are not part of the raw-text Tier 1 token stream

### Tier 2 (explicit linguistic concepts)
- Intended purpose: Provide abstract, human-readable linguistic concepts derived from parser/NER/morphosyntactic annotations rather than shallow task-specific word lists.
- Standard resources used:
  - spaCy `en_core_web_sm` for tokenization, POS/tag information, sentence segmentation, dependency parsing, and named entity recognition.
  - Lightweight explicit rules layered over those annotations.
- Active concept list:
  - `has_first_person_reference`: any first-person reference token.
  - `has_second_person_reference`: any second-person reference token.
  - `has_third_person_reference`: any third-person reference token.
  - `has_exclamation_clause`: raw text contains `"!"`.
  - `has_quoted_span`: raw text contains quote characters.
  - `has_negated_predicate`: dependency-marked negation.
  - `has_modalized_predicate`: modal auxiliary or modalized predicate structure.
  - `has_direct_address`: vocative structure or likely direct address pattern.
- Disabled-but-retained concepts:
  - `has_person_entity`
  - `has_org_entity`
  - `has_gpe_entity`
  - `has_question_clause`
  - `has_future_construction`
  - `has_imperative_clause`
  - `has_coordination`
  - `has_copular_predication`
  - `has_adjectival_predication`
- Known failure modes:
  - `has_direct_address` is an approximate structural heuristic, not full discourse analysis.
  - `has_modalized_predicate` is broad by design and will fire on many non-toxic modal contexts.
  - Reference concepts are still lexicon-backed rather than purely structural.
- Deviations from standards:
  - Uses a compact hand-selected abstract taxonomy instead of attempting exhaustive linguistic coverage.
  - The active Tier 2 inventory was pruned after two random Jigsaw audits to prefer robustness on noisy web text over broader linguistic coverage.

### Tier 2 Baseline (archived rule primitives)
- Archived implementation file: `concepts/tier2_baseline_primitives.py`
- Status: retained for comparability and fallback only; no longer the active Tier 2.

### Tier 3 (cluster concepts)
- Intended purpose:
- Upstream dependency (embeddings + clustering):
- Assignment format contract:
- Cluster naming and ordering:
- Known caveats:

## 5) Standards vs Deviations Log
For each deviation from a standard method/resource, document:
- ID:
- Date:
- Tier:
- Standard baseline:
- What was changed:
- Why:
- Expected tradeoff:
- Evidence:

## 6) Evaluation Protocol
- Sanity checks:
  - shape and id alignment
  - binary matrix validation
  - deterministic repeatability
- Coverage checks:
  - dead concepts (near 0)
  - always-on concepts (near 1)
- Manual audit:
  - sample size
  - audit rubric
- Cross-dataset checks:
  - datasets used
  - acceptance criteria

## 7) Versioning
- Version scheme (e.g., semver): semantic versioning (`major.minor.patch`) for concept definitions.
- What triggers a version bump:
  - Patch: bugfix/no intended behavior change
  - Minor: concept additions/refinements
  - Major: breaking definition changes
- Changelog:
  - version: `0.1.0`
  - date: 2026-03-07
  - summary: Initial implementation of Tier 1, Tier 2, Tier 3 scaffolding and Jigsaw demo integration.
  - impact metrics: pending cross-dataset benchmark table.
  - version: `0.1.1`
  - date: 2026-03-12
  - summary: Tier 1 updated to use spaCy `en_core_web_sm` tokenization and top-k token vocabulary without a seeded base lexicon.
  - impact metrics: pending rerun of Tier 1 demo and integration checks.
  - version: `0.1.2`
  - date: 2026-03-12
  - summary: Tier 1 optimized to cache per-text token sets and exclude pure punctuation tokens from the candidate vocabulary.
  - impact metrics: `10,000`-row Tier 1 run completed in `2.14s` with shape `(10000, 300)`.
  - version: `0.2.0`
  - date: 2026-03-12
  - summary: Active Tier 2 replaced with a spaCy-based explicit linguistic concept layer; prior shallow rule primitives archived as a baseline.
  - impact metrics: initial sanity check passed with `17` Tier 2 concepts and correct `uint8` output.
  - version: `0.2.1`
  - date: 2026-03-12
  - summary: Active Tier 2 reduced to an 8-concept robust subset after repeated random-audit pruning; noisier linguistic concepts kept disabled for possible later reintroduction.
  - impact metrics: two 50-example random audits on Jigsaw web text looked substantially cleaner than the initial 17-concept Tier 2.

## 8) Open Questions
- 
