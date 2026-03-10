# Concept Library Spec

Use this file as the source of truth for concept definitions, standards, and deviations.

## 1) Scope
- Project: `jigsaw-toxicity-compexp` concept library
- Owner: Matthew Kimotsuki
- Last updated: 2026-03-07
- Version: `concept-spec-v0.1`
- Goal: Build deterministic, dataset-agnostic binary concept annotations for text examples.

## 2) Global Design Rules
- Determinism: fixed ordering, no randomness in concept outputs.
- Binary outputs: all concept values must be in `{0,1}` with `uint8`.
- Alignment: `values[i, :]` must map to `text_ids[i]`.
- Reproducibility: all tunable params recorded in `meta`.

## 3) Preprocessing Contract
- Tokenizer: regex `r"[A-Za-z]+(?:'[A-Za-z]+)?"` from `concepts/utils.py`.
- Lowercasing policy: all tokens lowercased.
- Punctuation handling: ignored by tokenizer except apostrophes inside matched tokens.
- Number handling: Tier 2 `has_number` uses regex `r"\b\d+(?:\.\d+)?\b"` on raw text.
- URL/email handling: Tier 2 uses dedicated raw-text regex checks.
- Unicode handling: non-ASCII symbols are mostly excluded by tokenizer; raw-text regex checks still operate on original text.

## 4) Tier Definitions
### Tier 1 (word concepts)
- Intended purpose:
- Vocabulary construction method:
- Fitting split (e.g., train only):
- Parameters (`k`, `min_df`, `max_df`, stopwords):
- Standard source/method used:
- Deviations from standard:

### Tier 2 (rule primitives)
- Intended purpose: Provide transparent, portable lexical/surface-form concept features without model dependencies.
- Standard resources used (regex standards, lexicons, NLP refs):
  - Regex-style lexical matching and token-set membership (rule-based NLP baseline practice).
  - No external lexicon package yet; current lexicons are small, manually defined starter sets.
- Concept list:
  - `has_first_person`: any token in `{i, me, my, mine, myself, we, us, our, ours, ourselves}`.
  - `has_second_person`: any token in `{you, your, yours, yourself, yourselves, u, ur}`.
  - `has_negation`: token in `{not, no, never, none, cannot}` or token suffix `n't`.
  - `has_modal`: any token in `{can, could, should, would, may, might, must, will, shall}`.
  - `is_question`: raw text contains `"?"`.
  - `is_exclamation`: raw text contains `"!"`.
  - `has_url`: raw text regex match `(?:https?://|www\.)\S+`.
  - `has_email`: raw text regex match `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b`.
  - `has_number`: raw text regex match `\b\d+(?:\.\d+)?\b`.
  - `has_all_caps_word`: raw text regex match `\b[A-Z]{3,}\b`.
  - `long_sentence`: token count `>= 25`.
  - `has_negative_word`: any token in `{idiot, stupid, dumb, hate, awful, terrible, trash, loser}`.
  - `has_positive_word`: any token in `{good, great, love, nice, excellent, amazing, thanks, happy}`.
  - `has_violence_word`: any token in `{kill, hurt, attack, fight, punch, shoot, stab, destroy}`.
  - `has_money_word`: token in `{money, cash, dollar, dollars, bucks, price, pay, paid}` or raw text contains `"$"`.
- Known failure modes:
  - `has_modal` may overfire in non-modal contexts (e.g., ambiguous token `can`).
  - `has_all_caps_word` may fire on formatting artifacts, acronyms, or redirects.
  - `has_number` may fire on IDs or formatting noise unrelated to semantics.
  - Lexicon concepts can miss paraphrases/slang and can overfire in quoted/meta text.
- Deviations from standards:
  - Uses compact in-code starter lexicons instead of established external lexicon resources.
  - Uses shallow surface rules instead of parser/model-based detection.

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

## 8) Open Questions
- 
