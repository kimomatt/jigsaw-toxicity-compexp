"""Phase 1 and Phase 2 concept annotation demo."""

from __future__ import annotations

import csv
import time
from typing import Dict, List, Optional, Sequence, Tuple
from pathlib import Path

import numpy as np

from concepts.build import build_concept_set
from concepts.tier1_words import build_tier1_vocabulary
from concepts.utils import coverage_stats, pretty_print_single, validate_binary_matrix

# ds_split is a list of dictionaries, where each dictionary represents a single data example with keys like "id" and "text". The function dataset_to_examples takes this list of dictionaries and extracts the "id" and "text" values into two separate lists, which are then returned as a tuple. This allows us to convert from a more general dataset format (list of dicts) into the specific aligned format (ids and texts) that our concept building functions expect.
# example ds_split input: [{"id": "s0", "text": "I love this product."}, {"id": "s1", "text": "You are awful and I hate this."}, {"id": "s2", "text": "Email me at example@email.com"}]
def dataset_to_examples(ds_split: Sequence[Dict[str, str]]) -> Tuple[List[str], List[str]]:
    """Phase 3 stub adapter: convert list-of-dicts into aligned ids and texts."""
    # converts general dataset records into exact format build_concept_set expects, which is a list of texts and an optional list of text ids.
    ids: List[str] = []
    texts: List[str] = []
    for i, item in enumerate(ds_split):
        if "id" not in item or "text" not in item:
            raise ValueError(f"Expected keys 'id' and 'text' at index {i}")
        ids.append(str(item["id"]))
        texts.append(str(item["text"]))
    return ids, texts


def load_jigsaw_examples_from_csv(
    dataset_dir: Path, limit: Optional[int] = 200
) -> Tuple[List[str], List[str]]:
    """Load Jigsaw Kaggle train.csv into aligned ids/texts lists."""
    csv_path = dataset_dir / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find: {csv_path}")

    ids: List[str] = []
    texts: List[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # going thru each row in the csv file
        for row in reader:
            if limit is not None and len(texts) >= limit:
                break
            # we expect the csv to have a column named "comment_text" which contains the text of the comment. if the "comment_text" column is missing, text defaults to ""
            text = row.get("comment_text", "")
            if text is None:
                continue
            text = str(text).strip()
            if not text:
                continue
            row_id = row.get("id")
            if row_id is None or str(row_id).strip() == "":
                # generating a fallback id if the "id" column is missing or empty, using the current number of texts loaded to create a unique id like "row_0", "row_1", etc. This ensures that every example has an id, even if the original CSV doesn't provide one.
                row_id = f"row_{len(texts)}"
            ids.append(str(row_id))
            texts.append(text)

    if not texts:
        raise ValueError(f"No usable rows found in: {csv_path}")
    return ids, texts


def print_random_audit(
    texts: Sequence[str],
    ids: Sequence[str],
    values: np.ndarray,
    names: Sequence[str],
    n: int = 10,
    seed: int = 42,
) -> None:
    """Print a small random manual-audit sample with fired concepts only."""
    # print a random sample of rows and which concepts fired 
    if len(texts) == 0:
        print("No examples available for audit.")
        return
    if len(texts) != len(ids) or values.shape[0] != len(texts):
        raise ValueError("Audit inputs must be row-aligned (texts, ids, values).")

    count = min(n, len(texts))
    rng = np.random.default_rng(seed)

    # sample_idx is a 1D array of random indices into the texts/ids/values arrays, with length equal to the smaller of n and the total number of texts. This allows us to randomly select a subset of examples for manual auditing, without replacement (i.e., we won't select the same example multiple times).
    # size is how many random indices we want to generate, which is determined by the count variable. replace=False means that once an index is selected, it cannot be selected again, ensuring that we get unique examples in our audit sample, len(texts) is the total number of examples available, so we won't try to sample more examples than we have.
    sample_idx = rng.choice(len(texts), size=count, replace=False)

    print(f"\nManual random audit ({count} examples, seed={seed}):")
    # indexing starts at 1
    for i, row_idx in enumerate(sample_idx, start=1):
        # values.shape[1] → number of columns (features)
        # values[row_idx, j] → value of feature j for that row
        # names[j] → name of feature j
        fired = [names[j] for j in range(values.shape[1]) if int(values[row_idx, j]) == 1]
        text_preview = texts[row_idx].replace("\n", " ").strip()
        if len(text_preview) > 220:
            text_preview = text_preview[:220] + "..."

        print(f"\n[{i}] id={ids[row_idx]}")
        print(f"text: {text_preview}")
        # if the fired list is empty, we print "(none)" to indicate that no concepts fired for that example. Otherwise, we join the names of the fired concepts with commas and print them.
        print("fired:", ", ".join(fired) if fired else "(none)")


def run_phase_1() -> None:
    """Phase 1: single-sentence sanity checks across all tiers."""
    sentence = "YOU are an idiot!!! Email me at foo@bar.com? I will not do that."
    word_list = ["idiot", "stupid"]

    tier1 = build_concept_set(
        texts=[sentence],
        tier=1,
        tier1_words=word_list,
        text_ids=["ex0"],
        meta={"demo_phase": 1},
    )
    tier2 = build_concept_set(
        texts=[sentence],
        tier=2,
        text_ids=["ex0"],
        meta={"demo_phase": 1},
    )
    # for tier 3, we create a single cluster assignment of 0 for our one example, which should result in a single concept "cluster::0" that fires for that example.
    tier3 = build_concept_set(
        texts=[sentence],
        tier=3,
        tier3_assignments=np.array([0], dtype=np.int64),
        text_ids=["ex0"],
        meta={"demo_phase": 1},
    )

    print("=== Phase 1: Single Sentence ===")
    print("\nTier 1:")
    pretty_print_single(sentence, tier1.concept_names, tier1.values)
    print("Tier 1 matrix shape:", tier1.values.shape)

    print("\nTier 2:")
    pretty_print_single(sentence, tier2.concept_names, tier2.values)
    print("Tier 2 matrix shape:", tier2.values.shape)

    print("\nTier 3:")
    pretty_print_single(sentence, tier3.concept_names, tier3.values)
    print("Tier 3 matrix shape:", tier3.values.shape)


def run_phase_2() -> None:
    """Phase 2: batch mode with ids and coverage reporting."""
    texts = [
        "You are awesome and I love this!",
        "This is terrible, I hate it.",
        "Can you send $20 to me?",
        "Visit https://example.com for details.",
        "Email me at hello@sample.org now.",
        "WE WILL WIN THIS FIGHT!!!",
        "I will not do that.",
        "u should never say that to your team.",
        "The price is 19.99 dollars today.",
        "What is your plan for tomorrow?",
    ]
    ids = [f"ex{i}" for i in range(len(texts))]
    tier1_words = ["idiot", "stupid", "love", "hate", "money"]

    tier2_batch = build_concept_set(
        texts=texts,
        tier=2,
        text_ids=ids,
        meta={"demo_phase": 2, "note": "batch tier2"},
    )
    tier1_batch = build_concept_set(
        texts=texts,
        tier=1,
        tier1_words=tier1_words,
        text_ids=ids,
        meta={"demo_phase": 2, "note": "batch tier1"},
    )

    print("\n=== Phase 2: Batch + IDs ===")
    print("Tier 2 matrix shape:", tier2_batch.values.shape)
    print("Tier 2 text_ids aligned:", tier2_batch.text_ids == ids)
    validate_binary_matrix(tier2_batch.values)
    print("Tier 2 binary matrix: valid")

    top10 = coverage_stats(tier2_batch.values, tier2_batch.concept_names)[:10]
    print("\nTier 2 top 10 firing rates:")
    for name, rate in top10:
        print(f"{name:20s} {rate:.3f}")

    print("\nTier 1 batch matrix shape:", tier1_batch.values.shape)

    # double checking that text_ids we passed in are correctly aligned in the ConceptSet output (no reason they wouldn't be, but just to be sure our build_concept_set function is handling the text_ids correctly and preserving the alignment between the input ids and the output ConceptSet's text_ids).
    print("Tier 1 text_ids aligned:", tier1_batch.text_ids == ids)


def run_phase_3_stub() -> None:
    """Phase 3: show dataset adapter pattern without dataset library imports."""
    ds_split = [
        {"id": "s0", "text": "I love this product."},
        {"id": "s1", "text": "You are awful and I hate this."},
        {"id": "s2", "text": "Email me at test@example.com"},
    ]
    ids, texts = dataset_to_examples(ds_split)
    conceptset = build_concept_set(
        texts=texts,
        tier=2,
        text_ids=ids,
        meta={"demo_phase": 3, "adapter": "stub"},
    )
    print("\n=== Phase 3: Dataset Adapter Stub ===")
    print("Adapter produced:", len(ids), "examples")
    print("First id/text:", ids[0], "|", texts[0])
    print("Tier 2 matrix shape from adapter:", conceptset.values.shape)
    print("IDs aligned:", conceptset.text_ids == ids)


def run_phase_4_real_jigsaw() -> None:
    """Phase 4: integrate real Jigsaw CSV if present locally."""
    dataset_dir = Path("jigsaw-toxic-comment-classification-challenge")
    print("\n=== Phase 4: Real Jigsaw CSV (Local) ===")
    try:
        ids, texts = load_jigsaw_examples_from_csv(dataset_dir=dataset_dir, limit=200)
    except (FileNotFoundError, ValueError) as exc:
        print("Skipping Phase 4:", exc)
        return

    conceptset = build_concept_set(
        texts=texts,
        tier=2,
        text_ids=ids,
        meta={"demo_phase": 4, "dataset": "jigsaw_train_csv", "limit": len(texts)},
    )
    print("Loaded examples:", len(texts))
    print("Tier 2 matrix shape:", conceptset.values.shape)
    print("IDs aligned:", conceptset.text_ids == ids)
    validate_binary_matrix(conceptset.values)
    print("Binary matrix: valid")

    top10 = coverage_stats(conceptset.values, conceptset.concept_names)[:10]
    print("Top 10 firing rates:")
    for name, rate in top10:
        print(f"{name:20s} {rate:.3f}")
    print_random_audit(
        texts=texts,
        ids=ids,
        values=conceptset.values,
        names=conceptset.concept_names,
        n=10,
        seed=42,
    )


def run_phase_5_ramp() -> None:
    """Phase 5: staged scale-up on real Jigsaw data."""
    dataset_dir = Path("jigsaw-toxic-comment-classification-challenge")
    print("\n=== Phase 5: Ramp Run (1k -> 10k -> full) ===")

    ramps: List[Tuple[str, Optional[int]]] = [
        ("1k", 1_000),
        ("10k", 10_000),
        ("full", None),
    ]
    for label, limit in ramps:
        try:
            ids, texts = load_jigsaw_examples_from_csv(dataset_dir=dataset_dir, limit=limit)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping ramp {label}:", exc)
            return

        # start timer
        t0 = time.perf_counter()
        conceptset = build_concept_set(
            texts=texts,
            tier=2,
            text_ids=ids,
            meta={"demo_phase": 5, "dataset": "jigsaw_train_csv", "ramp": label},
        )
        # get elapsed seconds
        dt = time.perf_counter() - t0

        validate_binary_matrix(conceptset.values)
        # :>4s means right align string in width 4, :6d means integer width 6, :.2f means float with 2 decimal places
        print(
            f"{label:>4s}: rows={len(texts):6d}, shape={conceptset.values.shape}, "
            f"ids_aligned={conceptset.text_ids == ids}, seconds={dt:.2f}"
        )


def run_phase_6_tier1_vocab() -> None:
    """Phase 6: build Tier 1 vocabulary from top-k frequent spaCy tokens."""
    dataset_dir = Path("jigsaw-toxic-comment-classification-challenge")
    print("\n=== Phase 6: Tier 1 Vocabulary Builder ===")
    try:
        fit_ids, fit_texts = load_jigsaw_examples_from_csv(dataset_dir=dataset_dir, limit=10_000)
    except (FileNotFoundError, ValueError) as exc:
        print("Skipping Phase 6:", exc)
        return

    del fit_ids  # not needed for vocab fitting
    max_doc_frac = 0.4
    vocab = build_tier1_vocabulary(
        fit_texts,
        top_k=300,
        min_doc_freq=20,
        max_doc_frac=max_doc_frac,
    )
    print("Built vocabulary size:", len(vocab))
    print("First 20 vocab words:", ", ".join(vocab[:20]))

    ids, texts = load_jigsaw_examples_from_csv(dataset_dir=dataset_dir, limit=10_000)
    t0 = time.perf_counter()
    conceptset = build_concept_set(
        texts=texts,
        tier=1,
        tier1_words=vocab,
        text_ids=ids,
        meta={
            "demo_phase": 6,
            "dataset": "jigsaw_train_csv",
            "tier1_top_k": 300,
            "tier1_min_doc_freq": 20,
            "tier1_max_doc_frac": max_doc_frac,
            "tier1_vocab_size": len(vocab),
            "fit_rows": len(fit_texts),
        },
    )
    dt = time.perf_counter() - t0
    validate_binary_matrix(conceptset.values)
    print(
        f"Tier 1 run: rows={len(texts)}, shape={conceptset.values.shape}, "
        f"ids_aligned={conceptset.text_ids == ids}, seconds={dt:.2f}"
    )

    top10 = coverage_stats(conceptset.values, conceptset.concept_names)[:10]
    print("Tier 1 top 10 firing rates:")
    for name, rate in top10:
        print(f"{name:20s} {rate:.3f}")


def main() -> None:
    run_phase_1()
    run_phase_2()
    run_phase_3_stub()
    run_phase_4_real_jigsaw()
    # run_phase_5_ramp()
    run_phase_6_tier1_vocab()


if __name__ == "__main__":
    main()
