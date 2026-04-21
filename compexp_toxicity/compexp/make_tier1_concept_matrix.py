"""Phase 1 full concept annotation"""

from __future__ import annotations

import csv
import json
import time
from typing import Dict, List, Optional, Sequence, Tuple
from pathlib import Path

import numpy as np

from concepts.build import build_concept_set
from concepts.tier1_words import build_tier1_vocabulary
from concepts.utils import coverage_stats, validate_binary_matrix

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
    dataset_dir: Path, limit: Optional[int] = None
) -> Tuple[List[str], List[str]]:
    """Load Jigsaw data from forward pass metadata output into aligned ids/texts lists."""
    csv_path = dataset_dir / "val_metadata.csv"
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
            # we expect the val metadata csv to have a column named "input" which contains the text of the comment. if the "input" column is missing, text defaults to ""
            text = row.get("input", "")
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


# basically need to do phase 6 stuff, which is building a tier 1 concept set based on a vocabulary of top-k frequent non-stopword tokens from the Jigsaw dataset, and then computing the concept values for each text based on whether it contains each of those top-k words, and then printing out some stats about the resulting concept set.

def main() -> None:

    run_output_dir = Path("compexp_outputs_full")
    output_dir = run_output_dir / "conceptset_tier1"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        ids, texts = load_jigsaw_examples_from_csv(dataset_dir=run_output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print("Unable to load jigsaw examples from csv:", exc)
        return

    max_doc_frac = 0.7
    vocab = build_tier1_vocabulary(
        texts,
        top_k=300,
        min_doc_freq=20,
        max_doc_frac=max_doc_frac,
    )
    print("Built vocabulary size:", len(vocab))
    print("First 20 vocab words:", ", ".join(vocab[:20]))

    t0 = time.perf_counter()
    conceptset = build_concept_set(
        texts=texts,
        tier=1,
        tier1_words=vocab,
        text_ids=ids,
        meta={
            "dataset": "val_metadata_csv",
            "tier1_top_k": 300,
            "tier1_min_doc_freq": 20,
            "tier1_max_doc_frac": max_doc_frac,
            "tier1_vocab_size": len(vocab),
            "fit_rows": len(texts),
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

    # saving in npy format in the output directory
    np.save(output_dir / "conceptset_tier1.npy", conceptset.values)

    # saving concept names
    with open(output_dir / "conceptset_tier1_names.txt", "w", encoding="utf-8") as f:
        for name in conceptset.concept_names:
            f.write(name + "\n")

    # maybe also save tier1 concept metadata
    tier1_concept_metadata = conceptset.meta or {}
    with open(output_dir / "conceptset_tier1_metadata.json", "w") as f:
        json.dump(tier1_concept_metadata, f, indent=2)


if __name__ == "__main__":
    main()
