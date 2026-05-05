import pytest
import csv

from compexp_toxicity.compexp.make_tier1_concept_matrix import dataset_to_examples, load_jigsaw_examples_from_csv


# def dataset_to_examples(ds_split: Sequence[Dict[str, str]]) -> Tuple[List[str], List[str]]:
#     """Phase 3 stub adapter: convert list-of-dicts into aligned ids and texts."""
#     # converts general dataset records into exact format build_concept_set expects, which is a list of texts and an optional list of text ids.
#     ids: List[str] = []
#     texts: List[str] = []
#     for i, item in enumerate(ds_split):
#         if "id" not in item or "text" not in item:
#             raise ValueError(f"Expected keys 'id' and 'text' at index {i}")
#         ids.append(str(item["id"]))
#         texts.append(str(item["text"]))
#     return ids, texts

def test_dataset_to_examples_happy():
    ds_split = [{"id": "s0", "text": "I love this product."}, {"id": "s1", "text": "You are awful and I hate this."}, {"id": "s2", "text": "Email me at example@email.com."}]
    ids, texts = dataset_to_examples(ds_split)
    assert ids == ["s0", "s1", "s2"]
    assert texts == ["I love this product.", "You are awful and I hate this.", "Email me at example@email.com."]

def test_dataset_to_examples_missing_id():
    ds_split = [{"text": "I love this product."}]
    with pytest.raises(ValueError, match="Expected keys 'id' and 'text' at index 0"):
        dataset_to_examples(ds_split)

def test_dataset_to_examples_missing_text():
    ds_split = [{"id": "s0"}]
    with pytest.raises(ValueError, match="Expected keys 'id' and 'text' at index 0"):
        dataset_to_examples(ds_split)

# def load_jigsaw_examples_from_csv(
#     dataset_dir: Path, limit: Optional[int] = None
# ) -> Tuple[List[str], List[str]]:
#     """Load Jigsaw data from forward pass metadata output into aligned ids/texts lists."""
#     csv_path = dataset_dir / "val_metadata.csv"
#     if not csv_path.exists():
#         raise FileNotFoundError(f"Could not find: {csv_path}")

#     ids: List[str] = []
#     texts: List[str] = []
#     with csv_path.open("r", encoding="utf-8", newline="") as f:
#         reader = csv.DictReader(f)
#         # going thru each row in the csv file
#         for row in reader:
#             if limit is not None and len(texts) >= limit:
#                 break
#             # we expect the val metadata csv to have a column named "input" which contains the text of the comment. if the "input" column is missing, text defaults to ""
#             text = row.get("input", "")
#             if text is None:
#                 continue
#             text = str(text).strip()
#             if not text:
#                 continue
#             row_id = row.get("id")
#             if row_id is None or str(row_id).strip() == "":
#                 # generating a fallback id if the "id" column is missing or empty, using the current number of texts loaded to create a unique id like "row_0", "row_1", etc. This ensures that every example has an id, even if the original CSV doesn't provide one.
#                 row_id = f"row_{len(texts)}"
#             ids.append(str(row_id))
#             texts.append(text)

#     if not texts:
#         raise ValueError(f"No usable rows found in: {csv_path}")
#     return ids, texts

def test_load_jigsaw_examples_from_csv_happy(tmp_path):

    # create a temporary CSV file with the expected structure
    csv_path = tmp_path / "val_metadata.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "input"])
        writer.writeheader()
        writer.writerow({"id": "s0", "input": "I love this product."})
        writer.writerow({"id": "s1", "input": "You are awful and I hate this."})
        writer.writerow({"id": "s2", "input": "Email me at example@email.com."})

    ids, texts = load_jigsaw_examples_from_csv(tmp_path)
    assert ids == ["s0", "s1", "s2"]
    assert texts == ["I love this product.", "You are awful and I hate this.", "Email me at example@email.com."]

def test_load_jigsaw_examples_from_csv_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Could not find:"):
        load_jigsaw_examples_from_csv(tmp_path)

def test_load_jigsaw_examples_from_csv_missing_input_column(tmp_path):

    csv_path = tmp_path / "val_metadata.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id"])
        writer.writeheader()
        writer.writerow({"id": "s0"})
        writer.writerow({"id": "s1"})
        writer.writerow({"id": "s2"})

    with pytest.raises(ValueError, match="No usable rows found in:"):
        load_jigsaw_examples_from_csv(tmp_path)
  
def test_load_jigsaw_examples_from_csv_empty_input(tmp_path):

    csv_path = tmp_path / "val_metadata.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "input"])
        writer.writeheader()
        writer.writerow({"id": "s0", "input": ""})
        writer.writerow({"id": "s1", "input": "   "})
        writer.writerow({"id": "s2", "input": None})

    with pytest.raises(ValueError, match="No usable rows found in:"):
        load_jigsaw_examples_from_csv(tmp_path)

def test_load_jigsaw_examples_from_csv_missing_id_column(tmp_path):

    csv_path = tmp_path / "val_metadata.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["input"])
        writer.writeheader()
        writer.writerow({"input": "I love this product."})
        writer.writerow({"input": "You are awful and I hate this."})
        writer.writerow({"input": "Email me at example@email.com."})

    ids, texts = load_jigsaw_examples_from_csv(tmp_path)
    assert ids == ["row_0", "row_1", "row_2"]
    assert texts == ["I love this product.", "You are awful and I hate this.", "Email me at example@email.com."]

def test_load_jigsaw_examples_from_csv_limit(tmp_path):

    csv_path = tmp_path / "val_metadata.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "input"])
        writer.writeheader()
        writer.writerow({"id": "s0", "input": "I love this product."})
        writer.writerow({"id": "s1", "input": "You are awful and I hate this."})
        writer.writerow({"id": "s2", "input": "Email me at example@email.com."})

    ids, texts = load_jigsaw_examples_from_csv(tmp_path, limit=2)
    assert ids == ["s0", "s1"]
    assert texts == ["I love this product.", "You are awful and I hate this."]

