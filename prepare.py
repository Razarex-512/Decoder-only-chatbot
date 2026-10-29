"""Fixed splits, training-only character vocabulary, and independent padded records."""
import hashlib
import json
from pathlib import Path
import re

import torch

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data" / "conversations.jsonl"
SPECIAL = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<user>": 3, "<assistant>": 4, "<eos>": 5}
FORMAT_VERSION = 3


def normalize_question(text):
    """Canonical question case, whitespace and terminal punctuation, shared by chat."""
    return " ".join(text.lower().split()).rstrip(" .?!") + "?"


class Tokenizer:
    def __init__(self, characters):
        if characters != sorted(set(characters)) or any(len(c) != 1 for c in characters):
            raise ValueError("Characters must be sorted, unique, single characters.")
        self.characters = characters
        self.to_id = {c: i + len(SPECIAL) for i, c in enumerate(characters)}
        self.labels = list(SPECIAL) + characters

    @classmethod
    def from_records(cls, records):
        return cls(sorted(set("".join(normalize_question(r["question"]) + r["answer"] for r in records
                                      if r["split"] == "train"))))

    def encode(self, text):
        return [self.to_id.get(c, SPECIAL["<unk>"]) for c in text]

    def decode(self, ids):
        # Structural markers are skipped; unknown characters are visible replacements.
        return "".join(self.labels[i] if i >= len(SPECIAL) else "\ufffd" if i == 1 else "" for i in ids)

    def record(self, question, answer):
        return [2, 3] + self.encode(normalize_question(question)) + [4] + self.encode(answer) + [5]

    def metadata(self):
        return {"characters": self.characters, "special_ids": SPECIAL, "question_normalization": "lower-space-terminal-question-v2"}

    @classmethod
    def from_metadata(cls, metadata):
        if metadata.get("special_ids") != SPECIAL:
            raise ValueError("Incompatible special token IDs; run prepare.py and train.py again.")
        if metadata.get("question_normalization") != "lower-space-terminal-question-v2":
            raise ValueError("Incompatible question normalization; prepare and retrain together.")
        return cls(metadata["characters"])


def validate_records(records):
    families, pairs = {}, {}
    for index, row in enumerate(records, 1):
        if row.get("split") not in ("train", "validation", "test"):
            raise ValueError(f"Record {index}: invalid split.")
        if any(not isinstance(row.get(k), str) or not row[k].strip()
               for k in ("family", "question", "answer")):
            raise ValueError(f"Record {index}: family, question and answer must be nonempty strings.")
        split, family = row["split"], row["family"]
        if family in families and families[family] != split:
            raise ValueError(f"Family {family!r} spans splits.")
        families[family] = split
        normalized = tuple(re.sub(r"[\W_]+", " ", row[k].casefold()).strip()
                           for k in ("question", "answer"))
        if normalized in pairs:
            raise ValueError(f"Duplicate normalized pair in records {pairs[normalized]} and {index}.")
        pairs[normalized] = index
    if set(families.values()) != {"train", "validation", "test"}:
        raise ValueError("All three fixed splits must be nonempty.")


def read_records(path=DATA):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_records(records)
    return records


def batchify(examples, device="cpu"):
    length = max(len(row) - 1 for row in examples)
    x = torch.zeros(len(examples), length, dtype=torch.long, device=device)
    y = torch.zeros_like(x)
    for i, row in enumerate(examples):
        x[i, :len(row) - 1] = torch.tensor(row[:-1], device=device)
        y[i, :len(row) - 1] = torch.tensor(row[1:], device=device)
    return x, y


def answer_targets(inputs, targets):
    """Keep answer characters and EOS; never count the ASSISTANT marker itself.

    The input ASSISTANT position predicts the first answer character. All later
    nonpadding targets belong to the answer, including its final EOS token.
    """
    in_answer = inputs.eq(SPECIAL["<assistant>"]).cumsum(dim=1).gt(0)
    return targets.masked_fill(~in_answer, SPECIAL["<pad>"])


def prepare(records=None, output=ARTIFACTS):
    records = read_records() if records is None else records
    validate_records(records)
    tokenizer = Tokenizer.from_records(records)
    splits = {name: [] for name in ("train", "validation", "test")}
    for i, row in enumerate(records, 1):
        encoded = tokenizer.record(row["question"], row["answer"])
        if len(encoded) > 257:
            raise ValueError(f"Record {i} ({row['family']}): {len(encoded)} tokens; shorten question/answer to <=257.")
        splits[row["split"]].append(encoded)
    canonical = json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")
    payload = {"format_version": FORMAT_VERSION, "tokenizer": tokenizer.metadata(),
               "data_hash": hashlib.sha256(canonical).hexdigest(), "splits": splits, "records": records}
    output.mkdir(parents=True, exist_ok=True)
    (output / "prepared.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "tokenizer.json").write_text(json.dumps(tokenizer.metadata(), indent=2), encoding="utf-8")
    return payload


def load_prepared():
    path = ARTIFACTS / "prepared.json"
    if not path.exists():
        raise ValueError("Prepared data missing. Run python prepare.py first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format_version") != FORMAT_VERSION:
        raise ValueError("Prepared format mismatch. Run python prepare.py again.")
    canonical = json.dumps(read_records(), sort_keys=True, ensure_ascii=False).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != data["data_hash"]:
        raise ValueError("Corpus changed. Run python prepare.py and retrain before evaluation.")
    return data


if __name__ == "__main__":
    try:
        data = prepare()
        print(f"Vocabulary: {len(data['tokenizer']['characters']) + len(SPECIAL)} tokens")
        for name, rows in data["splits"].items():
            print(f"{name}: {len(rows)} examples; longest serialized record: {max(map(len, rows))} tokens")
    except (ValueError, OSError) as error:
        raise SystemExit(str(error))
