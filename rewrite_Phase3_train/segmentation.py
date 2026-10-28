
import json
import random
import jieba

from pathlib import Path
from collections import Counter

INPUT_PATH = Path("rewrite_Phase3_train") / "dataset" / "manifest.json"
OUTPUT_DIR = Path("rewrite_Phase3_train") / "dataset"

SEED = 42
MIN_FREQ = 1


def get_group(item):
    parts = item["video_id"].split("_")
    return "_".join(parts[:2])


def tokenize(text):
    return [
        token
        for token in jieba.lcut(text.strip())
        if not token.isspace()
    ]


def build_vocab(rows):
    counts = Counter(
        token
        for row in rows
        for token in tokenize(row["text"])
    )

    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3
    }

    for word, count in counts.most_common():
        if count >= MIN_FREQ:
            vocab[word] = len(vocab)

    return vocab


def add_token_ids(rows, vocab):
    for row in rows:
        tokens = tokenize(row["text"])

        row["token_ids"] = (
            [vocab["<bos>"]]
            + [vocab.get(token, vocab["<unk>"]) for token in tokens]
            + [vocab["<eos>"]]
        )

    return rows


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    groups = sorted(set(get_group(row) for row in manifest))

    random.Random(SEED).shuffle(groups)

    n = len(groups)
    n_test = max(1, round(n * 0.1))
    n_val = max(1, round(n * 0.1))

    test_groups = set(groups[:n_test])
    val_groups = set(groups[n_test:n_test + n_val])

    train = []
    val = []
    test = []

    for row in manifest:
        group = get_group(row)

        if group in test_groups:
            test.append(row)
        elif group in val_groups:
            val.append(row)
        else:
            train.append(row)

    vocab = build_vocab(train)

    train = add_token_ids(train, vocab)
    val = add_token_ids(val, vocab)
    test = add_token_ids(test, vocab)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_json(OUTPUT_DIR / "train.json", train)
    save_json(OUTPUT_DIR / "val.json", val)
    save_json(OUTPUT_DIR / "test.json", test)
    save_json(OUTPUT_DIR / "vocab.json", vocab)

    print("Training:", len(train))
    print("Validation:", len(val))
    print("Test:", len(test))
    print("Vocabulary Size:", len(vocab))


if __name__ == "__main__":
    main()
