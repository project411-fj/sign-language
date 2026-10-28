
import json
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

DATASET_DIR = Path("rewrite_Phase3_train") / "dataset"

BATCH_SIZE = 1
FRAME_STEP = 2
PAD_ID = 0


class SignLanguageDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        keypoints = np.load(item["npy_path"])
        keypoints = keypoints[::FRAME_STEP].copy()

        token_ids = item["token_ids"]

        keypoints = torch.from_numpy(keypoints).float()
        tokens = torch.tensor(token_ids, dtype=torch.long)

        return keypoints, tokens


def collate_fn(batch):
    keypoints, tokens = zip(*batch)

    src_lengths = torch.tensor([
        len(kp) for kp in keypoints
    ])

    src = pad_sequence(
        keypoints,
        batch_first=True,
        padding_value=0
    )

    tgt = pad_sequence(
        tokens,
        batch_first=True,
        padding_value=PAD_ID
    )

    src_key_padding_mask = (
        torch.arange(src.size(1))[None, :]
        >= src_lengths[:, None]
    )

    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]

    tgt_key_padding_mask = tgt_in == PAD_ID

    tgt_len = tgt_in.size(1)

    causal_mask = torch.triu(
        torch.ones(
            (tgt_len, tgt_len),
            dtype=torch.bool
        ),
        diagonal=1
    )

    return (
        src,
        src_key_padding_mask,
        tgt_in,
        tgt_out,
        tgt_key_padding_mask,
        causal_mask
    )


def main():
    train_dataset = SignLanguageDataset(
        DATASET_DIR / "train.json"
    )

    val_dataset = SignLanguageDataset(
        DATASET_DIR / "val.json"
    )

    test_dataset = SignLanguageDataset(
        DATASET_DIR / "test.json"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    print("Training:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))

    batch = next(iter(train_loader))

    names = [
        "src",
        "src_key_padding_mask",
        "tgt_in",
        "tgt_out",
        "tgt_key_padding_mask",
        "causal_mask"
    ]

    for name, tensor in zip(names, batch):
        print(name, tensor.shape)


if __name__ == "__main__":
    main()
