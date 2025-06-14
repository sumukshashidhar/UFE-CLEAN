from typing import List

from datasets import Dataset, load_dataset
from tqdm import tqdm

TARGET_CHARS = 40_000_000_000
BUFFER_SIZE = 100_000
SEED = 42
HF_REPO = "sumuks/Ultra-FineWeb-10B"


def sample_fineweb(
    target_chars: int = TARGET_CHARS,
    buffer_size: int = BUFFER_SIZE,
    seed: int = SEED,
) -> Dataset:
    """Return a random Ultra‑FineWeb sample meeting ``target_chars`` length."""

    stream = load_dataset("openbmb/Ultra-FineWeb", split="en", streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=buffer_size)

    picked: List[dict] = []
    char_count = 0

    for row in tqdm(stream, desc="sampling"):
        text = row["content"]
        char_count += len(text)
        picked.append(row)
        if char_count >= target_chars:
            break

    return Dataset.from_list(picked)


if __name__ == "__main__":  # pragma: no cover – CLI entry‑point
    ds = sample_fineweb()
    ds.push_to_hub(HF_REPO, private=False)