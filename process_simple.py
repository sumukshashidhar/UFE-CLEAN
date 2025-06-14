from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME   = "HuggingFaceTB/fineweb-edu-classifier"
DATASET_NAME = "sumuks/Ultra-FineWeb-1M"
PROCESSED_DATASET_NAME = "sumuks/UFE-CLEAN-1M"
BATCH_SIZE = 4096


ultrafineweb = load_dataset(DATASET_NAME, split="train").select_columns(["content", "score"])
ultrafineweb = ultrafineweb.rename_column("score", "openbmb_fasttext_classifier_score")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = (
    AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    .to(dtype=torch.bfloat16 if DEVICE.type == "cuda" else torch.float32)
    .to(DEVICE)
    .eval()
)
model = torch.compile(model)  # PyTorch 2.x graph optimiser

def tok_fn(batch):
    return tokenizer(batch["content"], truncation=True)

raw_ds   = load_dataset(DATASET_NAME, split="train")
original_scores = raw_ds["score"]
tok_ds   = raw_ds.map(
    tok_fn,
    batched=True,
    batch_size=BATCH_SIZE,
    num_proc=96,
    remove_columns=raw_ds.column_names,
)

collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
loader   = DataLoader(
    tok_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=96,
    collate_fn=collator,
)

scores: list[float] = []
with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    for batch in tqdm(loader, desc="Infer"):
        batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
        logits = model(**batch).logits.squeeze(-1)
        scores.extend(logits.cpu().float().tolist())

ultrafineweb = ultrafineweb.add_column("fineweb_edu_classifier_score", scores)

# Save the dataset
ultrafineweb.push_to_hub(PROCESSED_DATASET_NAME)