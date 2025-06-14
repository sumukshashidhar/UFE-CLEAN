"""Score Ultra-FineWeb with proper memory saturation."""
from __future__ import annotations
import torch
from datasets import load_dataset, IterableDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm

# Config
MODEL_NAME = "HuggingFaceTB/fineweb-edu-classifier"
DATASET_NAME = "sumuks/Ultra-FineWeb-10B"
OUTPUT_REPO = "sumuks/UFE-CLEAN-10B"
BATCH_SIZE = 4096
BUFFER_SIZE = 100_000  # Load this many samples into memory
STREAMING = False

accelerator = Accelerator(mixed_precision="bf16")

# Load with buffering
ds = load_dataset(DATASET_NAME, split="train", streaming=STREAMING)
ds = ds.rename_column("score", "openbmb-fasttext-classifier-score")

if STREAMING:
    # Add buffering to streaming dataset
    ds = ds.shuffle(buffer_size=BUFFER_SIZE)  # This creates an in-memory buffer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["content"], truncation=True)

# For streaming, use smaller map batch size but process more
tok_ds = ds.map(
    tokenize,
    batched=True,
    batch_size=1000,  # Smaller chunks for streaming
    remove_columns=ds.column_names,
)

# DataLoader with prefetching
collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
loader = DataLoader(
    tok_ds,
    batch_size=BATCH_SIZE,
    num_workers=8,  # More workers to keep buffer full
    pin_memory=True,
    collate_fn=collator,
    prefetch_factor=4,  # Prefetch multiple batches
)

# Model setup
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).eval()
model = torch.compile(model)
model, loader = accelerator.prepare(model, loader)

# For streaming, process in chunks and save incrementally
if STREAMING:
    chunk_scores = []
    chunk_idx = 0
    CHUNK_SIZE = 1_000_000
    
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(loader, disable=not accelerator.is_local_main_process)):
            logits = model(**batch).logits.squeeze(-1)
            
            if accelerator.use_distributed:
                gathered = accelerator.gather_for_metrics(logits)
                if accelerator.is_main_process:
                    chunk_scores.extend(gathered.cpu().float().tolist())
            else:
                chunk_scores.extend(logits.cpu().float().tolist())
            
            # Save chunk when it's large enough
            if len(chunk_scores) >= CHUNK_SIZE and accelerator.is_main_process:
                # Save this chunk
                torch.save(chunk_scores, f"scores_chunk_{chunk_idx}.pt")
                chunk_scores = []
                chunk_idx += 1
    
    # Save final chunk
    if chunk_scores and accelerator.is_main_process:
        torch.save(chunk_scores, f"scores_chunk_{chunk_idx}.pt")
else:
    # Non-streaming: collect all scores
    all_scores = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Scoring", disable=not accelerator.is_local_main_process):
            logits = model(**batch).logits.squeeze(-1)
            if accelerator.use_distributed:
                gathered = accelerator.gather_for_metrics(logits)
                if accelerator.is_main_process:
                    all_scores.extend(gathered.cpu().float().tolist())
            else:
                all_scores.extend(logits.cpu().float().tolist())
    
    # Save all at once
    if accelerator.is_main_process:
        final_ds = load_dataset(DATASET_NAME, split="train")
        final_ds = final_ds.rename_column("score", "openbmb-fasttext-classifier-score")
        final_ds = final_ds.add_column("fineweb-edu-classifier-score", all_scores)
        final_ds.push_to_hub(OUTPUT_REPO)

accelerator.end_training()