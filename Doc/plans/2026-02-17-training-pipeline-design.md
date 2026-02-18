# Gap 8: Training Data Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Auto-collect HITL corrections as Qwen2.5-VL training data; provide standalone training script.

**Architecture:** When a user corrects an extraction, the server saves the page image + full extraction (with corrections applied) as a Qwen2.5-VL conversation sample. A separate `train_qwen2vl.py` script reads these samples and runs QLoRA fine-tuning. On startup, the extractor auto-loads the LoRA adapter if one exists.

**Tech Stack:** `peft`, `trl`, `transformers`, `bitsandbytes`, `Pillow`

---

## Task 1: Create Training Data Collector

**Files:**

- Create: `backend/training_collector.py`

### What it stores per sample

Every correction saves the MAXIMUM useful signal for training:

```python
{
    # ─── Identity ───
    "id": "uuid",
    "timestamp": "2026-02-17T15:30:00",
    "source_pdf": "CPL Form 2.pdf",
    "page_num": 1,                          # multipage-ready

    # ─── Image ───
    "image_path": "training_data/images/CPL_Form_2_p1_abc123.png",

    # ─── Extraction context ───
    "fields_requested": ["Patient Name", "DOB", "Address", ...],
    "model_used": "qwen",
    "voting_rounds": 3,

    # ─── Complete results (with corrections merged) ───
    "extraction_results": {
        "Patient Name": "TILLISS, LENORE",
        "DOB": "01/02/1938",
        ...
    },

    # ─── What was corrected ───
    "corrections": {
        "Patient Name": {
            "original": "TILLISS, LENOORE",
            "corrected": "TILLISS, LENORE"
        }
    },

    # ─── Confidence data (helps identify weak fields) ───
    "confidences": {
        "Patient Name": 0.67,
        "DOB": 0.95,
        ...
    }
}
```

### Qwen2.5-VL conversation format export

The training script converts each sample to the official Qwen2.5-VL format:

```json
{
  "image": "training_data/images/CPL_Form_2_p1_abc123.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nExtract these fields from the document image and return as JSON: Patient Name, DOB, Address, ...\nReturn a JSON object with exactly these keys.\nFor each field, find the matching label in the document and copy its value exactly as written.\nInclude complete multi-line values (e.g. full addresses with city, state, zip).\nIf a field is not found, set its value to empty string.\n\nExample output format:\n{\"Patient Name\": \"DOE, JOHN\", \"DOB\": \"01/15/1980\", \"Address\": \"123 MAIN ST, SPRINGFIELD, IL 62701\"}\n\nNow extract from this document:"
    },
    {
      "from": "gpt",
      "value": "{\"Patient Name\": \"TILLISS, LENORE\", \"DOB\": \"01/02/1938\", ...}"
    }
  ]
}
```

### Key design choices

1. **Save ALL fields, not just corrections** — the model learns the full extraction task, not just individual fixes
2. **Corrections are merged into results** — the "gpt" answer contains the corrected values mixed with approved originals = ground truth
3. **Approvals also count** — when user approves an AI value, it confirms correctness. We can generate training samples from fully-approved extractions too (with `"corrections": {}`)
4. **Multipage-ready** — `page_num` tracked per sample, images named with page number

### Step 1: Write `training_collector.py`

```python
class TrainingDataCollector:
    def __init__(self, data_dir="training_data")
    def save_sample(self, image, filename, page_num, fields, results, corrections, confidences, model_used, voting_rounds) -> str
    def export_for_training(self) -> str  # Convert samples.jsonl → qwen2vl_train.json
    def get_stats(self) -> dict
    def get_samples(self) -> list
    def clear(self)
```

### Step 2: Wire into `server.py`

In `resolve_review()` — when a correction is made:

1. Load the extraction context (image, filename, page_num)
2. Load the full extraction results from the most recent extraction
3. Merge the correction into results
4. Call `collector.save_sample()`

In `extract_fields()` — after extraction:

1. Store the full results + confidences in `_extraction_context` (not just the image)

---

## Task 2: Create Training Script

**Files:**

- Create: `backend/train_qwen2vl.py`

### QLoRA Configuration (from Qwen2.5-VL official docs + Context7)

```python
# LoRA config (official Qwen2.5-VL values)
lora_r = 8
lora_alpha = 16
lora_dropout = 0.0
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

# Training config
learning_rate = 2e-5
epochs = 3
batch_size = 1  # GPU constrained
gradient_accumulation_steps = 8  # effective batch = 8
max_length = 4096
```

### CLI interface

```bash
# Basic training
python train_qwen2vl.py

# Custom options
python train_qwen2vl.py --epochs 5 --lr 1e-5 --output adapters/my_adapter

# Data stats only (no training)
python train_qwen2vl.py --stats-only
```

### Steps

1. Load training data from `training_data/qwen2vl_train.json`
2. Load base model with 4-bit quantization
3. Apply LoRA adapters
4. Train with SFTTrainer from `trl`
5. Save adapter to `backend/adapters/qwen2vl_lora/`

---

## Task 3: Auto-Load LoRA Adapter

**Files:**

- Modify: `backend/models/qwen2vl_extractor.py`
- Modify: `backend/config.py`

### Steps

1. Add `QWEN2VL_LORA_PATH` to config.py (default: `backend/adapters/qwen2vl_lora/`)
2. In `get_qwen2vl_model()`, after loading base model, check if LoRA adapter exists
3. If exists, load with `PeftModel.from_pretrained()`
4. Print adapter info (trainable params, when trained)

---

## Task 4: Expand Extraction Context for Training

**Files:**

- Modify: `backend/server.py`

### Steps

1. Expand `_extraction_context` to also store: `fields_requested`, `results`, `confidences`, `model_used`, `voting_rounds`
2. Update `save_extraction_context()` to accept all fields
3. Wire `resolve_review()` to call `collector.save_sample()` with full context
4. Also save training samples for fully-approved extractions (all approvals = ground truth confirmation)

---

## Verification Plan

### Automated

```bash
# Syntax check
python -c "import ast; ast.parse(open('backend/training_collector.py').read())"
python -c "import ast; ast.parse(open('backend/train_qwen2vl.py').read())"
python -c "import ast; ast.parse(open('backend/server.py').read())"

# Training data export test
python -c "from training_collector import get_training_collector; print(get_training_collector().get_stats())"
```

### Manual

1. Run extraction → correct a field → verify sample saved to `training_data/`
2. Run `python train_qwen2vl.py --stats-only` → verify sample count
3. Verify server starts without errors
