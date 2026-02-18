# Human-in-the-Loop (HITL) Guide

Complete guide to the HITL system for PDF AI Extraction. This system collects human corrections, trains the model, and continuously improves extraction accuracy.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Step-by-Step Workflow](#step-by-step-workflow)
4. [Data Collection](#data-collection)
5. [Review Queue UI](#review-queue-ui)
6. [Training Data Management](#training-data-management)
7. [Training Process](#training-process)
8. [LoRA Adapter Auto-Loading](#lora-adapter-auto-loading)
9. [Configuration Reference](#configuration-reference)
10. [API Reference](#api-reference)
11. [File Structure](#file-structure)
12. [Troubleshooting](#troubleshooting)
13. [Best Practices](#best-practices)

---

## Overview

The HITL system creates a closed feedback loop:

```
Extract PDF  -->  User reviews  -->  Corrections saved  -->  Model trained  -->  Better extraction
     ^                                                                               |
     |_______________________________________________________________________________|
```

**Why it matters:** Every correction you make teaches the model to avoid the same mistake on similar documents. After 20-50 corrections, the model can improve extraction accuracy by 5-15% on your specific document types.

---

## Architecture

### Components

| Component          | File                    | Purpose                                      |
| ------------------ | ----------------------- | -------------------------------------------- |
| Confidence Router  | `server.py`             | Routes low-confidence fields to Review Queue |
| HITL Manager       | `hitl_manager.py`       | Stores and manages the review queue          |
| Review Queue UI    | `review_queue.js`       | Frontend for approving/correcting fields     |
| Training Collector | `training_collector.py` | Saves corrections as training samples        |
| Training Script    | `train_qwen2vl.py`      | QLoRA fine-tuning with collected data        |
| LoRA Auto-Loader   | `qwen2vl_extractor.py`  | Loads trained adapter at startup             |
| Config             | `config.py`             | `QWEN2VL_LORA_PATH` setting                  |

### Data Flow Diagram

```
       EXTRACTION PHASE                     REVIEW PHASE                    TRAINING PHASE
 ┌─────────────────────────┐       ┌──────────────────────────┐      ┌───────────────────────┐
 │  1. Upload PDF           │       │  4. Review Queue shows    │      │  7. Export samples     │
 │  2. Qwen2.5-VL extracts  │──────>│     low-confidence fields │      │  8. QLoRA fine-tunes   │
 │  3. Confidence scores    │       │  5. User approves/corrects│────> │  9. LoRA adapter saved  │
 │     calculated per field  │       │  6. Training sample saved │      │ 10. Auto-loaded on     │
 └─────────────────────────┘       └──────────────────────────┘      │     next startup       │
                                                                      └───────────────────────┘
```

---

## Step-by-Step Workflow

### Phase 1: Extract a PDF

1. Open the app at `http://localhost:5173`
2. Upload a PDF document
3. Select or add fields to extract
4. Click **Extract**
5. The system extracts fields and calculates **confidence scores** per field

### Phase 2: Review Flagged Fields

6. Fields with confidence **below 70%** are automatically flagged for review
7. Click the **Review** tab in the navigation
8. For each flagged field, you see:
   - The field name and source file
   - The AI's predicted value
   - The confidence percentage (color-coded: green > 70%, yellow 40-70%, red < 40%)
9. For each field, choose one action:
   - **Approve** — The AI's value is correct. Click the checkmark button.
   - **Correct** — The AI's value is wrong. Type the correct value in the input box and click the edit button.
   - **Delete** — Remove this item from the queue (counts as resolved).

### Phase 3: Training Data is Collected Automatically

10. When you **approve** or **correct** a field, the system automatically saves a training sample.
11. Each sample includes:
    - The full page image (PNG)
    - All fields that were requested
    - The ground truth values (with your corrections merged in)
    - What was corrected (original vs corrected value)
    - Confidence scores, model used, voting rounds, timestamp

### Phase 4: Train the Model

12. Once you have 20+ samples (50+ recommended), run training:

```bash
cd backend

# Check how many samples you have
python train_qwen2vl.py --stats-only

# Export and train
python train_qwen2vl.py
```

13. Training takes 10-60 minutes depending on GPU and sample count
14. A LoRA adapter is saved to `backend/adapters/qwen2vl_lora/`

### Phase 5: Improved Model Auto-Loads

15. **Restart the server** (`.\start.bat`)
16. On startup, the extractor detects the LoRA adapter and loads it automatically
17. You'll see in the server console:

```
Loading LoRA adapter from backend/adapters/qwen2vl_lora...
LoRA adapter loaded (trained on 47 samples, loss: 0.23)
```

18. All future extractions use the improved model

---

## Data Collection

### What Gets Saved

Every time you approve or correct a field in the Review tab, the system stores a **complete training sample**:

```json
{
  "id": "a1b2c3d4",
  "timestamp": "2026-02-18T12:30:00",
  "source_pdf": "CPL Form 2.pdf",
  "page_num": 1,
  "image_path": "training_data/images/CPL_Form_2_p1_a1b2c3d4.png",
  "fields_requested": ["Patient Name", "DOB", "Address", "Phone"],
  "model_used": "qwen",
  "voting_rounds": 3,
  "extraction_results": {
    "Patient Name": "LENORE SMITH",
    "DOB": "01/15/1980",
    "Address": "123 MAIN ST, SPRINGFIELD, IL 62701",
    "Phone": "555-0123"
  },
  "corrections": {
    "Patient Name": {
      "original": "LENOORE SMITH",
      "corrected": "LENORE SMITH"
    }
  },
  "confidences": {
    "Patient Name": 0.45,
    "DOB": 0.92,
    "Address": 0.88,
    "Phone": 0.95
  },
  "is_corrected": true
}
```

### Storage Location

```
backend/
└── training_data/
    ├── images/                           # Page images (PNG files)
    │   ├── CPL_Form_2_p1_a1b2c3d4.png
    │   ├── Order_Form_p1_e5f6g7h8.png
    │   └── ...
    ├── samples.jsonl                     # One JSON object per line
    └── metadata.json                     # Stats and counters
```

### Approved vs Corrected Samples

Both types are valuable for training:

- **Corrected samples** — MOST valuable. Teach the model what it got wrong.
- **Approved samples** — Reinforce correct behavior. Prevent catastrophic forgetting.

The system saves both automatically. You don't need to do anything special.

---

## Review Queue UI

### Training Panel

The training panel at the top of the Review tab shows:

| Element          | Description                                                                     |
| ---------------- | ------------------------------------------------------------------------------- |
| Sample count     | Total training samples collected                                                |
| Correction count | How many fields were corrected                                                  |
| Document count   | Unique PDFs processed                                                           |
| Readiness badge  | **Collecting** (< 20), **Can begin training** (20-49), **Ready to train** (50+) |
| Recommendation   | AI-generated advice on when to train                                            |

### Buttons

| Button          | Action                                                     |
| --------------- | ---------------------------------------------------------- |
| **View Data**   | Toggle the training data panel showing all samples         |
| **Export Data** | Export samples to Qwen2.5-VL format (`qwen2vl_train.json`) |

### Review Cards

Each review card shows:

- **Field name** — Which field was flagged
- **Source file** — The PDF it came from
- **Confidence %** — Color-coded (green/yellow/red)
- **AI value** — What the model predicted
- **Approve** — Accept the prediction as correct
- **Correct** — Enter the right value
- **Delete** — Remove from queue

---

## Training Data Management

### Checking Stats (CLI)

```bash
cd backend
python train_qwen2vl.py --stats-only
```

Output:

```
============================================================
Training Data Statistics
============================================================
  Total samples:      47
  Total corrections:  23
  Total approvals:    24
  Unique documents:   8
  Unique fields:      12
  Last export:        Never
  Ready for training: Yes

  47 samples. Good amount for strong improvement (+10-12%). Ready to train.

  Documents:
    - CPL Form 2.pdf: 15 samples
    - Order Form.pdf: 12 samples
    - Medical Record.pdf: 20 samples

  Most corrected fields:
    - Patient Name
    - Address
    - Date of Birth
============================================================
```

### Checking Stats (API)

```
GET http://localhost:8000/api/training/stats
```

Returns:

```json
{
  "total_samples": 47,
  "total_corrections": 23,
  "total_approvals": 24,
  "unique_documents": 8,
  "unique_fields": 12,
  "most_corrected_fields": ["Patient Name", "Address"],
  "ready_for_training": true,
  "recommendation": "47 samples. Good amount for strong improvement (+10-12%)."
}
```

### Exporting Data

**From the UI:** Click the **Export Data** button in the Review tab.

**From CLI:**

```bash
python train_qwen2vl.py --export-only
```

**From API:**

```
POST http://localhost:8000/api/training/export
```

All methods produce `training_data/qwen2vl_train.json` in the official Qwen2.5-VL conversation format.

### Clearing Data

**From API (use with caution):**

```
DELETE http://localhost:8000/api/training/clear
```

This deletes all samples and images.

---

## Training Process

### Prerequisites

```bash
pip install peft trl accelerate bitsandbytes datasets
```

Requirements:

- NVIDIA GPU with CUDA support
- Minimum 4 GB VRAM (6+ GB recommended)
- At least 20 training samples (50+ recommended)

### Running Training

```bash
cd backend

# Full pipeline: show stats, export data, train
python train_qwen2vl.py

# Custom options
python train_qwen2vl.py --epochs 5 --lr 1e-5 --lora-r 16 --lora-alpha 32
```

### CLI Arguments

| Argument        | Default                 | Description                        |
| --------------- | ----------------------- | ---------------------------------- |
| `--data-dir`    | `training_data`         | Directory containing training data |
| `--output`      | `adapters/qwen2vl_lora` | Output directory for LoRA adapter  |
| `--epochs`      | `3`                     | Number of training epochs          |
| `--lr`          | `2e-5`                  | Learning rate                      |
| `--batch-size`  | `1`                     | Per-device batch size              |
| `--grad-accum`  | `8`                     | Gradient accumulation steps        |
| `--lora-r`      | `8`                     | LoRA rank                          |
| `--lora-alpha`  | `16`                    | LoRA alpha                         |
| `--max-length`  | `2048`                  | Max sequence length                |
| `--stats-only`  | —                       | Only show stats, don't train       |
| `--export-only` | —                       | Only export data, don't train      |

### What Training Does Internally

1. **Loads samples** from `training_data/qwen2vl_train.json`
2. **Loads the base Qwen2.5-VL model** with 4-bit quantization (NF4)
3. **Applies QLoRA** with rank=8, alpha=16, targeting `q_proj`, `v_proj`, `k_proj`, `o_proj`
4. **Trains** for 3 epochs with gradient checkpointing (fits in ~4-6 GB VRAM)
5. **Saves** the LoRA adapter + training metadata to `adapters/qwen2vl_lora/`

### What QLoRA Is

QLoRA (Quantized Low-Rank Adaptation) is a memory-efficient fine-tuning method:

- **Quantization (Q):** The base model is loaded in 4-bit precision, using ~75% less VRAM
- **LoRA:** Only trains ~0.5% of parameters (small rank-decomposition matrices added to attention layers)
- **Result:** Full fine-tuning quality at a fraction of the compute cost

### Training Duration Estimates

| Samples | GPU (4GB VRAM) | GPU (8GB VRAM) | GPU (16GB VRAM) |
| ------- | -------------- | -------------- | --------------- |
| 20      | ~10 min        | ~5 min         | ~3 min          |
| 50      | ~25 min        | ~12 min        | ~7 min          |
| 100     | ~50 min        | ~25 min        | ~15 min         |

### Expected Accuracy Improvement

| Samples | Expected Improvement                               |
| ------- | -------------------------------------------------- |
| 20-30   | +5-8% accuracy on similar documents                |
| 50-75   | +10-12% accuracy                                   |
| 100+    | +12-15% accuracy (diminishing returns beyond this) |

---

## LoRA Adapter Auto-Loading

### How It Works

When the server starts and loads the Qwen2.5-VL model:

1. It checks if `backend/adapters/qwen2vl_lora/adapter_config.json` exists
2. If found, it calls `PeftModel.from_pretrained()` to merge the LoRA weights
3. The merged model is used for all subsequent extractions
4. If `training_meta.json` exists, it logs the training details

### Code Location

```python
# In backend/models/qwen2vl_extractor.py, after model loads:
from config import QWEN2VL_LORA_PATH
lora_path = Path(QWEN2VL_LORA_PATH)
if lora_path.exists() and (lora_path / "adapter_config.json").exists():
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(lora_path))
    model.eval()
```

### Disabling the Adapter

To temporarily disable the LoRA adapter without deleting it:

- Rename the adapter directory: `adapters/qwen2vl_lora` to `adapters/qwen2vl_lora_disabled`
- Restart the server

### Retraining

When you collect more corrections and want to retrain:

1. Run `python train_qwen2vl.py` again
2. The new adapter overwrites the old one
3. Restart the server to load the updated adapter

---

## Configuration Reference

### config.py

```python
# Path to the LoRA adapter directory
QWEN2VL_LORA_PATH = os.path.join(BASE_DIR, "adapters", "qwen2vl_lora")
```

### Confidence Threshold

In `server.py`, the HITL routing threshold:

```python
CONFIDENCE_THRESHOLD = 0.7  # Fields below this go to review queue
```

Fields with confidence below 0.7 (70%) are automatically flagged. Adjust this to be stricter (higher) or more lenient (lower).

---

## API Reference

### Review Queue

| Method | Endpoint                    | Description                                |
| ------ | --------------------------- | ------------------------------------------ |
| GET    | `/api/reviews`              | List all review items (pending + resolved) |
| POST   | `/api/reviews/{id}/resolve` | Approve, correct, or delete an item        |
| DELETE | `/api/reviews/clear`        | Clear the entire review queue              |
| POST   | `/api/flag`                 | Manually flag a field for review           |

### Training Data

| Method | Endpoint                | Description                                  |
| ------ | ----------------------- | -------------------------------------------- |
| GET    | `/api/training/stats`   | Get sample counts, readiness, recommendation |
| GET    | `/api/training/samples` | List all training samples (lightweight)      |
| POST   | `/api/training/export`  | Export to Qwen2.5-VL format                  |
| DELETE | `/api/training/clear`   | Delete all training data                     |

### Request Examples

**Resolve a review item (correct):**

```bash
curl -X POST http://localhost:8000/api/reviews/abc123/resolve \
  -H "Content-Type: application/json" \
  -d '{"action": "correct", "corrected_value": "LENORE SMITH"}'
```

**Resolve a review item (approve):**

```bash
curl -X POST http://localhost:8000/api/reviews/abc123/resolve \
  -H "Content-Type: application/json" \
  -d '{"action": "approve"}'
```

**Export training data:**

```bash
curl -X POST http://localhost:8000/api/training/export
```

---

## File Structure

```
backend/
├── adapters/
│   └── qwen2vl_lora/                # Trained LoRA adapter (created after training)
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── training_meta.json
├── training_data/                    # Collected training samples
│   ├── images/                       # Page images (PNG)
│   ├── samples.jsonl                 # Raw samples (one per line)
│   ├── metadata.json                 # Counters and stats
│   └── qwen2vl_train.json           # Exported format (created by export)
├── hitl_manager.py                   # Review queue management
├── training_collector.py             # Training data collection
├── train_qwen2vl.py                  # Training script
├── config.py                         # QWEN2VL_LORA_PATH setting
├── server.py                         # API endpoints + confidence routing
└── models/
    └── qwen2vl_extractor.py          # LoRA auto-loading

src/
└── components/
    ├── review_queue.js               # Review Queue UI component
    └── icons.js                      # SVG icon system
```

---

## Troubleshooting

### "No training samples yet"

**Cause:** You haven't approved or corrected any fields in the Review tab yet.

**Fix:** Extract a PDF, then go to the Review tab and approve or correct the flagged fields.

### "CUDA not available"

**Cause:** No NVIDIA GPU found, or CUDA not installed.

**Fix:**

- Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads
- Or use Google Colab (free T4 GPU with 16 GB VRAM)
- Or use cloud GPUs (RunPod, Lambda Labs)

### "OOM (Out of Memory) during training"

**Cause:** Not enough GPU VRAM.

**Fix:**

- Reduce max length: `--max-length 1024`
- Reduce LoRA rank: `--lora-r 4 --lora-alpha 8`
- Ensure gradient checkpointing is enabled (it is by default)
- Use a GPU with more VRAM (6+ GB recommended)

### LoRA adapter not loading

**Cause:** Adapter directory doesn't contain `adapter_config.json`.

**Fix:**

- Verify the file exists: `backend/adapters/qwen2vl_lora/adapter_config.json`
- Check the console for error messages on startup
- Re-run training: `python train_qwen2vl.py`

### "Fields not appearing in Review tab"

**Cause:** All fields have confidence above the threshold (70%).

**Fix:** This is normal for clean, well-formatted PDFs. You can:

- Manually flag a field by clicking the flag icon on any result card
- Lower the threshold in `server.py`: `CONFIDENCE_THRESHOLD = 0.5`

---

## Best Practices

### Getting the Most Out of Training

1. **Diverse documents** — Train on different PDF layouts, not just one form type
2. **Correct precisely** — Type exact values as they appear in the document
3. **Include multi-line values** — Full addresses, complete names with suffixes
4. **Approve correct fields too** — Approvals reinforce good behavior
5. **Train after 50+ samples** — Better results than training at 20

### When to Retrain

- After collecting 20-30 new corrections since last training
- When you start extracting a new type of document
- When accuracy on a specific field drops

### Training Cadence

| Usage Pattern             | Training Frequency |
| ------------------------- | ------------------ |
| Low volume (1-5 PDFs/day) | Monthly            |
| Medium (5-20 PDFs/day)    | Weekly             |
| High (20-100 PDFs/day)    | Every 2-3 days     |

### What NOT to Do

- Don't train with fewer than 5 samples (results will be unreliable)
- Don't correct values to something not in the document (hallucination risk)
- Don't clear training data unless you're starting a completely different project
- Don't run training while the server is processing extractions (GPU contention)
