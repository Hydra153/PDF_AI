"""
Configuration settings for the backend server
Auto-selects the best Qwen VL model based on available GPU VRAM.
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Device settings (auto-detect GPU)
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_GPU = torch.cuda.is_available()

# ─── VRAM-Based Model Auto-Selection ───
# Picks the best Qwen VL model that fits in available VRAM.
# Override with environment variable: QWEN_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

QWEN_MODEL_TIERS = [
    # (min_vram_gb, model_id, display_name)
    # Thresholds assume 4-bit NF4 quantization (actual VRAM: model_params/2 + overhead)
    (10.0, "Qwen/Qwen2.5-VL-7B-Instruct",  "Qwen2.5-VL-7B"),   # 95% DocVQA
    (3.5,  "Qwen/Qwen2.5-VL-3B-Instruct",  "Qwen2.5-VL-3B"),   # 92% DocVQA — ~1.8GB in 4-bit
    (0.0,  "Qwen/Qwen2-VL-2B-Instruct",    "Qwen2-VL-2B"),     # 88% DocVQA
]

def _select_best_model():
    """Auto-select the best model that fits in available VRAM."""
    # Allow manual override via environment variable
    env_model = os.environ.get("QWEN_MODEL")
    if env_model:
        return env_model, env_model.split("/")[-1]
    
    if not USE_GPU:
        # CPU mode — use smallest model
        return QWEN_MODEL_TIERS[-1][1], QWEN_MODEL_TIERS[-1][2]
    
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    for min_vram, model_id, display_name in QWEN_MODEL_TIERS:
        if vram_gb >= min_vram:
            return model_id, display_name
    
    # Fallback to smallest
    return QWEN_MODEL_TIERS[-1][1], QWEN_MODEL_TIERS[-1][2]


QWEN2VL_MODEL, QWEN2VL_DISPLAY_NAME = _select_best_model()
QWEN2VL_QUANTIZATION = "4bit"   # 4bit NF4 quantization for low VRAM
QWEN2VL_MAX_BATCH = 5           # Max fields per inference batch

# ─── VRAM-Based Image Resolution ───
# Qwen2-VL's processor has built-in smart_resize() that handles dynamic resolution.
# We configure min_pixels/max_pixels to control VRAM usage during inference.
# The processor resizes internally — no manual PIL resize needed.
# Ref: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct (Usage section)

def _select_max_pixels():
    """Select max_pixels for processor based on GPU VRAM.
    
    Qwen2-VL default: max_pixels = 1280 * 28 * 28 = 1,003,520 (~1M pixels).
    We use a higher budget for better detail on form documents.
    The 2B model is designed to work at this resolution.
    Only reduce for very low VRAM or CPU mode.
    """
    if not USE_GPU:
        return 512 * 28 * 28   # ~401K pixels for CPU mode
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    # NOTE: GPUs report slightly below marketing specs (4GB → 3.999, 8GB → 7.999)
    # Use conservative thresholds to avoid misclassification
    if vram >= 7.0:
        return 1680 * 28 * 28  # ~1.3M pixels — higher detail for forms
    elif vram >= 3.5:
        return 1680 * 28 * 28  # ~1.3M pixels — higher detail with 4-bit quantization
    else:
        return 512 * 28 * 28   # ~401K pixels — very low VRAM safety net

QWEN2VL_MIN_PIXELS = 256 * 28 * 28    # ~200K — processor default
QWEN2VL_MAX_PIXELS = _select_max_pixels()

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = True

# CORS settings
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",
    "http://127.0.0.1:3000",
    "*"  # Allow all for development
]

# Upload settings
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_PDF_PAGES = 50                # Max pages to process per PDF
ALLOWED_EXTENSIONS = {".pdf"}

# ─── Image Enhancement for Scanned PDFs ───
# Applies auto-contrast, brightness correction, sharpening, and denoising
# to scanned document images before VLM extraction. Improves character
# visibility on dark/blurry scans. Disable for clean digital PDFs.
ENHANCE_SCANNED_IMAGES = True

# ─── LoRA Adapter (fine-tuned model) ───
# If a trained LoRA adapter exists at this path, the extractor auto-loads it.
# Train with: python train_qwen2vl.py
QWEN2VL_LORA_PATH = os.path.join(BASE_DIR, "adapters", "qwen2vl_lora")

# Logging
LOG_LEVEL = "INFO"

# ─── Startup Info ───
print(f"🔧 Configuration loaded:")
print(f"   Device: {DEVICE}")
print(f"   GPU Available: {USE_GPU}")
if USE_GPU:
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {vram_gb:.2f} GB")
    print(f"   Model (auto-selected): {QWEN2VL_DISPLAY_NAME} ({QWEN2VL_QUANTIZATION})")
    print(f"   Model ID: {QWEN2VL_MODEL}")
    print(f"   Pixel budget: {QWEN2VL_MIN_PIXELS:,} – {QWEN2VL_MAX_PIXELS:,} pixels")
else:
    print(f"   Model: {QWEN2VL_DISPLAY_NAME} (CPU mode)")
