"""
FastAPI Server for PDF Data Extraction
Provides REST API endpoints for document understanding and field extraction

Extraction engine: Qwen2.5-VL-3B (Vision-Language Model, OCR-free)
"""
import os
import time
import asyncio
import logging
import warnings

# ─── Suppress noisy third-party logs ───
# httpx: ~40 lines of HuggingFace Hub cache-check HTTP requests per model load
# huggingface_hub: "unauthenticated requests" warning on every model load
# transformers: "image processor loaded as fast processor" deprecation notice
# watchfiles: "N changes detected" noise during dev hot-reload
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", message=".*fast processor by default.*")

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import json
from pydantic import BaseModel

from config import CORS_ORIGINS, MAX_FILE_SIZE, MAX_PDF_PAGES, ALLOWED_EXTENSIONS, QWEN2VL_MODEL, QWEN2VL_DISPLAY_NAME
from utils.pdf_processor import process_pdf, process_image, is_supported_file, is_image_file
from hitl_manager import get_hitl_manager
from training_collector import get_training_collector

# ─── Qwen2-VL Model ───
_qwen2vl_available = False
try:
    from models.qwen2vl_extractor import Qwen2VLExtractor, unload_qwen2vl_model
    _qwen2vl_available = True
    print(f"✅ {QWEN2VL_DISPLAY_NAME} available")
except ImportError as e:
    print(f"❌ {QWEN2VL_DISPLAY_NAME} not available: {e}")

if not _qwen2vl_available:
    print("⚠️ WARNING: No extraction model available!")


# ─── FastAPI App ───
app = FastAPI(
    title="PDF Data Extractor API",
    description="Document understanding and field extraction using Qwen VL",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Extraction Context (for HITL training data) ───
# Stores the COMPLETE extraction context so training samples can be saved
# when the user later approves or corrects fields.
_extraction_context = {
    "image": None,
    "filename": "",
    "page_num": 1,
    "fields_requested": [],
    "results": {},
    "signals": {},
    "model_used": "qwen",
    "voting_rounds": 1,
}

# ─── GPU Concurrency Protection (FIX 3) ───
# Semaphore(1) = only 1 GPU inference at a time. Prevents VRAM corruption
# when multiple requests arrive simultaneously.
# Reference: https://docs.python.org/3/library/asyncio-sync.html#asyncio.Semaphore
_gpu_semaphore = asyncio.Semaphore(1)
_GPU_TIMEOUT_SECONDS = 300  # Max wait time — PaddleOCR-VL first load can take ~60-120s

# ─── Extraction Progress Tracking ───
_extraction_progress = {
    "active": False,
    "step": 0,
    "total": 0,
    "message": "",
    "percent": 0,
}

def _set_progress(step: int, total: int, message: str):
    """Update extraction progress for frontend polling."""
    _extraction_progress["active"] = True
    _extraction_progress["step"] = step
    _extraction_progress["total"] = total
    _extraction_progress["message"] = message
    _extraction_progress["percent"] = round((step / total) * 100) if total > 0 else 0

def _clear_progress():
    _extraction_progress["active"] = False
    _extraction_progress["step"] = 0
    _extraction_progress["total"] = 0
    _extraction_progress["message"] = ""
    _extraction_progress["percent"] = 0

@app.get("/api/progress")
async def get_progress():
    return _extraction_progress

# ─── Image Cache (multi-page) ───
# Caches processed (enhanced) and raw images per PDF to avoid redundant
# PDF→image conversion + enhancement on every /api/ask or /api/re-extract call.
# Keyed by MD5 hash of PDF bytes. In-memory only — cleared on new PDF or server restart.
import hashlib

_image_cache = {
    "hash": None,        # MD5 of the cached PDF
    "filename": None,    # Original filename
    "enhanced": [],      # List of enhanced PIL Images (one per page)
    "raw": [],           # List of raw PIL Images (one per page)
}

def get_or_process_file(file_bytes, filename="", need_raw=False):
    """
    Return cached images if the file hasn't changed, otherwise process and cache.
    Automatically detects PDF vs image files.
    Returns (enhanced_images: List[Image], raw_images: List[Image] or []).
    """
    global _image_cache
    file_hash = hashlib.md5(file_bytes).hexdigest()
    
    if _image_cache["hash"] == file_hash and _image_cache["enhanced"]:
        # Cache hit — but lazily load raw if needed and not yet cached
        if need_raw and not _image_cache["raw"]:
            if is_image_file(filename):
                import io
                from PIL import Image as PILImage
                _image_cache["raw"] = [PILImage.open(io.BytesIO(file_bytes)).convert('RGB')]
            else:
                from utils.pdf_processor import pdf_to_images
                _image_cache["raw"] = pdf_to_images(file_bytes)
        raw = _image_cache["raw"] if need_raw else []
        return _image_cache["enhanced"], raw
    
    # Cache miss — process file (PDF or image)
    if is_image_file(filename):
        images, _ = process_image(file_bytes)
    else:
        images, _ = process_pdf(file_bytes)
    
    if not images:
        return [], []
    
    # Enforce page limit
    if len(images) > MAX_PDF_PAGES:
        raise ValueError(f"PDF has {len(images)} pages (max {MAX_PDF_PAGES}). Please split the document.")
    
    raw = []
    if need_raw:
        if is_image_file(filename):
            import io
            from PIL import Image as PILImage
            raw = [PILImage.open(io.BytesIO(file_bytes)).convert('RGB')]
        else:
            from utils.pdf_processor import pdf_to_images
            raw = pdf_to_images(file_bytes)
    
    # Store in cache
    _image_cache = {
        "hash": file_hash,
        "filename": filename,
        "enhanced": images,
        "raw": raw,
    }
    
    print(f"📄 Cached {len(images)} page(s) for {filename}")
    return images, raw

def clear_image_cache():
    """Clear the image cache (called when PDF changes or server shuts down)."""
    global _image_cache
    _image_cache = {"hash": None, "filename": None, "enhanced": [], "raw": []}

def save_extraction_context(image, filename, page_num=1, fields=None, results=None,
                           signals=None, model_used="qwen", voting_rounds=1):
    """Save complete extraction context for training data collection."""
    global _extraction_context
    _extraction_context = {
        "image": image,
        "filename": filename,
        "page_num": page_num,
        "fields_requested": fields or [],
        "results": dict(results) if results else {},
        "signals": dict(signals) if signals else {},
        "model_used": model_used,
        "voting_rounds": voting_rounds,
    }

def get_extraction_context():
    """Get the last extraction context."""
    return _extraction_context


# ─── Request/Response Models ───
class FieldRequest(BaseModel):
    key: str
    question: Optional[str] = None

class ExtractionResponse(BaseModel):
    success: bool
    data: dict
    message: str = ""

class ReviewDecision(BaseModel):
    action: str  # 'approve' or 'correct'
    corrected_value: Optional[str] = None

class FlagRequest(BaseModel):
    filename: str
    field_name: str
    ai_value: str
    signal: str = "manual_flag"
    reason: str = "Manually flagged for review"


# ─── Core Endpoints ───

@app.get("/")
async def root():
    return {"message": "PDF Data Extractor API", "version": "4.0.0", "status": "running"}


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy" if _qwen2vl_available else "degraded",
        "models": {
            "qwen": {
                "name": QWEN2VL_DISPLAY_NAME,
                "available": _qwen2vl_available,
            },
        },
        "model": QWEN2VL_DISPLAY_NAME,
        "model_id": QWEN2VL_MODEL,
        "model_available": _qwen2vl_available,
    }


@app.post("/api/extract", response_model=ExtractionResponse)
async def extract_fields(
    file: UploadFile = File(...),
    fields: str = Form(...),
    model: str = Form("qwen"),  # Only qwen supported
    voting_rounds: int = Form(1),    # 1 = normal, 3 = accuracy boost (majority voting)
    checkbox_enabled: str = Form("false"),  # "true" = run checkbox auto-detection
    raw_mode: str = Form("false"),  # "true" = skip image enhancement, feed raw image to VLM
    multipage: str = Form("false"),  # "true" = send all pages in one VLM call
):
    """
    Extract specified fields from PDF document using Qwen2-VL.
    
    FIX 3: Wrapped with asyncio.Semaphore(1) to serialize GPU access.
    If GPU is busy, request queues. If wait exceeds 120s, returns 503.
    
    Args:
        file: PDF file upload
        fields: JSON string of field list [{"key": "Name", "question": "..."}, ...]
    """
    try:
        # Validate
        if not is_supported_file(file.filename):
            raise HTTPException(status_code=400, detail="Only PDF and image files (PNG, JPG, TIFF, BMP) are supported")
        
        pdf_bytes = await file.read()
        if len(pdf_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        # Parse fields
        try:
            fields_list = json.loads(fields)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid fields JSON")
        
        if not fields_list:
            raise HTTPException(status_code=400, detail="No fields specified")
        
        # Validate model
        if not _qwen2vl_available:
            raise HTTPException(status_code=503, detail="Qwen2-VL model not available")
        
        # Process PDF → Images (cached, multi-page)
        t_start = time.time()
        _set_progress(1, 10, "Processing PDF...")
        print(f"📄 Processing PDF: {file.filename} (model: {model})")
        try:
            enhanced_images, raw_images = get_or_process_file(pdf_bytes, file.filename, need_raw=True)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        if not enhanced_images:
            raise HTTPException(status_code=400, detail="Could not process PDF")
        
        if not raw_images:
            raw_images = enhanced_images
        
        # DEV: Raw mode — bypass enhancement, feed raw images directly to VLM
        use_raw = raw_mode.lower() == "true"
        if use_raw:
            print(f"   🔧 DEV RAW MODE — skipping image enhancement, using raw images")
            enhanced_images = raw_images
        
        num_pages = len(enhanced_images)
        field_labels = [f['key'] for f in fields_list]
        print(f"🤖 Extracting {len(field_labels)} fields from {num_pages} page(s) with {model}...")
        _set_progress(2, 10, f"Extracting {len(field_labels)} fields from {num_pages} page(s)...")
        
        # ─── Multi-Page Extraction (serialized by GPU semaphore) ───
        all_results = {}          # field → value (first non-empty wins)
        all_signals = {}          # field → signal dict (includes page number)
        all_meta = {}             # merged extraction metadata
        
        try:
            async with asyncio.timeout(_GPU_TIMEOUT_SECONDS):
                async with _gpu_semaphore:
                    print(f"🔒 GPU semaphore acquired for extraction")
                    extractor = Qwen2VLExtractor()
                    
                    # ─── Multi-page fast-path: single VLM call across all pages ───
                    use_multipage = multipage.lower() == "true" and num_pages > 1
                    if use_multipage:
                        print(f"   📄 Multi-page mode: sending {num_pages} pages in one VLM call")
                        mp_results = extractor.extract_fields_multipage(enhanced_images, field_labels)
                        for field in field_labels:
                            value = mp_results.get(field, "")
                            all_results[field] = value
                            all_signals[field] = {
                                "source": "multipage",
                                "flags": [] if value.strip() else ["empty_value"],
                                "page": "all",
                            }
                        
                        # Checkbox multi-page
                        if checkbox_enabled.lower() == "true":
                            mp_checkboxes = extractor.extract_checkboxes_multipage(enhanced_images)
                            all_results["_checkboxes"] = mp_checkboxes
                    
                    # ─── Standard page-by-page extraction ───
                    if not use_multipage:
                        # ─── Table Pre-Scan: detect table structure per page ───
                        import re as _re
                        page_table_scans = {}  # page_idx → scan result
                        detected_columns = {}  # normalized_col_name → (original_col_name, page_idx)
                        detected_row_prefix = None  # e.g. "SL."
                        
                        for page_idx in range(num_pages):
                            scan = extractor.scan_table_structure(enhanced_images[page_idx])
                            page_table_scans[page_idx] = scan
                            if scan.get("has_table"):
                                for col in scan.get("columns", []):
                                    col_norm = col.strip().lower().rstrip(".")
                                    detected_columns[col_norm] = (col, page_idx)
                                if scan.get("row_index") and not detected_row_prefix:
                                    detected_row_prefix = scan["row_index"].strip().rstrip(".")
                        
                        if detected_columns:
                            print(f"   🔍 Detected table columns: {list(detected_columns.keys())}")
                            if detected_row_prefix:
                                print(f"   🔍 Row index prefix: '{detected_row_prefix}'")
                        
                        # ─── Smart field routing using scan results ───
                        table_column_fields = []   # field matches a column header → column extraction
                        table_row_fields = []      # field matches row_index + number → row extraction
                        table_keyword_fields = []  # field contains "table" → full table extraction
                        normal_fields = []         # everything else → normal batch extraction
                        
                        for f in field_labels:
                            f_lower = f.strip().lower()
                            f_norm = f_lower.rstrip(".")
                            
                            # Check if field contains "table" keyword
                            if "table" in f_lower:
                                table_keyword_fields.append(f)
                            # Check if field matches a detected column header (exact, case-insensitive)
                            elif f_norm in detected_columns:
                                table_column_fields.append(f)
                            # Check if field matches row index pattern: prefix + number (SL. 3, SL.3, SL 3)
                            elif detected_row_prefix:
                                prefix_norm = detected_row_prefix.lower()
                                # Match: "SL. 3", "SL.3", "SL 3", "sl.7", "Row 2", "#5"
                                row_pattern = rf'^({_re.escape(prefix_norm)}\.?\s*|row\s*(no\.?\s*)?|#)\s*\d+$'
                                if _re.match(row_pattern, f_lower):
                                    table_row_fields.append(f)
                                else:
                                    normal_fields.append(f)
                            # Check default row patterns (Row N, #N)
                            elif _re.match(r'^(row\s*(no\.?\s*)?|#)\s*\d+$', f_lower):
                                table_row_fields.append(f)
                            else:
                                normal_fields.append(f)
                        
                        if table_column_fields or table_row_fields or table_keyword_fields:
                            print(f"\n   📊 Smart routing: {len(table_keyword_fields)} table, {len(table_column_fields)} column, {len(table_row_fields)} row, {len(normal_fields)} normal")
                        
                        # ─── Process table fields ───
                        all_table_fields = table_keyword_fields + table_column_fields + table_row_fields
                        if all_table_fields:
                            for field_name in all_table_fields:
                                # Find which page has the table
                                table_pages = [idx for idx, s in page_table_scans.items() if s.get("has_table")]
                                if not table_pages:
                                    table_pages = list(range(num_pages))
                                
                                for page_idx in table_pages:
                                    page_img = enhanced_images[page_idx]
                                    try:
                                        table_data, table_type = extractor.extract_table(page_img, field_name)
                                        if table_data is not None and table_type is not None:
                                            all_results[field_name] = json.dumps(table_data, ensure_ascii=False)
                                            sig = {"source": "table", "flags": [], "page": page_idx + 1, "type": table_type}
                                            if table_type == "table" and isinstance(table_data, list):
                                                sig["rows"] = len(table_data)
                                                sig["cols"] = len(table_data[0]) if table_data else 0
                                            elif table_type == "column" and isinstance(table_data, list):
                                                sig["count"] = len(table_data)
                                            all_signals[field_name] = sig
                                            print(f"   📊 ✅ '{field_name}' → {table_type} (page {page_idx + 1})")
                                            break
                                    except Exception as e:
                                        print(f"   📊 Table attempt failed for '{field_name}' page {page_idx+1}: {e}")
                                if field_name not in all_results:
                                    all_results[field_name] = ""
                                    all_signals[field_name] = {"source": "table", "flags": ["not_found"], "page": 1}
                                    print(f"   📊 '{field_name}' — no table found")
                        
                        # ─── Normal field extraction ───
                        field_labels_for_extract = normal_fields
                        
                        for page_idx in range(num_pages):
                            page_num = page_idx + 1
                            page_img = enhanced_images[page_idx]
                            raw_img = raw_images[page_idx] if page_idx < len(raw_images) else page_img
                            
                            # Skip pages for fields already found (optimization)
                            remaining_fields = [f for f in field_labels_for_extract if not all_results.get(f, "").strip()]
                            if not remaining_fields and page_num > 1:
                                print(f"   ✅ All fields found — skipping page {page_num}/{num_pages}")
                                continue
                            
                            # On first page, extract all fields. On subsequent pages, only missing ones.
                            extract_fields_for_page = field_labels_for_extract if page_num == 1 else remaining_fields
                            if not extract_fields_for_page:
                                continue
                            
                            print(f"\n   📄 Page {page_num}/{num_pages}: extracting {len(extract_fields_for_page)} fields...")
                            page_results = extractor.extract(
                                page_img, [], [], extract_fields_for_page,
                                ocr_source="none",
                                voting_rounds=voting_rounds,
                            )
                            page_signals = extractor.get_last_signals()
                            
                            # Merge: first non-empty value wins per field
                            for field in extract_fields_for_page:
                                value = page_results.get(field, "")
                                if value.strip() and not all_results.get(field, "").strip():
                                    all_results[field] = value
                                    all_signals[field] = {
                                        **page_signals.get(field, {"source": "batch", "flags": []}),
                                        "page": page_num,
                                    }
                                elif field not in all_results:
                                    all_results[field] = value
                                    all_signals[field] = {
                                        **page_signals.get(field, {"source": "batch", "flags": ["empty_value"]}),
                                        "page": page_num,
                                    }
                    
                    # ─── Auto-detect checkbox fields (run on raw images) ───
                    model_signals = all_signals
                    results = all_results
                    cb_enabled = checkbox_enabled.lower() == "true"
                    if cb_enabled and model == "qwen" and _qwen2vl_available:
                        def _norm(s):
                            return ''.join(s.strip().lower().split())
                        
                        echo_fields = []
                        for field_name, value in results.items():
                            if isinstance(value, str):
                                if (field_name.strip().lower() == value.strip().lower() or
                                    _norm(field_name) == _norm(value)):
                                    echo_fields.append(field_name)
                        
                        if echo_fields:
                            print(f"   ☑️ Detected {len(echo_fields)} checkbox-style fields")
                            # Run checkbox detection on all raw pages
                            all_checkboxes = []
                            for page_idx, raw_img in enumerate(raw_images):
                                page_cbs = extractor.extract_checkboxes(raw_img, fields=None)
                                for cb in page_cbs:
                                    cb["_page"] = page_idx + 1
                                all_checkboxes.extend(page_cbs)
                            
                            cb_lookup = {}
                            cb_norm_lookup = {}
                            for cb in all_checkboxes:
                                cb_lookup[cb["label"].strip().lower()] = cb
                                cb_norm_lookup[_norm(cb["label"])] = cb
                            
                            for field_name in echo_fields:
                                fn_lower = field_name.strip().lower()
                                fn_norm = _norm(field_name)
                                matched = cb_lookup.get(fn_lower) or cb_norm_lookup.get(fn_norm)
                                
                                if not matched:
                                    for cb_label, cb_data in cb_lookup.items():
                                        if fn_lower in cb_label or cb_label in fn_lower:
                                            matched = cb_data
                                            break
                                
                                if not matched:
                                    try:
                                        # Try VQA fallback on the page where the field was found
                                        field_page = model_signals.get(field_name, {}).get("page", 1)
                                        fallback_img = raw_images[field_page - 1] if field_page <= len(raw_images) else raw_images[0]
                                        is_checked = extractor._extract_single_checkbox(fallback_img, field_name)
                                        matched = {"label": field_name, "checked": is_checked, "_page": field_page,
                                                   "signal": {"source": "checkbox_vqa", "flags": ["vqa_fallback"]}}
                                    except Exception:
                                        pass
                                
                                if matched:
                                    status_str = "Checked" if matched["checked"] else "Unchecked"
                                    results[field_name] = status_str
                                    model_signals[field_name] = {
                                        **matched.get("signal", {"source": "checkbox_batch", "flags": []}),
                                        "page": matched.get("_page", 1),
                                    }
                                else:
                                    results[field_name] = "Not Found"
                                    model_signals[field_name] = {"source": "checkbox_batch", "flags": ["not_found"], "page": 1}
                    
                    # ─── Table extraction fallback (for empty/echo fields) ───
                    table_candidates = []
                    for field_name, value in results.items():
                        val = value.strip() if isinstance(value, str) else ""
                        is_empty = not val
                        is_echo = val.lower() == field_name.strip().lower()
                        is_not_found = val.lower() in ("not found", "none", "")
                        # Heuristic: value looks like concatenated column data (3+ items separated by commas or newlines)
                        parts = [p.strip() for p in val.replace("\n", ",").split(",") if p.strip()]
                        is_multi_value = len(parts) >= 3
                        if is_empty or is_echo or is_not_found or is_multi_value:
                            table_candidates.append(field_name)
                    
                    if table_candidates:
                        print(f"\n   📊 Trying table extraction for {len(table_candidates)} field(s): {table_candidates}")
                        for field_name in table_candidates:
                            table_found = False
                            for page_idx in range(num_pages):
                                page_img = enhanced_images[page_idx]
                                try:
                                    table_data, table_type = extractor.extract_table(page_img, field_name)
                                    if table_data is not None and table_type is not None:
                                        # Store as JSON string
                                        results[field_name] = json.dumps(table_data, ensure_ascii=False)
                                        # Build signal info
                                        sig = {
                                            "source": "table",
                                            "flags": [],
                                            "page": page_idx + 1,
                                            "type": table_type,
                                        }
                                        if table_type == "table" and isinstance(table_data, list):
                                            sig["rows"] = len(table_data)
                                            sig["cols"] = len(table_data[0]) if table_data else 0
                                        elif table_type == "column" and isinstance(table_data, list):
                                            sig["count"] = len(table_data)
                                        model_signals[field_name] = sig
                                        table_found = True
                                        print(f"   📊 ✅ '{field_name}' → {table_type} (page {page_idx + 1})")
                                        break
                                except Exception as e:
                                    print(f"   📊 Table attempt failed for '{field_name}' page {page_idx+1}: {e}")
                            if not table_found:
                                print(f"   📊 '{field_name}' — no table found on any page")
                    
                    extraction_meta = extractor.get_last_meta()
                    
        except TimeoutError:
            print(f"⏱️ GPU busy — request timed out after {_GPU_TIMEOUT_SECONDS}s")
            raise HTTPException(
                status_code=503,
                detail=f"GPU busy — another extraction is in progress. Try again in a moment."
            )
        
        t_elapsed = time.time() - t_start
        _set_progress(9, 10, "Validating and scoring...")
        print(f"✅ Extraction complete in {t_elapsed:.1f}s ({num_pages} page(s))")

        # Save extraction context (first page image for training)
        save_extraction_context(
            image=enhanced_images[0],
            filename=file.filename,
            page_num=1,
            fields=field_labels,
            results=results,
            signals=model_signals,
            model_used=model,
            voting_rounds=voting_rounds,
        )

        # ─── Signal-based HITL routing ───
        hitl_manager = get_hitl_manager()
        flagged_fields = []
        validation_errors = {}
        
        meta = extraction_meta if model == "paddleocr" else extractor.get_last_meta()
        validation = meta.get("validation", {})
        
        for field_name, value in results.items():
            field_signal = model_signals.get(field_name, {"source": "unknown", "flags": []})
            v_result = validation.get(field_name, {})
            
            flags = list(field_signal.get("flags", []))
            reason_parts = []
            
            if v_result.get("is_valid") is False and v_result.get("error") != "Empty value":
                flags.append("validation_error")
                validation_errors[field_name] = v_result.get("error", "Unknown")
                reason_parts.append(f"Validation: {v_result.get('error', 'Unknown')}")
            
            if not value.strip():
                if "empty_value" not in flags:
                    flags.append("empty_value")
                reason_parts.append("Empty value")
            
            source = field_signal.get("source", "unknown")
            if "fallback_recovery" in flags:
                reason_parts.append("Recovered by per-field fallback (batch missed)")
            if "voting_disagreed" in flags:
                reason_parts.append(f"Voting disagreement: {field_signal.get('detail', '')}")
            if "vqa_fallback" in flags:
                reason_parts.append("Detected via single-item VQA fallback")
            if "not_found" in flags:
                reason_parts.append("No matching checkbox found")
            
            should_flag = bool(flags)
            
            if should_flag:
                reason = "; ".join(reason_parts) if reason_parts else f"Source: {source}"
                hitl_manager.add_item(
                    filename=file.filename,
                    field_name=field_name,
                    ai_value=value,
                    signal=source,
                    reason=reason,
                    page_num=field_signal.get("page", 1)
                )
                flagged_fields.append(field_name)
                print(f"⚠️ Auto-flagged '{field_name}' for review ({reason})")

        # Build normalized_values map from validation metadata
        normalized_values = {}
        for field_name, v_result in validation.items():
            norm = v_result.get("normalized", "")
            raw = v_result.get("raw", results.get(field_name, ""))
            if norm != raw:
                normalized_values[field_name] = norm

        # ─── Compute per-field confidence scores ───
        confidence_scores = {}
        for field_name, sig in model_signals.items():
            source = sig.get("source", "unknown")
            flags = sig.get("flags", [])
            
            # Base score by extraction source
            base_scores = {
                "batch": 0.95,
                "multipage": 0.90,
                "table": 0.90,
                "checkbox_vlm": 0.85,
                "checkbox_zoom": 0.80,
                "single_field": 0.75,
                "fallback_recovery": 0.60,
                "zoom": 0.50,
                "unknown": 0.50,
            }
            score = base_scores.get(source, 0.50)
            
            # Penalty for flags
            if "voting_disagreed" in flags:
                score -= 0.25
            if "empty_value" in flags:
                score -= 0.30
            if "vqa_fallback" in flags:
                score -= 0.15
            if "not_found" in flags:
                score = 0.0
            
            # Clamp 0-1
            confidence_scores[field_name] = round(max(0.0, min(1.0, score)), 2)
        
        response_data = {
            **results,
            "_meta": {
                "extraction_model": model,
                "time_seconds": round(t_elapsed, 1),
                "total_pages": num_pages,
                "confidence": confidence_scores,
                "flagged_fields": flagged_fields,
                "auto_flagged_count": len(flagged_fields),
                "validation_errors": validation_errors,
                "normalized_values": normalized_values,
            }
        }
        
        _set_progress(10, 10, "Done!")
        _clear_progress()
        return ExtractionResponse(
            success=True,
            data=response_data,
            message=f"Extracted {len(results)} fields from {num_pages} page(s) in {t_elapsed:.1f}s" + 
                    (f" ({len(flagged_fields)} flagged for review)" if flagged_fields else "")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auto-find-fields")
async def auto_find_fields(file: UploadFile = File(...)):
    """Auto-detect field labels in PDF using Qwen2-VL"""
    try:
        if not is_supported_file(file.filename):
            raise HTTPException(status_code=400, detail="Only PDF and image files (PNG, JPG, TIFF, BMP) are supported")
        
        pdf_bytes = await file.read()
        if len(pdf_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        if not _qwen2vl_available:
            raise HTTPException(status_code=503, detail="Qwen2-VL model not available")
        
        print(f"📄 Auto-detecting fields in: {file.filename}")
        try:
            enhanced_images, _ = get_or_process_file(pdf_bytes, file.filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        if not enhanced_images:
            raise HTTPException(status_code=400, detail="Could not process PDF")
        
        num_pages = len(enhanced_images)
        
        # Detect fields from ALL pages (union, deduplicated)
        try:
            async with asyncio.timeout(_GPU_TIMEOUT_SECONDS):
                async with _gpu_semaphore:
                    print(f"🔒 GPU semaphore acquired for auto-detect ({num_pages} pages)")
                    qwen = Qwen2VLExtractor()
                    all_fields = []
                    seen = set()
                    for page_idx, page_img in enumerate(enhanced_images):
                        print(f"   📄 Page {page_idx+1}/{num_pages}: detecting fields...")
                        page_fields = qwen.auto_detect_fields(page_img)
                        for f in page_fields:
                            f_lower = f.strip().lower()
                            if f_lower not in seen:
                                seen.add(f_lower)
                                all_fields.append(f)
        except TimeoutError:
            print(f"⏱️ GPU busy — auto-detect timed out after {_GPU_TIMEOUT_SECONDS}s")
            raise HTTPException(
                status_code=503,
                detail=f"GPU busy — another operation is in progress. Try again in a moment."
            )
        
        print(f"✅ Found {len(all_fields)} unique fields across {num_pages} page(s): {all_fields}")
        
        field_objects = [
            {"key": field, "question": f"What is the {field}?"}
            for field in all_fields
        ]
        
        return {"success": True, "fields": field_objects, "count": len(field_objects), "total_pages": num_pages}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error during auto-detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect-checkboxes")
async def detect_checkboxes(file: UploadFile = File(...)):
    """Auto-detect all physical checkboxes in a PDF document"""
    try:
        if not is_supported_file(file.filename):
            raise HTTPException(status_code=400, detail="Only PDF and image files (PNG, JPG, TIFF, BMP) are supported")
        
        pdf_bytes = await file.read()
        if len(pdf_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        if not _qwen2vl_available:
            raise HTTPException(status_code=503, detail="Qwen2-VL model not available")
        
        print(f"☑️ Detecting checkboxes in: {file.filename}")
        # IMPORTANT: Use raw PDF images WITHOUT enhancement.
        # The image enhancer binarizes and removes borders/lines, which
        # destroys checkbox marks (✓, ✗, filled squares).
        from utils.pdf_processor import pdf_to_images
        raw_images = pdf_to_images(pdf_bytes)
        
        if not raw_images:
            raise HTTPException(status_code=400, detail="Could not process PDF")
        
        if len(raw_images) > MAX_PDF_PAGES:
            raise HTTPException(status_code=400, detail=f"PDF has {len(raw_images)} pages (max {MAX_PDF_PAGES})")
        
        num_pages = len(raw_images)
        print(f"   📐 {num_pages} page(s) to scan for checkboxes")
        
        import time
        t_start = time.time()
        
        try:
            async with asyncio.timeout(_GPU_TIMEOUT_SECONDS):
                async with _gpu_semaphore:
                    print(f"🔒 GPU semaphore acquired for checkbox detection ({num_pages} pages)")
                    qwen = Qwen2VLExtractor()
                    all_checkboxes = []
                    for page_idx, page_img in enumerate(raw_images):
                        print(f"   📄 Page {page_idx+1}/{num_pages}: detecting checkboxes...")
                        page_cbs = qwen.extract_checkboxes(page_img, fields=None)
                        for cb in page_cbs:
                            cb["page"] = page_idx + 1
                        all_checkboxes.extend(page_cbs)
        except TimeoutError:
            print(f"⏱️ GPU busy — checkbox detection timed out after {_GPU_TIMEOUT_SECONDS}s")
            raise HTTPException(
                status_code=503,
                detail=f"GPU busy — another operation is in progress. Try again in a moment."
            )
        
        t_elapsed = time.time() - t_start
        
        print(f"✅ Found {len(all_checkboxes)} checkboxes across {num_pages} page(s) in {t_elapsed:.1f}s")
        
        return {
            "success": True,
            "checkboxes": all_checkboxes,
            "count": len(all_checkboxes),
            "total_pages": num_pages,
            "time_seconds": round(t_elapsed, 1),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error during checkbox detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Find All Tables ───

@app.post("/api/find-tables")
async def find_tables(file: UploadFile = File(...)):
    """Scan all pages and detect data tables (columns, rows) using VLM pre-scan."""
    try:
        if not is_supported_file(file.filename):
            raise HTTPException(status_code=400, detail="Only PDF and image files are supported")
        
        pdf_bytes = await file.read()
        if len(pdf_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        if not _qwen2vl_available:
            raise HTTPException(status_code=503, detail="Qwen2-VL model not available")
        
        print(f"📊 Finding tables in: {file.filename}")
        
        try:
            enhanced_images, _ = get_or_process_file(pdf_bytes, file.filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        if not enhanced_images:
            raise HTTPException(status_code=400, detail="Could not process file")
        
        num_pages = len(enhanced_images)
        
        import time
        t_start = time.time()
        
        try:
            async with asyncio.timeout(_GPU_TIMEOUT_SECONDS):
                async with _gpu_semaphore:
                    print(f"🔒 GPU semaphore acquired for table scan ({num_pages} pages)")
                    qwen = Qwen2VLExtractor()
                    all_tables = []
                    
                    for page_idx, page_img in enumerate(enhanced_images):
                        page_num = page_idx + 1
                        print(f"   📄 Page {page_num}/{num_pages}: scanning for tables...")
                        scan = qwen.scan_table_structure(page_img)
                        
                        if scan.get("has_table"):
                            columns = scan.get("columns", [])
                            row_prefix = scan.get("row_index", "")
                            print(f"   ✅ Page {page_num}: found table — {len(columns)} columns, row prefix: '{row_prefix}'")
                            
                            # Also extract the actual table rows for display
                            rows_json = None
                            try:
                                rows_data, _ = qwen._extract_table_or_column(page_img, "table")
                                if rows_data and rows_data not in ("NOT_A_TABLE", "NOT_FOUND", ""):
                                    rows_json = rows_data
                                    print(f"   📊 Extracted table rows for page {page_num}")
                            except Exception as ex:
                                print(f"   ⚠️ Could not extract rows: {ex}")
                            
                            all_tables.append({
                                "page": page_num,
                                "columns": columns,
                                "row_index_prefix": row_prefix,
                                "column_count": len(columns),
                                "rows_json": rows_json,  # The actual table data as JSON string
                            })
                        else:
                            print(f"   — Page {page_num}: no data table")
                            
        except TimeoutError:
            raise HTTPException(status_code=503, detail="GPU busy — try again in a moment.")
        
        t_elapsed = time.time() - t_start
        print(f"✅ Found {len(all_tables)} table(s) across {num_pages} page(s) in {t_elapsed:.1f}s")
        
        return {
            "success": True,
            "tables": all_tables,
            "count": len(all_tables),
            "total_pages": num_pages,
            "time_seconds": round(t_elapsed, 1),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error during table scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Document Q&A ───

@app.post("/api/ask")
async def ask_question(
    file: UploadFile = File(...),
    question: str = Form(...),
    model: str = Form("qwen"),
    history: str = Form("[]"),
):
    """
    Ask a natural language question about a PDF document.
    
    Enhanced: sends all pages (up to 5) for cross-page context.
    Accepts conversation history for follow-up questions.
    
    Args:
        history: JSON string of [{"role": "user"|"assistant", "content": "..."}]
    """
    try:
        if not is_supported_file(file.filename):
            raise HTTPException(status_code=400, detail="Only PDF and image files (PNG, JPG, TIFF, BMP) are supported")
        
        pdf_bytes = await file.read()
        if len(pdf_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        if not question or not question.strip():
            raise HTTPException(status_code=400, detail="Question is required")
        
        # Validate model
        if not _qwen2vl_available:
            raise HTTPException(status_code=503, detail="Qwen2-VL not available")
        
        # Parse conversation history
        try:
            conv_history = json.loads(history) if history else []
        except (json.JSONDecodeError, TypeError):
            conv_history = []
        
        # Process PDF → images (cached, multi-page)
        try:
            enhanced_images, _ = get_or_process_file(pdf_bytes, file.filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not enhanced_images:
            raise HTTPException(status_code=400, detail="Could not process PDF")
        
        num_pages = len(enhanced_images)
        t_start = time.time()
        
        # Send all pages (up to 5) in a single multi-page VLM call
        try:
            async with asyncio.timeout(_GPU_TIMEOUT_SECONDS):
                async with _gpu_semaphore:
                    qwen = Qwen2VLExtractor()
                    answer = qwen.ask_question_multipage(
                        enhanced_images, question.strip(), history=conv_history
                    )
                    
        except TimeoutError:
            raise HTTPException(
                status_code=503,
                detail="GPU busy — try again in a moment."
            )
        
        t_elapsed = time.time() - t_start
        
        # Handle sentinel responses from the model
        answer_type = "answer"
        if not answer:
            answer = "No answer found."
            answer_type = "system"
        elif "NOT_DOCUMENT_RELATED" in answer.upper():
            answer = "This question doesn't seem related to the document. Try asking about specific fields, values, or content in the PDF."
            answer_type = "system"
        elif "NOT_FOUND_IN_DOCUMENT" in answer.upper():
            answer = "This information was not found in the document."
            answer_type = "system"
        
        return {
            "success": True,
            "answer": answer,
            "answer_type": answer_type,
            "total_pages": num_pages,
            "time_seconds": round(t_elapsed, 1),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Q&A error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ─── CSV Export ───

@app.post("/api/export-csv")
async def export_csv(request: Request):
    """
    Convert extraction results to CSV format.
    
    Accepts JSON body: {"data": {"field1": "value1", ...}, "filename": "doc.pdf"}
    Returns CSV file download.
    """
    import csv
    import io
    
    try:
        body = await request.json()
        data = body.get("data", {})
        filename = body.get("filename", "extraction").replace(".pdf", "").replace(".png", "")
        
        if not data:
            raise HTTPException(status_code=400, detail="No data to export")
        
        # Filter out _meta
        export_data = {k: v for k, v in data.items() if k != "_meta"}
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header row
        writer.writerow(["Field", "Value"])
        
        # Data rows
        for field, value in export_data.items():
            writer.writerow([field, value])
        
        csv_content = output.getvalue()
        
        from fastapi.responses import Response
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}_extraction.csv"'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Document Classification ───

@app.post("/api/classify")
async def classify_document(
    file: UploadFile = File(...),
):
    """
    Classify document type and suggest relevant fields using VLM.
    
    No hardcoded templates — the model dynamically analyzes the document
    and suggests what fields to extract.
    
    Returns:
        {doc_type: str, suggested_fields: [str], confidence: float}
    """
    try:
        if not is_supported_file(file.filename):
            raise HTTPException(status_code=400, detail="Only PDF and image files supported")
        
        pdf_bytes = await file.read()
        if len(pdf_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        if not _qwen2vl_available:
            raise HTTPException(status_code=503, detail="Qwen2-VL not available")
        
        try:
            enhanced_images, _ = get_or_process_file(pdf_bytes, file.filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not enhanced_images:
            raise HTTPException(status_code=400, detail="Could not process document")
        
        t_start = time.time()
        
        try:
            async with asyncio.timeout(_GPU_TIMEOUT_SECONDS):
                async with _gpu_semaphore:
                    qwen = Qwen2VLExtractor()
                    
                    # Use first page for classification
                    system_prompt = (
                        "You are a document classification expert. "
                        "Analyze the document image and determine: "
                        "1. What type of document it is (e.g., medical form, invoice, lab report, insurance claim, tax form, etc.) "
                        "2. A list of 5-15 specific field names that should be extracted from this document. "
                        "Return ONLY valid JSON."
                    )
                    user_prompt = (
                        "Classify this document and suggest fields to extract. "
                        'Return JSON: {"doc_type": "Document Type", "suggested_fields": ["Field 1", "Field 2", ...]}'
                    )
                    
                    raw = qwen._multipage_extract(
                        enhanced_images[:1], system_prompt, user_prompt, max_tokens=512
                    )
                    
        except TimeoutError:
            raise HTTPException(status_code=503, detail="GPU busy — try again")
        
        t_elapsed = time.time() - t_start
        
        # Parse classification result
        try:
            repaired = Qwen2VLExtractor._repair_json(raw)
            result = json.loads(repaired)
            doc_type = result.get("doc_type", "Unknown")
            fields = result.get("suggested_fields", [])
        except (json.JSONDecodeError, ValueError):
            doc_type = "Unknown"
            fields = []
        
        return {
            "success": True,
            "doc_type": doc_type,
            "suggested_fields": fields,
            "time_seconds": round(t_elapsed, 1),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/re-extract")
async def re_extract_field(
    file: UploadFile = File(...),
    field_name: str = Form(...),
    model: str = Form("qwen"),
):
    """
    Re-extract a single field from a PDF.
    Used for the 'resend' button on result cards.
    """
    try:
        if not is_supported_file(file.filename):
            raise HTTPException(status_code=400, detail="Only PDF and image files (PNG, JPG, TIFF, BMP) are supported")
        
        pdf_bytes = await file.read()
        if len(pdf_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        if not field_name or not field_name.strip():
            raise HTTPException(status_code=400, detail="Field name is required")
        
        # Validate model
        if not _qwen2vl_available:
            raise HTTPException(status_code=503, detail="Qwen2-VL not available")
        
        # Process PDF → images (cached, multi-page)
        try:
            enhanced_images, _ = get_or_process_file(pdf_bytes, file.filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not enhanced_images:
            raise HTTPException(status_code=400, detail="Could not process PDF")
        field = field_name.strip()
        t_start = time.time()
        
        # Try each page until we find a non-empty value
        try:
            async with asyncio.timeout(_GPU_TIMEOUT_SECONDS):
                async with _gpu_semaphore:
                    qwen = Qwen2VLExtractor()
                    value = ""
                    found_page = 1
                    for page_idx, page_img in enumerate(enhanced_images):
                        page_value = qwen._extract_single_field(page_img, field)
                        if page_value and page_value.strip():
                            value = page_value
                            found_page = page_idx + 1
                            break
                        
        except TimeoutError:
            raise HTTPException(
                status_code=503,
                detail="GPU busy — try again in a moment."
            )
        
        t_elapsed = time.time() - t_start
        
        return {
            "success": True,
            "field": field,
            "value": value if value else "",
            "signal": "re-extract",
            "page": found_page,
            "time_seconds": round(t_elapsed, 1),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Re-extract error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── HITL Review Endpoints ───


@app.get("/api/reviews")
async def get_reviews():
    manager = get_hitl_manager()
    return {"items": manager.queue, "stats": manager.get_stats()}

@app.post("/api/reviews/{item_id}/resolve")
async def resolve_review(item_id: str, decision: ReviewDecision):
    """Resolve a review item and collect training data"""
    manager = get_hitl_manager()
    
    item = None
    for i in manager.queue:
        if i["id"] == item_id:
            item = i
            break
    
    success = manager.resolve_item(item_id, decision.action, decision.corrected_value)
    if not success:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # ─── Collect training data (Qwen2.5-VL conversation format) ───
    if item and decision.action in ("correct", "approve"):
        try:
            collector = get_training_collector()
            context = get_extraction_context()

            if context["image"] is not None and context["results"]:
                # Merge correction into results = ground truth
                ground_truth = dict(context["results"])
                corrections = {}

                if decision.action == "correct" and decision.corrected_value:
                    field = item["field_name"]
                    corrections[field] = {
                        "original": ground_truth.get(field, ""),
                        "corrected": decision.corrected_value,
                    }
                    ground_truth[field] = decision.corrected_value

                sample_id = collector.save_sample(
                    image=context["image"],
                    filename=context["filename"],
                    page_num=context["page_num"],
                    fields_requested=context["fields_requested"],
                    extraction_results=ground_truth,
                    corrections=corrections,
                    signals=context.get("signals", {}),
                    model_used=context["model_used"],
                    voting_rounds=context["voting_rounds"],
                )
                if sample_id:
                    action_label = "correction" if corrections else "approval"
                    print(f"📚 Training sample saved ({action_label}): {sample_id}")
        except Exception as e:
            print(f"⚠️ Failed to save training sample: {e}")
    
    return {"success": True, "stats": manager.get_stats()}

@app.post("/api/flag")
async def flag_for_review(request: FlagRequest):
    manager = get_hitl_manager()
    item_id = manager.add_item(
        filename=request.filename,
        field_name=request.field_name,
        ai_value=request.ai_value,
        signal=request.signal,
        reason=request.reason,
        context_image=None
    )
    return {"success": True, "item_id": item_id}

@app.delete("/api/reviews/clear")
async def clear_queue():
    manager = get_hitl_manager()
    manager.queue = []
    manager._save_queue()
    return {"success": True, "message": "Queue cleared"}

@app.delete("/api/reviews/clear-resolved")
async def clear_resolved():
    manager = get_hitl_manager()
    manager.queue = [item for item in manager.queue if item["status"] == "pending"]
    manager._save_queue()
    return {"success": True, "stats": manager.get_stats()}


# ─── Training Data Endpoints ───

@app.get("/api/training/stats")
@app.get("/api/training/status")
async def get_training_stats():
    """Get training data statistics and readiness info."""
    collector = get_training_collector()
    return collector.get_stats()

@app.get("/api/training/samples")
async def get_training_samples():
    """Get all training samples."""
    collector = get_training_collector()
    samples = collector.get_samples()
    # Return lightweight version (no image data)
    return [
        {
            "id": s.get("id", ""),
            "timestamp": s.get("timestamp", ""),
            "source_pdf": s.get("source_pdf", ""),
            "page_num": s.get("page_num", 1),
            "fields_requested": s.get("fields_requested", []),
            "corrections": s.get("corrections", {}),
            "is_corrected": s.get("is_corrected", False),
        }
        for s in samples
    ]

@app.post("/api/training/export")
async def export_training_data():
    """Export training data to Qwen2.5-VL format for fine-tuning."""
    collector = get_training_collector()
    output_path = collector.export_for_training()
    if output_path:
        return {"success": True, "path": output_path, "message": "Training data exported"}
    return {"success": False, "message": "No training data to export"}

@app.delete("/api/training/clear")
async def clear_training_data():
    """Clear all training data (use with caution)."""
    collector = get_training_collector()
    collector.clear()
    return {"success": True, "message": "Training data cleared"}


# ─── Entry Point ───
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting PDF Data Extractor API...")
    print(f"📍 Server: http://localhost:8000")
    print(f"📖 API docs: http://localhost:8000/docs")
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
