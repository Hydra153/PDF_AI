"""
FastAPI Server for PDF Data Extraction
Provides REST API endpoints for document understanding and field extraction

Extraction engines:
  - PaddleOCR-VL 1.5 (0.9B params, document parsing + markdown extraction)
  - Qwen2.5-VL-3B (Vision-Language Model, OCR-free)

VRAM Management:
  On a 4GB GPU, only one model can be loaded at a time.
  When switching models via the 'model' parameter, the previous model
  is unloaded from GPU before the new one loads.
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

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import json
from pydantic import BaseModel

from config import CORS_ORIGINS, MAX_FILE_SIZE, ALLOWED_EXTENSIONS, QWEN2VL_MODEL, QWEN2VL_DISPLAY_NAME
from utils.pdf_processor import process_pdf
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

# ─── PaddleOCR-VL Model ───
_paddleocr_available = False
try:
    from models.paddleocr_extractor import PaddleOCRExtractor, unload_paddleocr_pipeline
    _paddleocr_available = True
    print(f"✅ PaddleOCR-VL 1.5 available")
except ImportError as e:
    print(f"⚠️ PaddleOCR-VL 1.5 not available: {e}")

if not _qwen2vl_available and not _paddleocr_available:
    print("⚠️ WARNING: No extraction model available!")

# ─── VRAM Model Swap Tracking ───
# On 4GB GPU, only one model fits in VRAM at a time.
# Track which model is currently loaded to swap when needed.
_active_model = None  # "qwen" or "paddleocr" or None


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
        "status": "healthy" if (_qwen2vl_available or _paddleocr_available) else "degraded",
        "models": {
            "qwen": {
                "name": QWEN2VL_DISPLAY_NAME,
                "available": _qwen2vl_available,
            },
            "paddleocr": {
                "name": "PaddleOCR-VL 1.5",
                "available": _paddleocr_available,
            },
        },
        "active_model": _active_model,
        # Backward compat
        "model": QWEN2VL_DISPLAY_NAME,
        "model_id": QWEN2VL_MODEL,
        "model_available": _qwen2vl_available,
    }


@app.post("/api/extract", response_model=ExtractionResponse)
async def extract_fields(
    file: UploadFile = File(...),
    fields: str = Form(...),
    model: str = Form("paddleocr"),  # "paddleocr" (default) or "qwen"
    voting_rounds: int = Form(1),    # 1 = normal, 3 = accuracy boost (majority voting)
    checkbox_enabled: str = Form("false"),  # "true" = run checkbox auto-detection
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
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
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
        
        # Validate model selection
        if model == "paddleocr" and not _paddleocr_available:
            raise HTTPException(status_code=503, detail="PaddleOCR-VL 1.5 not available")
        if model == "qwen" and not _qwen2vl_available:
            raise HTTPException(status_code=503, detail="Qwen2-VL model not available")
        if model not in ("qwen", "paddleocr"):
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}. Use 'qwen' or 'paddleocr'")
        
        # Process PDF → Image
        t_start = time.time()
        print(f"📄 Processing PDF: {file.filename} (model: {model})")
        images, _ = process_pdf(pdf_bytes)
        
        # Also keep raw images for checkbox fallback
        from utils.pdf_processor import pdf_to_images
        raw_images = pdf_to_images(pdf_bytes)
        
        if not images:
            raise HTTPException(status_code=400, detail="Could not process PDF")
        
        image = images[0]  # TODO: Multi-page support
        raw_image = raw_images[0] if raw_images else image
        field_labels = [f['key'] for f in fields_list]
        print(f"🤖 Extracting {len(field_labels)} fields with {model}...")
        
        # ─── GPU Inference (serialized by semaphore) ───
        try:
            async with asyncio.timeout(_GPU_TIMEOUT_SECONDS):
                async with _gpu_semaphore:
                    global _active_model
                    print(f"🔒 GPU semaphore acquired for extraction")
                    
                    # ─── VRAM Swap: unload previous model if switching ───
                    if _active_model and _active_model != model:
                        print(f"🔄 Swapping model: {_active_model} → {model}")
                        if _active_model == "qwen" and _paddleocr_available:
                            unload_qwen2vl_model()
                        elif _active_model == "paddleocr" and _qwen2vl_available:
                            unload_paddleocr_pipeline()
                    
                    # ─── Run selected extractor ───
                    if model == "paddleocr":
                        extractor = PaddleOCRExtractor()
                        results = extractor.extract(image, [], [], field_labels)
                        model_signals = extractor.get_last_signals()
                        extraction_meta = extractor.get_last_meta()
                    else:
                        extractor = Qwen2VLExtractor()
                        results = extractor.extract(
                            image, [], [], field_labels,
                            ocr_source="none",
                            voting_rounds=voting_rounds,
                        )
                        model_signals = extractor.get_last_signals()
                        extraction_meta = extractor.get_last_meta()
                    
                    # ─── Auto-detect checkbox fields (only when checkbox mode enabled) ───
                    cb_enabled = checkbox_enabled.lower() == "true"
                    if cb_enabled and model == "qwen" and _qwen2vl_available:
                        # Normalize helper: strip spaces/punctuation for comparison
                        def _norm(s):
                            return ''.join(s.strip().lower().split())  # "Blue Berries" → "blueberries"
                        
                        echo_fields = []
                        for field_name, value in results.items():
                            if isinstance(value, str):
                                # Match: exact OR normalized (e.g. "Blue Berries" == "Blueberries")
                                if (field_name.strip().lower() == value.strip().lower() or
                                    _norm(field_name) == _norm(value)):
                                    echo_fields.append(field_name)
                        
                        if echo_fields:
                            print(f"   ☑️ Detected {len(echo_fields)} checkbox-style fields (value ≈ field name)")
                            print(f"   🔄 Running batch checkbox detection on raw image...")
                            qwen = Qwen2VLExtractor()
                            all_checkboxes = qwen.extract_checkboxes(raw_image, fields=None)
                            
                            # Build lookup: lowercase label → checkbox result + normalized lookup
                            cb_lookup = {}
                            cb_norm_lookup = {}
                            for cb in all_checkboxes:
                                cb_lookup[cb["label"].strip().lower()] = cb
                                cb_norm_lookup[_norm(cb["label"])] = cb
                            
                            # Match each echo field to detected checkboxes
                            for field_name in echo_fields:
                                fn_lower = field_name.strip().lower()
                                fn_norm = _norm(field_name)
                                matched = cb_lookup.get(fn_lower)
                                
                                # Try normalized match (Blue Berries → blueberries)
                                if not matched:
                                    matched = cb_norm_lookup.get(fn_norm)
                                
                                # Try fuzzy substring match
                                if not matched:
                                    for cb_label, cb_data in cb_lookup.items():
                                        if fn_lower in cb_label or cb_label in fn_lower:
                                            matched = cb_data
                                            break
                                
                                # ─── Single-item VQA fallback for missed fields (e.g. Turkey) ───
                                if not matched:
                                    print(f"      🔍 '{field_name}' not in batch — trying single-item VQA...")
                                    try:
                                        is_checked = qwen._extract_single_checkbox(raw_image, field_name)
                                        matched = {
                                            "label": field_name,
                                            "checked": is_checked,
                                            "signal": {
                                                "source": "checkbox_vqa",
                                                "flags": ["vqa_fallback"],
                                            },
                                        }
                                        print(f"      {'☑' if is_checked else '☐'} Single-item: '{field_name}' → {'Checked' if is_checked else 'Unchecked'}")
                                    except Exception as e:
                                        print(f"      ⚠️ Single-item fallback failed for '{field_name}': {e}")
                                
                                if matched:
                                    status_str = "Checked" if matched["checked"] else "Unchecked"
                                    results[field_name] = status_str
                                    model_signals[field_name] = matched.get("signal", {"source": "checkbox_batch", "flags": []})
                                    print(f"      {'☑' if matched['checked'] else '☐'} '{field_name}' → {status_str} [{model_signals[field_name].get('source', 'batch')}]")
                                else:
                                    results[field_name] = "Not Found"
                                    model_signals[field_name] = {"source": "checkbox_batch", "flags": ["not_found"]}
                                    print(f"      ❓ '{field_name}' → No matching checkbox found")
                    
                    _active_model = model
                    
        except TimeoutError:
            print(f"⏱️ GPU busy — request timed out after {_GPU_TIMEOUT_SECONDS}s")
            raise HTTPException(
                status_code=503,
                detail=f"GPU busy — another extraction is in progress. Try again in a moment."
            )
        
        t_elapsed = time.time() - t_start
        print(f"✅ Extraction complete in {t_elapsed:.1f}s")

        # Save complete extraction context for training data collection
        save_extraction_context(
            image=image,
            filename=file.filename,
            page_num=1,
            fields=field_labels,
            results=results,
            signals=model_signals,
            model_used=model,
            voting_rounds=voting_rounds,
        )

        # ─── Signal-based HITL routing ───
        # Any field with flags gets routed for human review
        hitl_manager = get_hitl_manager()
        flagged_fields = []
        validation_errors = {}
        
        # Get validation metadata from extractor
        meta = extraction_meta if model == "paddleocr" else extractor.get_last_meta()
        validation = meta.get("validation", {})
        
        for field_name, value in results.items():
            field_signal = model_signals.get(field_name, {"source": "unknown", "flags": []})
            v_result = validation.get(field_name, {})
            
            # Collect flags from both extraction signals and validation
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
            
            # Add source-specific reasons
            source = field_signal.get("source", "unknown")
            if "fallback_recovery" in flags:
                reason_parts.append("Recovered by per-field fallback (batch missed)")
            if "voting_disagreed" in flags:
                reason_parts.append(f"Voting disagreement: {field_signal.get('detail', '')}")
            if "vqa_fallback" in flags:
                reason_parts.append("Detected via single-item VQA fallback")
            if "not_found" in flags:
                reason_parts.append("No matching checkbox found")
            
            # Flag if any flags exist (excluding empty_value for non-required fields)
            should_flag = bool(flags)
            
            if should_flag:
                reason = "; ".join(reason_parts) if reason_parts else f"Source: {source}"
                hitl_manager.add_item(
                    filename=file.filename,
                    field_name=field_name,
                    ai_value=value,
                    signal=source,
                    reason=reason,
                    page_num=1
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

        response_data = {
            **results,
            "_meta": {
                "extraction_model": model,
                "time_seconds": round(t_elapsed, 1),
                "signals": {k: {"source": v.get("source", "unknown"), "flags": v.get("flags", [])} for k, v in model_signals.items()},
                "flagged_fields": flagged_fields,
                "auto_flagged_count": len(flagged_fields),
                "validation_errors": validation_errors,
                "normalized_values": normalized_values,
            }
        }
        
        return ExtractionResponse(
            success=True,
            data=response_data,
            message=f"Extracted {len(results)} fields in {t_elapsed:.1f}s" + 
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
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        pdf_bytes = await file.read()
        if len(pdf_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        if not _qwen2vl_available:
            raise HTTPException(status_code=503, detail="Qwen2-VL model not available")
        
        print(f"📄 Auto-detecting fields in: {file.filename}")
        images, _ = process_pdf(pdf_bytes)
        
        if not images:
            raise HTTPException(status_code=400, detail="Could not process PDF")
        
        image = images[0]
        
        # Use Qwen2-VL to detect field labels (serialized by semaphore)
        try:
            async with asyncio.timeout(_GPU_TIMEOUT_SECONDS):
                async with _gpu_semaphore:
                    print(f"🔒 GPU semaphore acquired for auto-detect")
                    qwen = Qwen2VLExtractor()
                    detected_fields = qwen.auto_detect_fields(image)
        except TimeoutError:
            print(f"⏱️ GPU busy — auto-detect timed out after {_GPU_TIMEOUT_SECONDS}s")
            raise HTTPException(
                status_code=503,
                detail=f"GPU busy — another operation is in progress. Try again in a moment."
            )
        
        print(f"✅ Found {len(detected_fields)} fields: {detected_fields}")
        
        field_objects = [
            {"key": field, "question": f"What is the {field}?"}
            for field in detected_fields
        ]
        
        return {"success": True, "fields": field_objects, "count": len(field_objects)}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error during auto-detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect-checkboxes")
async def detect_checkboxes(file: UploadFile = File(...)):
    """Auto-detect all physical checkboxes in a PDF document"""
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
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
        images = pdf_to_images(pdf_bytes)
        
        if not images:
            raise HTTPException(status_code=400, detail="Could not process PDF")
        
        image = images[0]
        print(f"   📐 Checkbox image: {image.size[0]}x{image.size[1]} ({image.mode})")
        
        import time
        t_start = time.time()
        
        try:
            async with asyncio.timeout(_GPU_TIMEOUT_SECONDS):
                async with _gpu_semaphore:
                    print(f"🔒 GPU semaphore acquired for checkbox detection")
                    qwen = Qwen2VLExtractor()
                    checkboxes = qwen.extract_checkboxes(image, fields=None)
        except TimeoutError:
            print(f"⏱️ GPU busy — checkbox detection timed out after {_GPU_TIMEOUT_SECONDS}s")
            raise HTTPException(
                status_code=503,
                detail=f"GPU busy — another operation is in progress. Try again in a moment."
            )
        
        t_elapsed = time.time() - t_start
        
        print(f"✅ Found {len(checkboxes)} checkboxes in {t_elapsed:.1f}s")
        
        return {
            "success": True,
            "checkboxes": checkboxes,
            "count": len(checkboxes),
            "time_seconds": round(t_elapsed, 1),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error during checkbox detection: {e}")
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
