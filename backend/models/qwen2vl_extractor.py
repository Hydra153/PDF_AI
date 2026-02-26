"""
Qwen VL Extractor — OCR-free document field extraction using Vision-Language Model

Pipeline:
    PDF → Image → Qwen VL → JSON Output (no OCR needed)

The model reads the document image directly and extracts all requested fields
in a single pass, returning structured JSON. This eliminates OCR error propagation
and handles ambiguous layouts better than OCR + DocQA approaches.

Auto-selects the best model based on available VRAM:
    ≥8GB → Qwen2.5-VL-7B-Instruct (95% DocVQA)
    ≥4GB → Qwen2.5-VL-3B-Instruct (92% DocVQA)
    <4GB → Qwen2-VL-2B-Instruct   (88% DocVQA)

Usage:
    from models.qwen2vl_extractor import Qwen2VLExtractor
    extractor = Qwen2VLExtractor()
    results = extractor.extract(image, fields=["Name", "DOB", "Address"])
"""

import torch
import json
import logging
import math
import re
from typing import Dict, List, Tuple, Optional
from PIL import Image

logger = logging.getLogger(__name__)

# Toggle: set to True to re-enable field validators (normalization + type checking)
ENABLE_VALIDATORS = True

# ─── Singleton Model Cache ───
_qwen2vl_model = None
_qwen2vl_processor = None


def get_qwen2vl_model():
    """Lazy-load Qwen VL model and processor (singleton). Model auto-selected from config."""
    global _qwen2vl_model, _qwen2vl_processor
    
    if _qwen2vl_model is not None:
        return _qwen2vl_model, _qwen2vl_processor
    
    try:
        from transformers import AutoProcessor
        from transformers import BitsAndBytesConfig
        from config import QWEN2VL_MODEL, QWEN2VL_DISPLAY_NAME, QWEN2VL_MIN_PIXELS, QWEN2VL_MAX_PIXELS
        
        model_name = QWEN2VL_MODEL
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Qwen2.5-VL uses a different class than Qwen2-VL
        if "Qwen2.5" in model_name:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
        else:
            from transformers import Qwen2VLForConditionalGeneration as ModelClass
        
        print(f"📥 Loading {QWEN2VL_DISPLAY_NAME} from {model_name}...")
        print(f"   Device: {device}")
        print(f"   Quantization: 4-bit NF4 (BitsAndBytes)")
        
        # 4-bit NF4 quantization — fits in ~1.5-2.0GB VRAM for 2B-3B models
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,  # nested quantization saves ~0.4GB
            llm_int8_enable_fp32_cpu_offload=True,
        )
        
        _qwen2vl_processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=QWEN2VL_MIN_PIXELS,
            max_pixels=QWEN2VL_MAX_PIXELS,
        )
        print(f"   📐 Processor pixel budget: {QWEN2VL_MIN_PIXELS:,} – {QWEN2VL_MAX_PIXELS:,} pixels")
        
        # Suppress the 1600+ line "Loading weights" progress bar from accelerate
        # In Docker (non-TTY), tqdm dumps every frame as a separate line.
        # We must monkey-patch tqdm itself since TQDM_DISABLE doesn't work here.
        import tqdm as _tqdm_mod
        import tqdm.auto as _tqdm_auto
        import logging as _logging
        _orig_tqdm = _tqdm_mod.tqdm
        _orig_auto_tqdm = _tqdm_auto.tqdm
        
        class _SilentTqdm(_orig_tqdm):
            def __init__(self, *args, **kwargs):
                kwargs['disable'] = True
                super().__init__(*args, **kwargs)
        
        _tqdm_mod.tqdm = _SilentTqdm
        _tqdm_auto.tqdm = _SilentTqdm
        _accel_logger = _logging.getLogger("accelerate")
        _prev_accel_level = _accel_logger.level
        _accel_logger.setLevel(_logging.ERROR)
        
        try:
            _qwen2vl_model = ModelClass.from_pretrained(
                model_name,
                quantization_config=quantization_config if device == "cuda" else None,
                device_map="auto",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
        finally:
            _tqdm_mod.tqdm = _orig_tqdm
            _tqdm_auto.tqdm = _orig_auto_tqdm
            _accel_logger.setLevel(_prev_accel_level)
        _qwen2vl_model.eval()
        
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
            print(f"   ✅ {QWEN2VL_DISPLAY_NAME} loaded ({vram_mb:.0f} MB VRAM)")
        else:
            print(f"   ✅ {QWEN2VL_DISPLAY_NAME} loaded (CPU mode)")
        
        # ─── Auto-load LoRA adapter if it exists ───
        try:
            from config import QWEN2VL_LORA_PATH
            from pathlib import Path
            lora_path = Path(QWEN2VL_LORA_PATH)
            if lora_path.exists() and (lora_path / "adapter_config.json").exists():
                from peft import PeftModel
                print(f"   🔧 Loading LoRA adapter from {lora_path}...")
                _qwen2vl_model = PeftModel.from_pretrained(
                    _qwen2vl_model, str(lora_path)
                )
                _qwen2vl_model.eval()
                # Load training metadata if available
                meta_path = lora_path / "training_meta.json"
                if meta_path.exists():
                    import json as _json
                    with open(meta_path) as _f:
                        meta = _json.load(_f)
                    print(f"   ✅ LoRA adapter loaded (trained on {meta.get('training_samples', '?')} samples, loss: {meta.get('train_loss', '?'):.4f})")
                else:
                    print(f"   ✅ LoRA adapter loaded")
        except ImportError:
            pass  # peft not installed — skip silently
        except Exception as e:
            print(f"   ⚠️ LoRA adapter found but failed to load: {e}")
        
        return _qwen2vl_model, _qwen2vl_processor
        
    except Exception as e:
        logger.error(f"Failed to load Qwen VL model: {e}")
        print(f"   ❌ Qwen VL loading failed: {e}")
        raise


def unload_qwen2vl_model():
    """Unload Qwen2-VL from GPU to free VRAM for other models."""
    global _qwen2vl_model, _qwen2vl_processor
    
    if _qwen2vl_model is not None:
        print("🔄 Unloading Qwen2-VL from GPU...")
        try:
            _qwen2vl_model.cpu()
            del _qwen2vl_model
            del _qwen2vl_processor
        except Exception:
            pass
        _qwen2vl_model = None
        _qwen2vl_processor = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
            print(f"   ✅ Qwen2-VL unloaded ({vram_mb:.0f} MB VRAM remaining)")
        else:
            print("   ✅ Qwen2-VL unloaded")


# ─── Hallucination Guard Constants ───
MAX_ANSWER_CHARS = 200
_HALLUCINATION_PATTERN = re.compile(
    r'\b(is|are|was|were|has|have|does|do|can|will|should|would|'
    r'less than|greater than|equal to|not found|cannot be|unable to|'
    r'the value of|it appears|based on|according to)\b',
    re.IGNORECASE
)

# Values the model returns as placeholders instead of real data
_PLACEHOLDER_VALUES = frozenset({'...', '…', 'N/A', 'n/a', 'none', 'None', 'null', 'NULL', '""', "''"})


class Qwen2VLExtractor:
    """Extract field values from documents using Qwen VL Vision-Language Model."""
    
    def __init__(self):
        self.model, self.processor = get_qwen2vl_model()
        self.device = next(self.model.parameters()).device
        self._last_signals = {}  # stores {source, flags} per field
        self._last_meta = {}  # stores signals + validation per field
    
    # ─── Smart Batching Helpers ─────────────────────────────────────────────
    
    def _field_similarity(self, f1: str, f2: str) -> float:
        """
        Compute Jaccard similarity between two field names using word overlap.
        
        Jaccard = |intersection| / |union|, which is more conservative than
        min-based scoring and avoids false positives on asymmetric names.
        
        Examples:
          "Patient Phone" vs "Client Phone"  → 1/3 = 0.33 (conflict at threshold 0.3)
          "Patient Name"  vs "DOB"           → 0/3 = 0.00 (no conflict)
          "Patient"       vs "Patient Name"  → 1/2 = 0.50 (conflict)
        """
        w1 = set(f1.lower().split())
        w2 = set(f2.lower().split())
        if not w1 or not w2:
            return 0.0
        intersection = w1 & w2
        union = w1 | w2
        return len(intersection) / len(union)
    
    def _smart_batch_fields(self, fields: List[str], batch_size: int = 5,
                            similarity_threshold: float = 0.3) -> List[List[str]]:
        """
        Create batches that avoid grouping semantically similar fields together.
        
        Algorithm (graph coloring from AI 3 reference):
          1. Compute pairwise word-overlap similarity between all fields
          2. Find "conflict pairs" — fields too similar to batch together
          3. Greedy graph coloring: assign fields to groups where no conflicts exist
          4. Round-robin pack groups into batches of ≤ batch_size
        
        Args:
            fields: List of field names
            batch_size: Max fields per batch (default 5)
            similarity_threshold: Fields with similarity > this are separated
                                  0.5 catches "Patient Phone"/"Client Phone" (share "Phone")
        
        Returns:
            List of batches, each a list of field names
        """
        if len(fields) <= batch_size:
            # Check if the small set has internal conflicts
            has_conflict = False
            for i in range(len(fields)):
                for j in range(i + 1, len(fields)):
                    if self._field_similarity(fields[i], fields[j]) > similarity_threshold:
                        has_conflict = True
                        break
                if has_conflict:
                    break
            if not has_conflict:
                return [fields]
        
        # Step 1: Find conflict pairs
        conflicts = {}
        for i in range(len(fields)):
            conflicts[i] = set()
        
        for i in range(len(fields)):
            for j in range(i + 1, len(fields)):
                sim = self._field_similarity(fields[i], fields[j])
                if sim > similarity_threshold:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
                    print(f"      🔗 Conflict: '{fields[i]}' ↔ '{fields[j]}' (sim: {sim:.2f})")
        
        # Step 2: Greedy graph coloring — assign each field to a group
        # where it has no conflicts with existing members
        field_to_group = {}
        num_groups = 0
        
        for field_idx in range(len(fields)):
            assigned = False
            for group_id in range(num_groups):
                # Check if field conflicts with anyone in this group
                has_conflict = any(
                    other_idx in conflicts[field_idx]
                    for other_idx, g in field_to_group.items()
                    if g == group_id
                )
                if not has_conflict:
                    field_to_group[field_idx] = group_id
                    assigned = True
                    break
            
            if not assigned:
                field_to_group[field_idx] = num_groups
                num_groups += 1
        
        # Step 3: Collect groups
        groups = {}
        for field_idx, group_id in field_to_group.items():
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(fields[field_idx])
        
        # Step 4: Round-robin pack groups into batches of ≤ batch_size
        group_lists = [list(g) for g in groups.values()]
        batches = []
        
        while any(group_lists):
            current_batch = []
            for group in group_lists:
                if group and len(current_batch) < batch_size:
                    current_batch.append(group.pop(0))
            if current_batch:
                batches.append(current_batch)
            group_lists = [g for g in group_lists if g]
        
        return batches
    
    # ─── Main Extract Method ─────────────────────────────────────────────
    
    def extract(
        self,
        image: Image.Image,
        words: List[str],       # Not used — Qwen VL reads pixels directly
        boxes: List[List[int]], # Not used — kept for API compatibility
        fields: List[str],
        ocr_source: str = "paddle",     # Not used
        boxes_normalized: bool = False, # Not used
        voting_rounds: int = 1,         # 1 = normal, 3 = accuracy boost
    ) -> Dict[str, str]:
        """
        Hybrid extraction: batch JSON first, per-field fallback for missed fields.
        
        Strategy:
            1. Batch extraction (all fields in one inference) — captures multi-line
               values and benefits from cross-field context
            2. If voting_rounds > 1: repeat batch N times, majority vote per field
            3. Identify empty/low-confidence fields from batch
            4. Re-extract only those fields one-at-a-time for better accuracy
            5. Merge: prefer per-field result if non-empty, else keep batch
        
        Args:
            image: Document page image (PIL RGB)
            words: (ignored) OCR text tokens
            boxes: (ignored) Bounding boxes
            fields: List of field names to extract
            ocr_source: (ignored) OCR engine type
            boxes_normalized: (ignored)
            voting_rounds: Number of batch extraction passes (3 for accuracy boost)
            
        Returns:
            Dict of {field_name: extracted_value}
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        w, h = image.size
        print(f"   📐 Image: {w}x{h} ({w*h:,} pixels) — processor handles resize")
        
        all_results = {}
        self._last_signals = {}
        
        # Clear VRAM fragmentation before inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Temperature schedule for voting diversity
        VOTING_TEMPERATURES = [0.1, 0.2, 0.3]
        
        # ── Step 1: Batch JSON extraction (with optional voting) ──
        if voting_rounds > 1:
            # ── Majority Voting Mode ──
            print(f"   🗳️ Step 1: Majority voting ({voting_rounds} rounds, {len(fields)} fields)")
            all_round_results = []
            
            for round_num in range(voting_rounds):
                temp = VOTING_TEMPERATURES[round_num % len(VOTING_TEMPERATURES)]
                print(f"   ── Round {round_num + 1}/{voting_rounds} (temp={temp}) ──")
                round_results = self._extract_batch_json(image, fields, temperature=temp)
                all_round_results.append(round_results)
                
                # Show round results
                filled = sum(1 for f in fields if round_results.get(f, "").strip())
                print(f"      Round {round_num + 1}: {filled}/{len(fields)} filled")
            
            # Majority vote per field
            print(f"\n   🗳️ Voting results:")
            voting_disagreed = set()  # Track explicitly — avoids floating point issues
            for field in fields:
                # Collect all non-empty answers for this field
                answers = [r.get(field, "").strip() for r in all_round_results]
                non_empty = [a for a in answers if a]
                
                if non_empty:
                    # Count occurrences and pick the most common
                    from collections import Counter
                    counter = Counter(non_empty)
                    winner, count = counter.most_common(1)[0]
                    all_results[field] = winner
                    # Track voting signal
                    self._last_signals[field] = {
                        "source": "voting",
                        "flags": [],
                        "detail": f"{count}/{voting_rounds} rounds agreed",
                    }
                    
                    if len(counter) > 1:
                        voting_disagreed.add(field)
                        self._last_signals[field]["flags"].append("voting_disagreed")
                        self._last_signals[field]["detail"] = f"{dict(counter)} → winner: '{winner}' ({count}/{voting_rounds})"
                        print(f"      🗳️ '{field}': {dict(counter)} → winner: '{winner}' ({count}/{voting_rounds})")
                    else:
                        print(f"      ✅ '{field}' = '{winner}' (unanimous {count}/{voting_rounds})")
                else:
                    all_results[field] = ""
                    self._last_signals[field] = {
                        "source": "voting",
                        "flags": ["empty_value"],
                        "detail": "all rounds returned empty",
                    }
                    print(f"      ❌ '{field}' = '' (all rounds empty)")
            
            # ── Step 1b: Zoom verification for disagreed fields ONLY ──
            # Only fields where voting rounds produced different answers
            disagreed_with_value = [
                f for f in voting_disagreed if all_results[f].strip()
            ]
            
            if disagreed_with_value:
                print(f"\n   🔍 Step 1b: Zoom verification for {len(disagreed_with_value)} disagreed fields")
                for field in disagreed_with_value:
                    voting_winner = all_results[field]
                    # Collect all candidates from voting rounds (case-insensitive set)
                    candidates = {}  # lowercase → original
                    for r in all_round_results:
                        v = r.get(field, "").strip()
                        if v:
                            candidates[v.lower()] = v
                    
                    zoom_value = self._zoom_extract_field(image, field)
                    
                    if zoom_value:
                        # Case-insensitive match against voting candidates
                        matched_candidate = candidates.get(zoom_value.lower())
                        if matched_candidate is not None:
                            # Zoomed result matches a voting candidate — use it
                            all_results[field] = matched_candidate
                            self._last_signals[field]["detail"] += f" → zoom overrode to '{matched_candidate}'"
                            if matched_candidate != voting_winner:
                                print(f"      🔍 Zoom overrides: '{field}' = '{voting_winner}' → '{matched_candidate}'")
                            else:
                                print(f"      ✅ Zoom confirms: '{field}' = '{zoom_value}'")
                                self._last_signals[field]["detail"] += " → zoom confirmed"
                        else:
                            # Zoomed result is a NEW value not seen in voting — keep voting winner
                            print(f"      ⚠️ Zoom got new value '{zoom_value}' for '{field}', keeping voting winner '{voting_winner}'")
                    else:
                        print(f"      ❌ Zoom returned empty for '{field}', keeping voting winner")
        else:
            # ── Normal single-pass mode ──
            print(f"   📋 Step 1: Batch extraction ({len(fields)} fields in one inference)")
            batch_results = self._extract_batch_json(image, fields)
            
            for field in fields:
                value = batch_results.get(field, "")
                all_results[field] = value
                self._last_signals[field] = {
                    "source": "batch",
                    "flags": [] if value.strip() else ["empty_value"],
                    "detail": "batch extraction" if value.strip() else "batch returned empty",
                }
        
        # Show results
        batch_empty = [f for f in fields if not all_results[f].strip()]
        batch_filled = [f for f in fields if all_results[f].strip()]
        print(f"   📋 Batch result: {len(batch_filled)}/{len(fields)} filled, {len(batch_empty)} empty")
        for field in fields:
            status = "✅" if all_results[field].strip() else "❌"
            print(f"      {status} '{field}' = '{all_results[field]}'")
        
        # ── Step 2: Per-field fallback for empty fields ──
        if batch_empty:
            print(f"\n   🔬 Step 2: Per-field fallback for {len(batch_empty)} empty fields: {batch_empty}")
            
            for i, field in enumerate(batch_empty):
                print(f"   [{i+1}/{len(batch_empty)}] Re-extracting: '{field}'")
                value = self._extract_single_field(image, field)
                
                # Hallucination guard
                if value and not self._validate_answer(value):
                    logger.warning(f"Hallucination guard: '{field}' answer rejected ({len(value)} chars)")
                    print(f"      🚫 Hallucination guard — clearing")
                    value = ""
                
                # Sanitize placeholders
                if value.strip() in _PLACEHOLDER_VALUES:
                    print(f"      ⚠️ Placeholder '{value}' sanitized to empty")
                    value = ""
                
                # Replace batch result if per-field found something
                if value.strip():
                    all_results[field] = value
                    self._last_signals[field] = {
                        "source": "fallback",
                        "flags": ["fallback_recovery"],
                        "detail": f"batch missed, recovered by per-field extraction",
                    }
                    print(f"      ✅ Per-field recovered: '{field}' = '{value}'")
                else:
                    print(f"      ❌ Per-field also empty for '{field}'")
        else:
            print(f"   ✅ All fields filled by batch — skipping per-field fallback")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Apply validators if enabled
        raw_values = dict(all_results)
        if ENABLE_VALIDATORS:
            validation_results = self._apply_validators(all_results)
            for field, v_result in validation_results.items():
                v_result["raw"] = raw_values.get(field, "")
                if v_result["normalized"] != raw_values.get(field, ""):
                    print(f"   🔄 Validator normalized '{field}': '{raw_values[field]}' → '{v_result['normalized']}'")
                if not v_result["is_valid"] and v_result["error"]:
                    print(f"   ⚠️ Validation failed '{field}': {v_result['error']}")
        else:
            validation_results = {}
        
        # Store metadata for server.py HITL routing
        self._last_meta = {
            "signals": dict(self._last_signals),
            "validation": validation_results,
        }
        
        # Final summary
        final_filled = sum(1 for v in all_results.values() if v.strip())
        print(f"\n   📊 Final: {final_filled}/{len(fields)} fields extracted")
        
        return raw_values
    
    def _extract_batch_json(
        self, image: Image.Image, fields: List[str], temperature: float = 0.1
    ) -> Dict[str, str]:
        """
        Batch extraction: all fields in one inference with JSON output.
        
        Uses system prompt persona for grounding + asks for structured JSON.
        Benefits from cross-field context (model sees all labels at once).
        Good for multi-line values (addresses) and related fields.
        
        Returns:
            Dict of {field_name: extracted_value}
        """
        # System prompt: extraction persona
        system_prompt = (
            "You are a high-precision form field extraction engine. "
            "Read the document image and extract only what is explicitly present. "
            "Do not guess, infer, correct, or normalize. "
            "Return values exactly as written in the document. "
            "For multi-line values (like addresses), include all lines separated by a comma. "
            "Output valid JSON only."
        )
        
        # Build field list for the prompt
        field_list = ", ".join(f'"{f}"' for f in fields)
        user_prompt = (
            f"Extract these fields from the document image and return as JSON: {field_list}. "
            f"Return a JSON object with exactly these keys. "
            f"For each field, find the matching label in the document and copy its value exactly as written. "
            f"Include complete multi-line values (e.g. full addresses with city, state, zip). "
            f"If a field is not found, set its value to empty string.\n\n"
            f"Example output format:\n"
            f'{{"Patient Name": "DOE, JOHN", "DOB": "01/15/1980", "Address": "123 MAIN ST, SPRINGFIELD, IL 62701"}}\n\n'
            f"Now extract from this document:"
        )
        
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        
        try:
            from qwen_vl_utils import process_vision_info
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,           # Batch needs more space for all fields
                    do_sample=True,
                    temperature=temperature,        # Varies per voting round
                    top_p=0.9,
                )
            
            # Decode
            input_len = inputs["input_ids"].shape[1]
            output_ids = outputs[:, input_len:]
            output_text = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            
            print(f"   📝 Batch raw output ({len(output_text)} chars): {output_text[:300]}...")
            
            # Parse JSON using existing robust parser
            return self._parse_json_output(output_text, fields)
            
        except Exception as e:
            logger.error(f"Batch extraction failed: {e}")
            print(f"   ❌ Batch extraction error: {e}")
            return {f: "" for f in fields}
    
    def _extract_single_field(
        self, image: Image.Image, field: str
    ) -> str:
        """
        Extract a single field value from a document image.
        
        Uses a focused plain-text prompt (no JSON) so the model concentrates
        entirely on one field. Returns the value and its confidence score.
        
        Returns:
            Tuple of (value: str, confidence: float)
        """
        # System prompt: sets extraction-only behavior
        system_prompt = (
            "You are a high-precision form field extraction engine. "
            "Read the document image and extract only what is explicitly present. "
            "Do not guess, infer, correct, or normalize. "
            "If uncertain, return NOT_FOUND. "
            "Output must contain only the extracted value text (or NOT_FOUND)."
        )
        
        # User prompt: per-field query with label-matching instructions
        user_prompt = (
            f"Find the value for the field: \"{field}\". "
            f"Locate the closest matching label (including minor wording variations/abbreviations) "
            f"and read the filled-in value in its associated entry area (same line/box or immediately following). "
            f"Return the complete value exactly as written, preserving capitalization, punctuation, spacing, and line breaks. "
            f"Do not include the label. "
            f"If the value is missing, blank, crossed out, illegible, ambiguous, or you are not confident it belongs to this field, return: NOT_FOUND. "
            f"Output only the value or NOT_FOUND."
        )
        
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        
        try:
            from qwen_vl_utils import process_vision_info
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            input_tokens = inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,            # Enough for full addresses and long values
                    do_sample=True,
                    temperature=0.1,               # Near-deterministic
                    top_p=0.9,
                )
            
            # Decode text — strip input tokens from output
            input_len = inputs["input_ids"].shape[1]
            output_ids = outputs[:, input_len:]
            
            output_text = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            
            # Handle NOT_FOUND and empty responses
            if not output_text or output_text.upper() in ("NOT_FOUND", "NOT FOUND", "N/A", "NONE"):
                return ""
            
            # Clean up: remove surrounding quotes if model wrapped the value
            value = output_text.strip().strip('"').strip("'").strip()
            
            return value
            
        except Exception as e:
            logger.error(f"Single-field extraction failed for '{field}': {e}")
            print(f"      ❌ Error: {e}")
            return ""
    
    def ask_question(
        self, image: Image.Image, question: str
    ) -> str:
        """
        Answer a natural language question about a document image.
        
        Unlike _extract_single_field (which is tuned for label→value extraction),
        this uses a general-purpose VQA prompt that handles:
        - Checkbox status ("Is X checked?")
        - Yes/no questions ("Does the form have a signature?")
        - Counting ("How many items are listed?")
        - Summary ("What type of document is this?")
        - Value extraction ("What is the total amount?")
        """
        system_prompt = (
            "You are a document analysis assistant. "
            "You ONLY answer questions about the document shown in the image. "
            "Rules: "
            "1. For checkbox questions, indicate whether each item is checked (✓), unchecked (☐), or crossed (✗). "
            "2. For yes/no questions, answer clearly with Yes or No followed by a brief explanation. "
            "3. For value questions, return the exact text as it appears in the document. "
            "4. Use markdown formatting: use **bold** for labels, use bullet points (- ) for lists, use numbered lists (1. ) for ordered items. "
            "5. If the question is NOT about the document (e.g. casual chat, personal questions, unrelated topics), "
            "respond ONLY with: NOT_DOCUMENT_RELATED "
            "6. If the question is gibberish or unintelligible, respond ONLY with: NOT_DOCUMENT_RELATED "
            "7. If the information is not found in the document, respond ONLY with: NOT_FOUND_IN_DOCUMENT "
            "Be concise, accurate, and well-formatted."
        )
        
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        try:
            from qwen_vl_utils import process_vision_info
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.2,
                    top_p=0.9,
                )
            
            input_len = inputs["input_ids"].shape[1]
            output_ids = outputs[:, input_len:]
            
            output_text = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            
            return output_text if output_text else ""
            
        except Exception as e:
            logger.error(f"Q&A failed for '{question}': {e}")
            print(f"      ❌ Q&A Error: {e}")
            return ""
    
    def ask_question_multipage(
        self, images: list, question: str, history: list = None
    ) -> str:
        """
        Answer a question using multi-page context.
        
        Sends up to 5 page images in a single VLM call so the model
        can reason across pages. Accepts optional conversation history
        for follow-up questions.
        
        Args:
            images: List of PIL Images (one per page, max 5)
            question: Natural language question
            history: List of {"role": "user"|"assistant", "content": str}
        
        Returns:
            Answer string
        """
        if not images:
            return ""
        
        # Cap at 5 pages to avoid VRAM issues
        images = images[:5]
        
        system_prompt = (
            "You are a document analysis assistant. "
            "You ONLY answer questions about the document shown in the image(s). "
            "Rules: "
            "1. For checkbox questions, indicate whether each item is checked (✓), unchecked (☐), or crossed (✗). "
            "2. For yes/no questions, answer clearly with Yes or No followed by a brief explanation. "
            "3. For value questions, return the exact text as it appears in the document. "
            "4. Use markdown formatting: use **bold** for labels, use bullet points (- ) for lists, use numbered lists (1. ) for ordered items. "
            "5. If the question is NOT about the document (e.g. casual chat, personal questions, unrelated topics), "
            "respond ONLY with: NOT_DOCUMENT_RELATED "
            "6. If the question is gibberish or unintelligible, respond ONLY with: NOT_DOCUMENT_RELATED "
            "7. If the information is not found in the document, respond ONLY with: NOT_FOUND_IN_DOCUMENT "
            "Be concise, accurate, and well-formatted."
        )
        
        # Build messages with conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if provided
        if history:
            for turn in history:
                messages.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", ""),
                })
        
        # Build user message with all page images + question
        user_content = []
        for i, img in enumerate(images):
            if img.mode != "RGB":
                img = img.convert("RGB")
            user_content.append({"type": "image", "image": img})
        
        if len(images) > 1:
            user_content.append({
                "type": "text",
                "text": f"This document has {len(images)} pages shown above. {question}",
            })
        else:
            user_content.append({"type": "text", "text": question})
        
        messages.append({"role": "user", "content": user_content})
        
        try:
            from qwen_vl_utils import process_vision_info
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            print(f"   💬 Multi-page Q&A ({len(images)} pages): '{question[:80]}...'")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                )
            
            input_len = inputs["input_ids"].shape[1]
            output_ids = outputs[:, input_len:]
            output_text = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            
            print(f"   📝 Answer ({len(output_text)} chars): {output_text[:200]}...")
            
            return output_text
            
        except Exception as e:
            logger.error(f"Multi-page Q&A failed: {e}")
            print(f"      ❌ Multi-page Q&A Error: {e}")
            return ""
    
    # ─── Multi-Page Extraction Architecture ───────────────────────────
    # Generic multi-page VLM call — used by all extraction types.
    # Future types (radio, signature, barcode, links) should add a new
    # public wrapper method that calls _multipage_extract with the
    # appropriate system/user prompts.
    
    def _multipage_extract(
        self, images: list, system_prompt: str, user_prompt: str,
        max_pages: int = 5, max_tokens: int = 2048
    ) -> str:
        """
        Generic multi-page VLM call — sends up to max_pages images in one inference.
        
        All multi-page extraction methods should call this instead of duplicating
        the VLM call boilerplate. Returns raw model output text.
        
        Args:
            images: List of PIL Images (one per page)
            system_prompt: System persona prompt
            user_prompt: User instruction prompt
            max_pages: Cap pages to avoid VRAM issues (default 5)
            max_tokens: Max output tokens (default 2048)
        
        Returns:
            Raw model output text string
        """
        if not images:
            return ""
        
        images = images[:max_pages]
        
        # Build user content: all page images + text prompt
        user_content = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": user_prompt})
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        
        try:
            from qwen_vl_utils import process_vision_info
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.9,
                )
            
            input_len = inputs["input_ids"].shape[1]
            output_ids = outputs[:, input_len:]
            return self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            
        except Exception as e:
            logger.error(f"Multi-page extraction failed: {e}")
            print(f"      ❌ Multi-page extraction error: {e}")
            return ""
    
    def extract_fields_multipage(
        self, images: list, fields: List[str]
    ) -> Dict[str, str]:
        """
        Extract fields across all pages in a single VLM call.
        
        Sends all page images at once so the model can locate fields
        anywhere in the document. Useful when fields span pages or
        the user doesn't know which page has which field.
        
        Returns:
            Dict of {field_name: extracted_value}
        """
        field_list = ", ".join(f'"{f}"' for f in fields)
        
        system_prompt = (
            "You are a high-precision form field extraction engine. "
            "Read ALL document pages and extract only what is explicitly present. "
            "Do not guess, infer, correct, or normalize. "
            "Return values exactly as written in the document. "
            "Output valid JSON only."
        )
        
        user_prompt = (
            f"This document has {len(images)} pages shown above. "
            f"Extract these fields from ANY page: {field_list}. "
            f"Return a JSON object with exactly these keys. "
            f"If a field is not found on any page, set its value to empty string.\n\n"
            f"Example output format:\n"
            f'{{"Patient Name": "DOE, JOHN", "DOB": "01/15/1980"}}\n\n'
            f"Now extract from this document:"
        )
        
        print(f"   📄 Multi-page field extraction ({len(images)} pages, {len(fields)} fields)")
        raw = self._multipage_extract(images, system_prompt, user_prompt, max_tokens=1024)
        print(f"   📝 Multi-page raw ({len(raw)} chars): {raw[:300]}...")
        
        return self._parse_json_output(raw, fields)
    
    def extract_checkboxes_multipage(
        self, images: list
    ) -> list:
        """
        Auto-detect checkboxes across all pages in a single VLM call.
        
        Returns:
            List of {"label": str, "checked": bool, "page": int}
        """
        system_prompt = (
            "You are a checkbox detection engine. "
            "Examine ALL pages of this document and find every physical checkbox. "
            "For each checkbox, determine if it is checked (filled/marked) or unchecked (empty). "
            "Return a JSON array of objects with: label, checked (boolean), page (1-indexed)."
        )
        
        user_prompt = (
            f"This document has {len(images)} pages. "
            f"Find ALL checkboxes on ALL pages. "
            f"Return JSON array: "
            f'[{{"label": "Item Name", "checked": true, "page": 1}}, ...]'
        )
        
        print(f"   ☑️ Multi-page checkbox detection ({len(images)} pages)")
        raw = self._multipage_extract(images, system_prompt, user_prompt, max_tokens=2048)
        
        # Parse checkbox list
        try:
            repaired = self._repair_json(raw)
            result = json.loads(repaired)
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback: try the existing parser
        return self._parse_checkbox_list(raw)
    
    def extract_table_multipage(
        self, images: list, query: str
    ) -> tuple:
        """
        Extract table data across all pages in a single VLM call.
        
        Args:
            query: Table name, column heading, or row reference
        
        Returns:
            Tuple of (data, table_type) where table_type is "table"|"column"|"row"|None
        """
        system_prompt = (
            "You are a table extraction engine. "
            "Examine ALL pages and find the requested table data. "
            "Return the data as a JSON array of row objects. "
            "Each row object should have column headers as keys."
        )
        
        user_prompt = (
            f"This document has {len(images)} pages. "
            f'Extract the table data for: "{query}". '
            f"Return as a JSON array of row objects.\n"
            f'Example: [{{"Name": "John", "Amount": "$100"}}, ...]'
        )
        
        print(f"   📊 Multi-page table extraction ({len(images)} pages): '{query}'")
        raw = self._multipage_extract(images, system_prompt, user_prompt, max_tokens=2048)
        
        data = self._parse_table_json(raw)
        if data is not None:
            if isinstance(data, list):
                return data, "table"
            elif isinstance(data, dict):
                return data, "row"
        
        return None, None
    
    def _zoom_extract_field(
        self, image: Image.Image, field: str
    ) -> str:
        """
        Multi-crop zoom extraction for a single field.
        
        Splits image into top/bottom halves (with 10% overlap) and extracts
        the field from each. Each half gives the model ~2× more pixels per
        area, resolving character-level ambiguities (e.g., 6 vs 0).
        
        Only one half will contain the field — returns that result.
        
        Returns:
            str: extracted value, or "" if not found
        """
        w, h = image.size
        overlap = int(h * 0.10)  # 10% overlap to avoid splitting a field
        
        # Top half: 0 to 60% of height
        top_crop = image.crop((0, 0, w, h // 2 + overlap))
        # Bottom half: 40% to 100% of height
        bottom_crop = image.crop((0, h // 2 - overlap, w, h))
        
        crops = [("top", top_crop), ("bottom", bottom_crop)]
        
        for crop_name, crop_img in crops:
            value = self._extract_single_field(crop_img, field)
            
            # Hallucination guard
            if value and not self._validate_answer(value):
                value = ""
            
            # Sanitize placeholders
            if value.strip() in _PLACEHOLDER_VALUES:
                value = ""
            
            if value.strip():
                print(f"      🔍 Zoom ({crop_name}): '{field}' = '{value}'")
                return value
        
        return ""
    
    # _compute_confidence removed — Review Signal System replaces numeric confidence
    # with source/flags-based review signals (see _last_signals)
    
    
    def _validate_answer(self, value: str) -> bool:
        """
        Hallucination guard: reject answers that look like model reasoning.
        
        Real form field values are short (< 200 chars). Answers like
        "The value of smoke is less than the value of alcohol." are model
        reasoning artifacts, not actual field values.
        
        Returns True if acceptable, False if hallucinated.
        """
        if not value:
            return True
        if len(value) > MAX_ANSWER_CHARS:
            return False
        # Full sentence with reasoning verb = likely hallucination
        if _HALLUCINATION_PATTERN.search(value) and len(value) > 40:
            return False
        return True
    
    # ─── Checkbox Extraction ───
    
    def extract_checkboxes(
        self, image: Image.Image, fields: List[str] = None
    ) -> List[dict]:
        """
        Extract checkbox status from a document image.
        
        Two modes:
            1. fields=None  → Auto-detect ALL physical checkboxes
            2. fields=[...]  → Extract specified checkbox fields only
        
        Returns:
            List of {"label": str, "checked": bool, "confidence": float}
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        if fields:
            # User-specified checkbox fields
            print(f"☑️ Extracting {len(fields)} specified checkbox fields...")
            results = []
            for field in fields:
                checked, confidence = self._extract_single_checkbox(image, field)
                
                # Zoom verification for low-confidence results
                if confidence < 0.7:
                    print(f"   🔬 Low confidence ({confidence:.2f}) for '{field}' — zoom verifying...")
                    zoom_checked, zoom_conf = self._verify_checkbox_zoom(image, field)
                    if zoom_conf > confidence:
                        checked = zoom_checked
                        confidence = zoom_conf
                        print(f"   ✅ Zoom improved: '{field}' = {'Checked' if checked else 'Unchecked'} (conf: {confidence:.2f})")
                
                results.append({
                    "label": field,
                    "checked": checked,
                    "confidence": round(confidence, 3),
                })
                status = "Checked" if checked else "Unchecked"
                print(f"   {'☑' if checked else '☐'} '{field}' = {status} (conf: {confidence:.2f})")
            
            return results
        else:
            # Auto-detect all checkboxes
            return self._detect_all_checkboxes(image)
    
    def _detect_all_checkboxes(self, image: Image.Image) -> List[dict]:
        """
        Auto-detect ALL physical checkboxes in a document image.
        """
        # ── Diagnostic: Ask VLM what it sees ──
        diag_messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe what you see in this document image in 2-3 sentences. What type of document is it? What elements does it contain?"},
            ]}
        ]
        try:
            from qwen_vl_utils import process_vision_info
            text = self.processor.apply_chat_template(
                diag_messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(diag_messages)
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                diag_out = self.model.generate(**inputs, max_new_tokens=200)
            diag_text = self.processor.batch_decode(
                diag_out[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )[0].strip()
            print(f"   🔍 Document description: {diag_text[:300]}")
        except Exception as e:
            print(f"   🔍 Diagnostic failed: {e}")
        
        # ── Primary: Simplified checkbox prompt ──
        system_prompt = (
            "You are a document form analyzer. "
            "Extract checkbox information from the document image."
        )
        
        user_prompt = (
            "This document is a form with checkboxes next to items. "
            "Scan the ENTIRE document thoroughly — top to bottom, left to right. "
            "Do NOT skip any checkboxes, even if the text is small or hard to read.\n\n"
            "For EVERY checkbox in the document, tell me:\n"
            "1. What text/label is next to the checkbox\n"
            "2. Whether the checkbox is checked (has a mark ✓✗X inside) or unchecked (empty)\n\n"
            "Return a JSON array like this:\n"
            '[{"label": "Item name", "checked": true}, {"label": "Other item", "checked": false}]\n\n'
            "IMPORTANT: Include ALL checkboxes. Count them to make sure you haven't missed any.\n"
            "Return ONLY the JSON array."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        
        try:
            from qwen_vl_utils import process_vision_info
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            print(f"   ☑️ Scanning document for checkboxes...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.9,
                )
            
            input_len = inputs["input_ids"].shape[1]
            output_ids = outputs[:, input_len:]
            
            output_text = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            
            print(f"   📋 Checkbox raw output: {output_text[:500]}")
            
            # Parse the JSON array (no confidence needed)
            checkboxes = self._parse_checkbox_list(output_text)
            
            print(f"   📊 Found {len(checkboxes)} checkboxes")
            
            # If first pass found nothing, try a descriptive fallback pass
            if not checkboxes:
                print(f"   🔄 Pass 1 empty — trying descriptive fallback...")
                checkboxes = self._detect_checkboxes_fallback(image)
                if checkboxes:
                    print(f"   📊 Fallback found {len(checkboxes)} checkboxes")
            
            # Log all checkbox results
            for cb in checkboxes:
                status = "Checked" if cb["checked"] else "Unchecked"
                source = cb.get("signal", {}).get("source", "batch")
                print(f"   {'☑' if cb['checked'] else '☐'} '{cb['label']}' = {status} [{source}]")
            
            return checkboxes
            
        except Exception as e:
            logger.error(f"Checkbox detection failed: {e}")
            print(f"   ❌ Checkbox detection error: {e}")
            return []
    
    def _detect_checkboxes_fallback(self, image: Image.Image) -> List[dict]:
        """
        Fallback: per-item VQA checkbox detection.
        
        Step 1: Ask model to list all items with checkboxes
        Step 2: For each item, ask individually if that checkbox is checked
        
        This avoids the 'mark everything checked' bias by forcing
        per-item visual inspection.
        """
        # Step 1: Get list of all items that have checkboxes
        system_prompt = (
            "You are a document reader. Read the document carefully."
        )
        
        user_prompt = (
            "This document has checkboxes next to some items. "
            "List ONLY the text labels of items that have a checkbox next to them. "
            "Return one item per line, nothing else.\n\n"
            "Example:\n"
            "Fresh celery\n"
            "Grapes\n"
            "Ice cream"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        
        try:
            from qwen_vl_utils import process_vision_info
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.9,
                )
            
            input_len = inputs["input_ids"].shape[1]
            output_ids = outputs[:, input_len:]
            
            items_text = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            
            print(f"   📋 Fallback items found: {items_text[:300]}")
            
            # Parse item list
            items = []
            for line in items_text.split("\n"):
                line = line.strip().lstrip("- •*0123456789.)")
                if line and len(line) > 1 and len(line) < 100:
                    items.append(line)
            
            if not items:
                return []
            
            print(f"   📊 Found {len(items)} items, now checking each...")
            
            # Step 2: For each item, ask if its checkbox is checked
            checkboxes = []
            for item in items:
                checked, conf = self._extract_single_checkbox(image, item)
                checkboxes.append({
                    "label": item,
                    "checked": checked,
                    "confidence": round(conf, 3),
                })
            
            return checkboxes
            
        except Exception as e:
            logger.error(f"Checkbox fallback detection failed: {e}")
            print(f"   ❌ Fallback error: {e}")
            return []
    
    def _extract_single_checkbox(
        self, image: Image.Image, field: str
    ) -> Tuple[bool, float]:
        """
        Extract a single checkbox status from a document image.
        
        Uses a focused prompt asking specifically about one checkbox.
        
        Returns:
            Tuple of (checked: bool, confidence: float)
        """
        system_prompt = (
            "You are a checkbox state reader. "
            "CHECKED = the box has a mark inside (✓, ✗, X, filled, darkened, any mark). "
            "UNCHECKED = the box is empty/blank/hollow inside with no mark. "
            "Respond with ONLY one word."
        )
        
        user_prompt = (
            f"Find the checkbox next to \"{field}\" in this document.\n"
            f"Look INSIDE the checkbox box. Is there any mark, check, X, or filling inside?\n"
            f"- If YES (any mark inside) → respond: checked\n"
            f"- If NO (empty/blank inside) → respond: unchecked\n\n"
            f"Reply with ONLY one word: checked OR unchecked"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        
        try:
            from qwen_vl_utils import process_vision_info
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.9,
                )
            
            input_len = inputs["input_ids"].shape[1]
            output_ids = outputs[:, input_len:]
            
            output_text = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip().lower()
            
            # Parse response
            checked = "check" in output_text and "uncheck" not in output_text
            
            return checked
            
        except Exception as e:
            logger.error(f"Single checkbox extraction failed for '{field}': {e}")
            print(f"      ❌ Checkbox error for '{field}': {e}")
            return False
    
    def _verify_checkbox_zoom(
        self, image: Image.Image, label: str
    ) -> bool:
        """
        Zoom verification for a single checkbox.
        
        Splits image into top/bottom halves (with 10% overlap) and re-checks
        the checkbox at higher effective resolution.
        
        Returns:
            bool: whether checkbox is checked
        """
        w, h = image.size
        overlap = int(h * 0.10)
        
        # Top half: 0 to 60% of height
        top_crop = image.crop((0, 0, w, h // 2 + overlap))
        # Bottom half: 40% to 100% of height
        bottom_crop = image.crop((0, h // 2 - overlap, w, h))
        
        crops = [("top", top_crop), ("bottom", bottom_crop)]
        results = []
        
        for crop_name, crop_img in crops:
            checked = self._extract_single_checkbox(crop_img, label)
            results.append(checked)
            print(f"      🔍 Zoom ({crop_name}): '{label}' = {'Checked' if checked else 'Unchecked'}")
        
        # Return majority result
        return sum(results) > len(results) / 2
    
    def _parse_checkbox_list(self, output_text: str) -> List[dict]:
        """
        Parse a JSON array of checkbox results from model output.
        
        Handles markdown code blocks, partial JSON, etc.
        
        Returns:
            List of {"label": str, "checked": bool, "signal": dict}
        """
        parsed = None
        
        # Try direct JSON parse
        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError:
            pass
        
        # Try markdown code block
        if parsed is None:
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', output_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
        
        # Try finding JSON array in text
        if parsed is None:
            bracket_match = re.search(r'\[.*\]', output_text, re.DOTALL)
            if bracket_match:
                try:
                    parsed = json.loads(bracket_match.group(0))
                except json.JSONDecodeError:
                    pass
        
        if not isinstance(parsed, list):
            logger.warning(f"Could not parse checkbox JSON: {output_text[:200]}")
            return []
        
        # Normalize and validate
        results = []
        for item in parsed:
            if isinstance(item, dict) and "label" in item:
                label = str(item["label"]).strip()
                checked = bool(item.get("checked", False))
                if label:
                    results.append({
                        "label": label,
                        "checked": checked,
                        "signal": {
                            "source": "checkbox_batch",
                            "flags": [],
                        },
                    })
        
        return results
    
    def _apply_validators(self, results: Dict[str, str]) -> Dict[str, dict]:
        """
        FIX 2: Run every extracted value through validators.py.
        
        Uses find_validator() for smart field-to-validator routing:
        - "Patient DOB" → validate_date (substring match on "DOB")
        - "Home Phone" → validate_phone (substring match on "Phone")
        - "Gibberish" → no validator → accepts non-empty as-is
        
        Valid values get normalized (e.g. dates → ISO 8601).
        Invalid values get flagged with error reason for HITL routing.
        
        Returns:
            Dict of {field: {is_valid, normalized, error}}
        """
        try:
            from models.validators import validate_field
        except ImportError:
            try:
                from validators import validate_field
            except ImportError:
                logger.warning("validators.py not found — skipping validation")
                return {
                    field: {"is_valid": True, "normalized": value, "error": None}
                    for field, value in results.items()
                }
        
        validation_results = {}
        
        for field, value in results.items():
            if not value or value.strip() == "":
                validation_results[field] = {
                    "is_valid": False,
                    "normalized": value,
                    "error": "Empty value",
                }
                continue
            
            try:
                is_valid, normalized, error = validate_field(value, field)
                validation_results[field] = {
                    "is_valid": is_valid,
                    "normalized": normalized,
                    "error": error,
                }
                
                if not is_valid:
                    logger.info(f"Validation failed — {field}: '{value}' → {error}")
            
            except Exception as e:
                logger.warning(f"Validator error for '{field}': {e}")
                validation_results[field] = {
                    "is_valid": True,
                    "normalized": value,
                    "error": None,
                }
        
        return validation_results
    
    def get_last_signals(self) -> Dict[str, dict]:
        """Return review signals from the last extraction."""
        return self._last_signals
    
    def get_last_meta(self) -> dict:
        """Return full metadata (signals + validation) from last extraction."""
        return self._last_meta
    # ─── Table Pre-Scan ──────────────────────────────────────────────────
    
    def scan_table_structure(self, image: Image.Image) -> dict:
        """
        Lightweight VLM scan to detect if a page has a real data table.
        
        Returns:
            {
                "has_table": bool,
                "columns": ["Col1", "Col2", ...],   # if has_table
                "row_index": "SL."                    # row index column name
            }
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        system_prompt = "You are a document layout analyzer. Output valid JSON only."
        
        user_prompt = (
            'Look at this document image. Is there a DATA TABLE — meaning a grid where '
            'MULTIPLE ROWS have the SAME COLUMN STRUCTURE with repeated similar data '
            '(like a list of items, transactions, or records)?\n\n'
            'A data table has:\n'
            '- A HEADER ROW with column titles\n'
            '- 2 or more DATA ROWS that all share the same columns\n'
            '- Each row represents a similar type of record\n\n'
            'A FORM is NOT a data table. Forms have fields like name, date, address '
            'where each row is a DIFFERENT type of information.\n\n'
            'If there IS a data table, return JSON:\n'
            '{"has_table": true, "columns": ["Col1", "Col2", ...], "row_index": "SL."}\n\n'
            'If there is NO data table (only forms, labels, or sections), return:\n'
            '{"has_table": false}'
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        
        try:
            from qwen_vl_utils import process_vision_info
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            print(f"   🔍 Table pre-scan...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,  # Short response expected
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            
            input_len = inputs["input_ids"].shape[1]
            output_ids = outputs[:, input_len:]
            output_text = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            
            print(f"   🔍 Scan result: {output_text[:200]}")
            
            # Parse the result
            parsed = self._parse_table_json(output_text)
            if isinstance(parsed, dict) and "has_table" in parsed:
                if parsed.get("has_table"):
                    columns = parsed.get("columns", [])
                    row_index = parsed.get("row_index", "")
                    # Post-scan validation: reject if columns look like form labels
                    if columns and self._validate_scan_columns(columns):
                        result = {
                            "has_table": True,
                            "columns": columns,
                            "row_index": str(row_index).strip(),
                        }
                        print(f"   🔍 ✅ Table detected: {len(columns)} columns, row_index='{row_index}'")
                        return result
                    else:
                        print(f"   🔍 Scan rejected — columns look like form labels")
                        return {"has_table": False}
                else:
                    print(f"   🔍 No data table found")
                    return {"has_table": False}
            
            print(f"   🔍 Could not parse scan response")
            return {"has_table": False}
            
        except Exception as e:
            logger.error(f"Table scan failed: {e}")
            print(f"   ❌ Table scan error: {e}")
            return {"has_table": False}
    
    def _validate_scan_columns(self, columns: list) -> bool:
        """
        Validate that detected columns look like real table headers, not form labels.
        
        Real table headers are short (SL., Price, Qty, Total).
        Form labels are long (Unit Holder Name, Mutual Fund Name).
        """
        if len(columns) < 2:
            return False
        
        # Average column name length — table headers are typically short
        avg_len = sum(len(str(c)) for c in columns) / len(columns)
        if avg_len > 25:
            print(f"   🔍 Avg column name length {avg_len:.0f} > 25 — looks like form labels")
            return False
        
        # If more than half of columns are > 30 chars, likely form labels
        long_cols = sum(1 for c in columns if len(str(c)) > 30)
        if long_cols > len(columns) / 2:
            print(f"   🔍 {long_cols}/{len(columns)} columns > 30 chars — looks like form labels")
            return False
        
        return True

    # ─── Table Extraction ───────────────────────────────────────────────
    
    def _detect_table_mode(self, query: str) -> str:
        """
        Detect what kind of table query this is.
        Returns: 'row' | 'table_or_column'
        """
        import re
        # Row patterns: "Row 2", "#2", "Row No. 3", "SL. 3", "SL 7", "Sr. No. 5", "No. 4"
        row_pattern = r'^(row\s*(no\.?\s*)?|#|sl\.?\s*|sr\.?\s*(no\.?\s*)?|no\.?\s*)\s*\d+$'
        if re.match(row_pattern, query.strip(), re.IGNORECASE):
            return 'row'
        return 'table_or_column'
    
    def _extract_row_number(self, query: str) -> int:
        """Extract the row number from a row query like 'Row 2' or '#3'."""
        import re
        m = re.search(r'\d+', query)
        return int(m.group()) if m else 1
    
    def extract_table(self, image: Image.Image, query: str) -> tuple:
        """
        Extract table data from a document image.
        
        Three modes:
          - Table name → full table as array of row objects
          - Column heading → array of column values
          - Row reference → single row as object
        
        Args:
            image: Document page image (PIL RGB)
            query: The field name (table name, column heading, or "Row N")
            
        Returns:
            (data, table_type) where:
              - data: parsed result (list of dicts, list of values, or dict)
              - table_type: "table" | "column" | "row" | None (if not a table)
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        mode = self._detect_table_mode(query)
        
        if mode == 'row':
            row_num = self._extract_row_number(query)
            return self._extract_table_row(image, row_num)
        else:
            return self._extract_table_or_column(image, query)
    
    def _extract_table_or_column(self, image: Image.Image, query: str) -> tuple:
        """Try to extract a full table or a specific column."""
        
        query_lower = query.strip().lower()
        is_table_query = "table" in query_lower
        
        system_prompt = (
            "You are a document table extraction engine. "
            "Extract ONLY from real data tables (grids with column headers or numbered rows, 2+ data rows, 2+ columns). "
            "Do NOT extract from form fields, section borders, or label-value pairs. "
            "Output valid JSON only."
        )
        
        if is_table_query:
            # User wants the whole table — find any/all data tables
            user_prompt = (
                f'Look at this document and find the data table.\n'
                f'A data table is a grid with column headers and multiple data rows.\n\n'
                f'Return ALL rows of the table as a JSON array of objects.\n'
                f'Each object should use the column headers as keys.\n'
                f'Example: [{{"SL": "1", "Description": "Item A", "Price": "$10", "Qty": "2", "Total": "$20"}}, '
                f'{{"SL": "2", "Description": "Item B", "Price": "$15", "Qty": "1", "Total": "$15"}}]\n\n'
                f'If there is no data table in this document, return exactly: NOT_A_TABLE'
            )
        else:
            # User specified a column heading — extract that column
            user_prompt = (
                f'Look at this document and find a table column with a heading that matches or is similar to "{query}".\n\n'
                f'If you find a matching column heading in a data table:\n'
                f'  Return ALL values in that column as a JSON array of strings.\n'
                f'  Example: ["value1", "value2", "value3"]\n\n'
                f'If "{query}" is not a column heading in any table, return exactly: NOT_A_TABLE'
            )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        
        try:
            from qwen_vl_utils import process_vision_info
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            print(f"   📊 Table extraction: querying '{query}'...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,  # Tables can be large
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            
            input_len = inputs["input_ids"].shape[1]
            output_ids = outputs[:, input_len:]
            output_text = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            
            print(f"   📊 Table raw output ({len(output_text)} chars): {output_text[:300]}")
            
            # Check for explicit "not a table" response
            if "NOT_A_TABLE" in output_text.upper():
                print(f"   📊 VLM says '{query}' is not a table")
                return None, None
            
            # Parse the JSON output
            parsed = self._parse_table_json(output_text)
            if parsed is None:
                return None, None
            
            # Determine type and validate
            if isinstance(parsed, list) and len(parsed) > 0:
                if isinstance(parsed[0], dict):
                    # Array of row objects → full table
                    if not self._validate_table(parsed):
                        print(f"   📊 Table validation failed — not a real table")
                        return None, None
                    print(f"   📊 ✅ Table extracted: {len(parsed)} rows, {len(parsed[0])} columns")
                    return parsed, "table"
                elif isinstance(parsed[0], str):
                    # Array of strings → column values
                    if len(parsed) < 2:
                        print(f"   📊 Column has < 2 values — rejected")
                        return None, None
                    print(f"   📊 ✅ Column extracted: {len(parsed)} values")
                    return parsed, "column"
            
            return None, None
            
        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            print(f"   ❌ Table extraction error: {e}")
            return None, None
    
    def _extract_table_row(self, image: Image.Image, row_num: int) -> tuple:
        """Extract a specific row from a table."""
        
        system_prompt = (
            "You are a document table extraction engine. "
            "Output valid JSON only."
        )
        
        user_prompt = (
            f"Find a data table in this document and extract row number {row_num}.\n"
            f"Return the row as a JSON object where keys are the column headers.\n"
            f'Example: {{"Sr.No": "{row_num}", "Name": "Alice", "Amount": "500"}}\n'
            f"If no table is found or row {row_num} doesn't exist, return: NOT_A_TABLE"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        
        try:
            from qwen_vl_utils import process_vision_info
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            print(f"   📊 Row extraction: querying row {row_num}...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            
            input_len = inputs["input_ids"].shape[1]
            output_ids = outputs[:, input_len:]
            output_text = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            
            print(f"   📊 Row raw output: {output_text[:300]}")
            
            if "NOT_A_TABLE" in output_text.upper():
                return None, None
            
            parsed = self._parse_table_json(output_text)
            if isinstance(parsed, dict) and len(parsed) >= 2:
                print(f"   📊 ✅ Row {row_num} extracted: {len(parsed)} columns")
                return parsed, "row"
            
            return None, None
            
        except Exception as e:
            logger.error(f"Row extraction failed: {e}")
            print(f"   ❌ Row extraction error: {e}")
            return None, None
    
    def _parse_table_json(self, output_text: str):
        """Parse JSON from table extraction output. Returns parsed data or None."""
        import re
        
        def _try_parse(text):
            """Try JSON parse, and if it fails, try fixing Python dict syntax."""
            # Direct JSON
            try:
                return json.loads(text)
            except (json.JSONDecodeError, ValueError):
                pass
            # Python dict → JSON: single quotes → double quotes, booleans
            try:
                fixed = text.replace("'", '"')
                fixed = fixed.replace("True", "true").replace("False", "false").replace("None", "null")
                return json.loads(fixed)
            except (json.JSONDecodeError, ValueError):
                pass
            return None
        
        # Try direct parse
        result = _try_parse(output_text)
        if result is not None:
            return result
        
        # Try markdown code block
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', output_text, re.DOTALL)
        if json_match:
            result = _try_parse(json_match.group(1))
            if result is not None:
                return result
        
        # Try finding JSON array or object
        bracket_match = re.search(r'[\[{].*[\]}]', output_text, re.DOTALL)
        if bracket_match:
            result = _try_parse(bracket_match.group(0))
            if result is not None:
                return result
        
        return None
    
    def _validate_table(self, rows: List[dict]) -> bool:
        """
        Validate that parsed rows are a real data table, not form layout.
        
        A real table needs:
          - 2+ rows
          - 2+ columns
          - EITHER: first row keys look like headers (text labels)
          - OR: first column values have sequential numbering
        """
        import re
        
        if len(rows) < 2:
            return False
        
        # Check column count
        col_counts = [len(r) for r in rows]
        if max(col_counts) < 2:
            return False
        
        # Get first column values (first key's values across rows)
        first_key = list(rows[0].keys())[0] if rows[0] else None
        if not first_key:
            return False
        
        first_col_values = [str(r.get(first_key, "")).strip() for r in rows]
        
        # Check for sequential numbering in first column
        has_numbering = self._has_sequential_numbering(first_col_values)
        
        # Check for proper column headers (keys are text labels, not numbers)
        keys = list(rows[0].keys())
        has_headers = any(
            not k.replace(" ", "").replace(".", "").replace("#", "").isdigit()
            for k in keys
        )
        
        if has_numbering or has_headers:
            return True
        
        print(f"   📊 Validation: no headers and no row numbering — rejected")
        return False
    
    def _has_sequential_numbering(self, values: List[str]) -> bool:
        """Check if values look like sequential row numbers."""
        import re
        
        # Extract numeric parts
        nums = []
        for v in values:
            # Match patterns: "1", "No.1", "No 1", "#1", "Pos 1", "Sr. No. 1", etc.
            m = re.search(r'\d+', v)
            if m:
                nums.append(int(m.group()))
        
        if len(nums) < 2:
            return False
        
        # Check if roughly sequential (allows gaps but must be increasing)
        is_sequential = all(nums[i] < nums[i+1] for i in range(len(nums)-1))
        # Also check simple 1,2,3... pattern
        is_simple_seq = nums == list(range(nums[0], nums[0] + len(nums)))
        
        return is_sequential or is_simple_seq
    
    def auto_detect_fields(self, image: Image.Image) -> List[str]:
        """
        Auto-detect field labels in a document image using Qwen2-VL.
        
        Scans the document visually and identifies all form fields, labels,
        and data points that could be extracted.
        
        Args:
            image: Document page image (PIL RGB)
            
        Returns:
            List of detected field names
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # No manual resize — processor handles resolution via min_pixels/max_pixels
        
        prompt = (
            "You are a document analysis assistant. "
            "Look at this document image and list ALL form field labels you can see. "
            "Include fields like names, dates, addresses, phone numbers, IDs, checkboxes, signatures, etc. "
            "Return ONLY a JSON array of field name strings, for example: "
            '[\"Patient Name\", \"Date of Birth\", \"Phone Number\", \"Address\"]. '
            "Be specific — instead of just 'Name', use 'Patient Name', 'Physician Name', etc. "
            "Do not include any explanation, only the JSON array."
        )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        try:
            from qwen_vl_utils import process_vision_info
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            print(f"   🔍 Qwen2-VL scanning for field labels...")
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            
            print(f"   📄 Auto-detect raw: {output_text[:300]}")
            
            return self._parse_field_list(output_text)
            
        except Exception as e:
            logger.error(f"Auto-detect fields failed: {e}")
            print(f"   ❌ Auto-detect error: {e}")
            return []
    
    def _parse_field_list(self, output_text: str) -> List[str]:
        """Parse a JSON array of field names from model output."""
        
        def _normalize_items(items: list) -> List[str]:
            """Handle both plain strings and dicts (e.g. {"field_name": "Date"})."""
            result = []
            for f in items:
                if not f:
                    continue
                if isinstance(f, dict):
                    # Model sometimes returns [{"Field Name": "Date"}, ...] or [{"field_name": "Date"}, ...]
                    # Extract the first value from the dict
                    vals = list(f.values())
                    if vals and isinstance(vals[0], str) and vals[0].strip():
                        result.append(vals[0].strip())
                elif isinstance(f, str) and f.strip():
                    result.append(f.strip())
            return result
        
        # Try direct JSON parse
        try:
            parsed = json.loads(output_text)
            if isinstance(parsed, list):
                return _normalize_items(parsed)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown code block
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', output_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if isinstance(parsed, list):
                    return _normalize_items(parsed)
            except json.JSONDecodeError:
                pass
        
        # Try finding JSON array in text
        bracket_match = re.search(r'\[.*?\]', output_text, re.DOTALL)
        if bracket_match:
            try:
                parsed = json.loads(bracket_match.group(0))
                if isinstance(parsed, list):
                    return _normalize_items(parsed)
            except json.JSONDecodeError:
                pass
        
        # Fallback: split comma-separated values
        fields = [f.strip().strip('"\'') for f in output_text.split(',')]
        return [f for f in fields if f and len(f) > 1]
    
    @staticmethod
    def _repair_json(text: str) -> str:
        """
        Attempt to repair common JSON malformations from VLM output.
        
        Fixes:
          - Markdown code fences (```json ... ```)
          - Trailing commas before } or ]
          - Single-quoted strings → double-quoted (safe for apostrophes)
          - Python booleans/None → JSON booleans/null
          - Truncated JSON (missing closing })
        """
        import re as _re
        # Strip markdown code fences
        text = _re.sub(r'^\s*```(?:json)?\s*', '', text)
        text = _re.sub(r'\s*```\s*$', '', text)
        # Trailing commas: {"a": 1,} or ["a",]
        text = _re.sub(r',\s*([}\]])', r'\1', text)
        # Python booleans/None → JSON
        text = text.replace('True', 'true').replace('False', 'false').replace('None', 'null')
        # Single-quoted strings → double-quoted (SAFE for apostrophes)
        # Only replace quotes at word boundaries, not inside contractions like O'Brien
        # Pattern: replace ' that is preceded/followed by non-alphabetic chars
        if text.count("'") > text.count('"'):
            text = _re.sub(r"(?<![a-zA-Z])'|'(?![a-zA-Z])", '"', text)
        # Truncated JSON: missing closing brace
        open_braces = text.count('{') - text.count('}')
        if open_braces > 0:
            text = text + '}' * open_braces
        open_brackets = text.count('[') - text.count(']')
        if open_brackets > 0:
            text = text + ']' * open_brackets
        return text.strip()
    
    def _parse_json_output(self, output_text: str, fields: List[str]) -> Dict[str, str]:
        """
        Parse JSON from model output text with automatic repair.
        
        Pipeline:
          1. Try direct json.loads()
          2. Try _repair_json() + json.loads()
          3. Extract from markdown code block (with repair)
          4. Extract from brace-matching regex (with repair)
          5. Fallback: key-value pattern matching from free text
        
        Uses fuzzy key matching when model returns slightly different keys.
        """
        result_dict = None
        
        # Tier 1: Try direct JSON parse first
        try:
            result = json.loads(output_text)
            if isinstance(result, dict):
                result_dict = result
        except json.JSONDecodeError:
            pass
        
        # Tier 2: Try repair + JSON parse
        if result_dict is None:
            try:
                repaired = self._repair_json(output_text)
                result = json.loads(repaired)
                if isinstance(result, dict):
                    result_dict = result
            except json.JSONDecodeError:
                pass
        
        # Tier 3: Try to extract JSON from markdown code block
        if result_dict is None:
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', output_text, re.DOTALL)
            if json_match:
                try:
                    repaired = self._repair_json(json_match.group(1))
                    result = json.loads(repaired)
                    if isinstance(result, dict):
                        result_dict = result
                except json.JSONDecodeError:
                    pass
        
        # Tier 4: Try to find any JSON-like object in the text
        if result_dict is None:
            brace_match = re.search(r'\{[^{}]*\}', output_text, re.DOTALL)
            if brace_match:
                try:
                    repaired = self._repair_json(brace_match.group(0))
                    result = json.loads(repaired)
                    if isinstance(result, dict):
                        result_dict = result
                except json.JSONDecodeError:
                    pass
        
        # If we got a JSON dict, do smart key matching then sanitize placeholders
        if result_dict is not None:
            matched = self._match_fields_to_keys(result_dict, fields)
            # Sanitize any placeholder values the model kept
            for key, val in matched.items():
                if isinstance(val, str) and val.strip() in _PLACEHOLDER_VALUES:
                    matched[key] = ""
            return matched
        
        # Tier 5: Last resort — extract field values from free text
        logger.warning(f"Could not parse JSON from Qwen2-VL output: {output_text[:200]}")
        results = {}
        for field in fields:
            pattern = re.compile(
                rf'["\']?{re.escape(field)}["\']?\s*[:=]\s*["\']?([^"\'\n,}}]+)',
                re.IGNORECASE
            )
            match = pattern.search(output_text)
            results[field] = match.group(1).strip() if match else ""
        
        return results
    
    def _match_fields_to_keys(self, result_dict: Dict, fields: List[str]) -> Dict[str, str]:
        """
        Match requested field names to model-returned JSON keys.
        
        Handles cases where the model returns slightly different key names
        (e.g. model returns "Phone" but we asked for "Phone Number").
        
        Matching priority:
            1. Exact match
            2. Case-insensitive match
            3. Substring containment (field in key or key in field)
        """
        matched = {}
        used_keys = set()
        
        # Pass 1: Exact match
        for field in fields:
            if field in result_dict:
                matched[field] = str(result_dict[field])
                used_keys.add(field)
        
        # Pass 2: Case-insensitive match for unmatched fields
        lower_map = {k.lower(): k for k in result_dict if k not in used_keys}
        for field in fields:
            if field in matched:
                continue
            if field.lower() in lower_map:
                orig_key = lower_map[field.lower()]
                matched[field] = str(result_dict[orig_key])
                used_keys.add(orig_key)
        
        # Pass 3: Substring containment for remaining unmatched fields
        remaining_keys = {k: v for k, v in result_dict.items() if k not in used_keys}
        for field in fields:
            if field in matched:
                continue
            field_lower = field.lower()
            for key, value in remaining_keys.items():
                key_lower = key.lower()
                if field_lower in key_lower or key_lower in field_lower:
                    matched[field] = str(value)
                    used_keys.add(key)
                    break
        
        # Fill any still-missing fields with empty string
        for field in fields:
            if field not in matched:
                matched[field] = ""
        
        return matched


