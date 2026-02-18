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
ENABLE_VALIDATORS = False

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
        
        _qwen2vl_model = ModelClass.from_pretrained(
            model_name,
            quantization_config=quantization_config if device == "cuda" else None,
            device_map="auto",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
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
        self._last_confidences = {}
        self._last_meta = {}  # stores confidence + validation per field
    
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
        self._last_confidences = {}
        
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
                    # Confidence based on agreement: 3/3 = 0.90, 2/3 = 0.80, 1/3 = 0.70
                    agreement = count / voting_rounds
                    self._last_confidences[field] = round(0.60 + (agreement * 0.30), 2)
                    
                    if len(counter) > 1:
                        voting_disagreed.add(field)
                        print(f"      🗳️ '{field}': {dict(counter)} → winner: '{winner}' ({count}/{voting_rounds})")
                    else:
                        print(f"      ✅ '{field}' = '{winner}' (unanimous {count}/{voting_rounds})")
                else:
                    all_results[field] = ""
                    self._last_confidences[field] = 0.0
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
                    
                    zoom_value, zoom_conf = self._zoom_extract_field(image, field)
                    
                    if zoom_value:
                        # Case-insensitive match against voting candidates
                        matched_candidate = candidates.get(zoom_value.lower())
                        if matched_candidate is not None:
                            # Zoomed result matches a voting candidate — use it
                            all_results[field] = matched_candidate
                            self._last_confidences[field] = max(zoom_conf, self._last_confidences[field])
                            if matched_candidate != voting_winner:
                                print(f"      🔍 Zoom overrides: '{field}' = '{voting_winner}' → '{matched_candidate}'")
                            else:
                                print(f"      ✅ Zoom confirms: '{field}' = '{zoom_value}'")
                                self._last_confidences[field] = min(self._last_confidences[field] + 0.05, 0.95)
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
                # Heuristic confidence for batch: non-empty = 0.75, empty = 0.0
                self._last_confidences[field] = 0.75 if value.strip() else 0.0
        
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
                value, confidence = self._extract_single_field(image, field)
                
                # Hallucination guard
                if value and not self._validate_answer(value):
                    logger.warning(f"Hallucination guard: '{field}' answer rejected ({len(value)} chars)")
                    print(f"      🚫 Hallucination guard — clearing")
                    value = ""
                    confidence = 0.0
                
                # Sanitize placeholders
                if value.strip() in _PLACEHOLDER_VALUES:
                    print(f"      ⚠️ Placeholder '{value}' sanitized to empty")
                    value = ""
                    confidence = 0.0
                
                # Replace batch result if per-field found something
                if value.strip():
                    all_results[field] = value
                    self._last_confidences[field] = confidence
                    print(f"      ✅ Per-field recovered: '{field}' = '{value}' (conf: {confidence:.3f})")
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
            "confidences": dict(self._last_confidences),
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
    ) -> Tuple[str, float]:
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
                    output_scores=True,            # For confidence computation
                    return_dict_in_generate=True,
                )
            
            # Decode text — strip input tokens from output
            input_len = inputs["input_ids"].shape[1]
            output_ids = outputs.sequences[:, input_len:]
            
            output_text = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            
            # Compute per-field confidence from token probabilities
            confidence = self._compute_confidence(outputs, input_len)
            
            # Handle NOT_FOUND and empty responses
            if not output_text or output_text.upper() in ("NOT_FOUND", "NOT FOUND", "N/A", "NONE"):
                return "", 0.0
            
            # Clean up: remove surrounding quotes if model wrapped the value
            value = output_text.strip().strip('"').strip("'").strip()
            
            return value, confidence
            
        except Exception as e:
            logger.error(f"Single-field extraction failed for '{field}': {e}")
            print(f"      ❌ Error: {e}")
            return "", 0.0
    
    def _zoom_extract_field(
        self, image: Image.Image, field: str
    ) -> Tuple[str, float]:
        """
        Multi-crop zoom extraction for a single field.
        
        Splits image into top/bottom halves (with 10% overlap) and extracts
        the field from each. Each half gives the model ~2× more pixels per
        area, resolving character-level ambiguities (e.g., 6 vs 0).
        
        Only one half will contain the field — returns that result.
        
        Returns:
            Tuple of (value: str, confidence: float)
        """
        w, h = image.size
        overlap = int(h * 0.10)  # 10% overlap to avoid splitting a field
        
        # Top half: 0 to 60% of height
        top_crop = image.crop((0, 0, w, h // 2 + overlap))
        # Bottom half: 40% to 100% of height
        bottom_crop = image.crop((0, h // 2 - overlap, w, h))
        
        crops = [("top", top_crop), ("bottom", bottom_crop)]
        best_value = ""
        best_conf = 0.0
        
        for crop_name, crop_img in crops:
            value, conf = self._extract_single_field(crop_img, field)
            
            # Hallucination guard
            if value and not self._validate_answer(value):
                value = ""
                conf = 0.0
            
            # Sanitize placeholders
            if value.strip() in _PLACEHOLDER_VALUES:
                value = ""
                conf = 0.0
            
            if value.strip() and conf > best_conf:
                best_value = value
                best_conf = conf
                print(f"      🔍 Zoom ({crop_name}): '{field}' = '{value}' (conf: {conf:.3f})")
        
        return best_value, best_conf
    
    def _compute_confidence(self, outputs, input_len: int) -> float:
        """
        Compute real per-field confidence from generation token probabilities.
        
        With per-field plain-text output, ALL generated tokens are value tokens
        (no JSON structure to dilute the signal). This gives accurate confidence.
        
        Method:
            1. compute_transition_scores → per-token log-probs
            2. Mean log-prob across all generated tokens
            3. Exponentiate to get confidence ∈ (0, 1]
            4. Low-token penalty: if any token has prob < 0.3, cap at 0.7
        
        Returns:
            float confidence in [0.0, 1.0]
        """
        try:
            if not hasattr(outputs, 'scores') or not outputs.scores:
                logger.warning("No scores in output — falling back to 0.50")
                return 0.50
            
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            
            log_probs = transition_scores[0]  # first (only) batch element
            
            if len(log_probs) == 0:
                return 0.50
            
            log_probs_float = log_probs.float().cpu()
            avg_log_prob = log_probs_float.mean().item()
            
            # Exponentiate to get confidence ∈ (0, 1]
            confidence = math.exp(avg_log_prob)
            confidence = round(min(1.0, max(0.0, confidence)), 3)
            
            # Low-token penalty: if any individual token probability < 0.3,
            # the model was likely guessing on that token → cap overall confidence
            token_probs = torch.exp(log_probs_float)
            min_token_prob = token_probs.min().item()
            if min_token_prob < 0.3 and confidence > 0.7:
                confidence = min(confidence, 0.7)
                print(f"      ⚠️ Low-token penalty: min token prob {min_token_prob:.3f} → capped at {confidence:.3f}")
            
            print(f"      🎯 Confidence: {confidence:.3f} (avg log-prob: {avg_log_prob:.4f}, {len(log_probs)} tokens, min_prob: {min_token_prob:.3f})")
            return confidence
            
        except Exception as e:
            logger.warning(f"Confidence computation failed ({e}) — falling back to 0.50")
            return 0.50
    
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
    
    def get_last_confidences(self) -> Dict[str, float]:
        """Return confidence scores from the last extraction."""
        return self._last_confidences
    
    def get_last_meta(self) -> dict:
        """Return full metadata (confidence + validation) from last extraction."""
        return self._last_meta
    
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
        # Try direct JSON parse
        try:
            result = json.loads(output_text)
            if isinstance(result, list):
                return [str(f).strip() for f in result if f]
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown code block
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', output_text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                if isinstance(result, list):
                    return [str(f).strip() for f in result if f]
            except json.JSONDecodeError:
                pass
        
        # Try finding JSON array in text
        bracket_match = re.search(r'\[.*?\]', output_text, re.DOTALL)
        if bracket_match:
            try:
                result = json.loads(bracket_match.group(0))
                if isinstance(result, list):
                    return [str(f).strip() for f in result if f]
            except json.JSONDecodeError:
                pass
        
        # Fallback: split comma-separated values
        fields = [f.strip().strip('"\'') for f in output_text.split(',')]
        return [f for f in fields if f and len(f) > 1]
    
    def _parse_json_output(self, output_text: str, fields: List[str]) -> Dict[str, str]:
        """
        Parse JSON from model output text.
        
        Handles cases where the model wraps JSON in markdown code blocks
        or includes extra text before/after the JSON.
        Uses fuzzy key matching when model returns slightly different keys.
        """
        result_dict = None
        
        # Try direct JSON parse first
        try:
            result = json.loads(output_text)
            if isinstance(result, dict):
                result_dict = result
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code block
        if result_dict is None:
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', output_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                    if isinstance(result, dict):
                        result_dict = result
                except json.JSONDecodeError:
                    pass
        
        # Try to find any JSON-like object in the text
        if result_dict is None:
            brace_match = re.search(r'\{[^{}]*\}', output_text, re.DOTALL)
            if brace_match:
                try:
                    result = json.loads(brace_match.group(0))
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
        
        # Last resort: try to extract field values from free text
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
