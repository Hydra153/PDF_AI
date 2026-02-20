"""
PaddleOCR-VL 1.5 Extractor — Document parsing + field extraction

Pipeline:
    PDF → Image → PaddleOCR-VL 1.5 → Structured Markdown/HTML → Parse → {field: value} JSON

Key challenges solved:
  - HTML tables with <td> cells containing embedded "Label: Value"
  - Column-based sections: e.g. Patient | Primary Insurance | Secondary Insurance
    in the same row — subsequent rows map cells to sections by column position
  - Duplicate labels: "Name:" appears under Patient, Lab, Insurance sections
  - Combined fields: "Sex/DOB/Age: Female 01/02/1938 87 Years"

IMPORTANT: Runs on PaddlePaddle (not PyTorch). On 4GB GPU, only one model at a time.
"""
import os
import re
import gc
import html
import logging
import tempfile
import difflib
from typing import Dict, List, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# ─── Singleton Pipeline Cache ───
_paddleocr_pipeline = None


def get_paddleocr_pipeline():
    """Lazy-load PaddleOCR-VL 1.5 pipeline (singleton)."""
    global _paddleocr_pipeline
    if _paddleocr_pipeline is None:
        logger.info("🔄 Loading PaddleOCR-VL 1.5 pipeline...")
        from paddleocr import PaddleOCRVL
        _paddleocr_pipeline = PaddleOCRVL(device="gpu")
        logger.info("✅ PaddleOCR-VL 1.5 pipeline loaded")
    return _paddleocr_pipeline


def unload_paddleocr_pipeline():
    """Unload PaddleOCR-VL from GPU to free VRAM."""
    global _paddleocr_pipeline
    if _paddleocr_pipeline is not None:
        logger.info("🔄 Unloading PaddleOCR-VL 1.5 from GPU...")
        try:
            del _paddleocr_pipeline
        except Exception:
            pass
        _paddleocr_pipeline = None
        try:
            import paddle
            paddle.device.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        try:
            import paddle
            mem = paddle.device.cuda.memory_allocated()
            logger.info(f"   ✅ PaddleOCR-VL unloaded ({mem / 1024 / 1024:.0f} MB VRAM remaining)")
        except Exception:
            logger.info("   ✅ PaddleOCR-VL unloaded")


# ─── Known section header words ───
SECTION_HEADERS = {
    "patient", "primary insurance", "secondary insurance",
    "guarantor", "physician", "diagnosis codes",
    "specimens", "insurance", "billing", "provider",
    "subscriber", "employer",
}

# ─── Field aliases: maps user field names → preferred label patterns ───
FIELD_ALIASES = {
    # Section-scoped first, flat fallback last
    "patient name":     ["Patient > Name", "Name"],
    "patient address":  ["Patient > Address", "Address"],
    "patient phone":    ["Patient > Phone", "Phone", "Telephone"],
    "dob":              ["Patient > Sex/DOB/Age", "Patient > DOB", "DOB",
                         "Date of birth", "Sex/DOB/Age"],
    "date of birth":    ["Patient > Sex/DOB/Age", "Patient > DOB", "DOB",
                         "Date of birth", "Sex/DOB/Age"],
    "date":             ["Collected Date", "Signed Date", "Date", "Date of Service", "Order Date"],
    "client name":      ["Name"],       # first (flat) occurrence = lab/client
    "client address":   ["Address"],    # first (flat) occurrence = lab/client
    "client phone":     ["Phone"],      # first (flat) occurrence = lab/client
    "primary insurance name":    ["Primary Insurance > Type", "Primary Insurance > Name",
                                  "Insurance Company"],
    "primary insurance address": ["Primary Insurance > Address", "Insurance Address"],
    "fasting":          ["Collected Date", "Fasting"],
    "sex":              ["Patient > Sex/DOB/Age", "Sex", "Sex/DOB/Age"],
    "age":              ["Patient > Sex/DOB/Age", "Age", "Sex/DOB/Age"],
}


class PaddleOCRExtractor:
    """
    Extract field values from documents using PaddleOCR-VL 1.5.
    Same interface as Qwen2VLExtractor.
    """

    def __init__(self):
        self.pipeline = get_paddleocr_pipeline()
        self._last_signals: Dict[str, dict] = {}
        self._last_meta: Dict = {}

    def extract(
        self,
        image: Image.Image,
        words: list,
        boxes: list,
        fields: List[str],
        **kwargs,
    ) -> Dict[str, str]:
        """Extract field values from document image using PaddleOCR-VL."""
        import time
        t_start = time.time()

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                tmp_path = f.name
                image.save(f, format="PNG")

            logger.info("🔍 PaddleOCR-VL parsing document...")
            output = self.pipeline.predict(input=tmp_path)

            markdown_text = ""
            for res in output:
                if hasattr(res, 'markdown') and res.markdown:
                    md = res.markdown
                    if isinstance(md, dict):
                        markdown_text = md.get('markdown_texts', '')
                    elif isinstance(md, str):
                        markdown_text = md
                    else:
                        markdown_text = str(md)

                if not markdown_text:
                    try:
                        md = res['markdown']
                        if isinstance(md, dict):
                            markdown_text = md.get('markdown_texts', '')
                        elif isinstance(md, str):
                            markdown_text = md
                    except (KeyError, TypeError):
                        pass

                if not markdown_text:
                    try:
                        md_dir = tempfile.mkdtemp(prefix="paddle_md_")
                        res.save_to_markdown(save_path=md_dir)
                        for root, dirs, files in os.walk(md_dir):
                            for fname in files:
                                if fname.endswith(".md"):
                                    with open(os.path.join(root, fname), "r", encoding="utf-8") as mf:
                                        markdown_text = mf.read()
                                    break
                            if markdown_text:
                                break
                    except Exception as e:
                        logger.warning(f"⚠️ Could not save markdown: {e}")

                break

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        if not markdown_text.strip():
            logger.warning("⚠️ PaddleOCR-VL returned empty output")
            return {f: "" for f in fields}

        logger.info(f"📝 PaddleOCR-VL output: {len(markdown_text)} chars")

        results = self._extract_fields_from_content(markdown_text, fields)

        t_elapsed = time.time() - t_start
        logger.info(f"✅ PaddleOCR-VL extraction: {len(results)} fields in {t_elapsed:.1f}s")

        self._last_signals = {
            f: {
                "source": "paddleocr",
                "flags": [] if v else ["empty_value"],
                "detail": "PaddleOCR-VL extraction" if v else "PaddleOCR returned empty",
            }
            for f, v in results.items()
        }
        self._last_meta = {
            "extraction_model": "paddleocr-vl-1.5",
            "time_seconds": round(t_elapsed, 1),
            "markdown_length": len(markdown_text),
            "signals": dict(self._last_signals),
            "validation": {
                f: {
                    "is_valid": bool(v),
                    "raw": v,
                    "normalized": v,
                    "error": "" if v else "Empty value",
                }
                for f, v in results.items()
            },
        }
        return results

    # ──────────────────────────────────────────────────
    # Core parsing
    # ──────────────────────────────────────────────────

    def _extract_fields_from_content(
        self, content: str, fields: List[str]
    ) -> Dict[str, str]:
        all_pairs = self._parse_all_pairs(content)

        logger.info(f"🔑 Discovered {len(all_pairs)} label-value pairs:")
        for label, value in list(all_pairs.items())[:25]:
            logger.info(f"   '{label}': '{value[:80]}'")

        results = {}
        for field in fields:
            value = self._match_field(field, all_pairs)
            results[field] = value

        return results

    def _parse_all_pairs(self, content: str) -> Dict[str, str]:
        """Extract all key-value pairs with section awareness."""
        pairs = {}
        html_pairs = self._parse_html_tables(content)
        pairs.update(html_pairs)
        text_pairs = self._parse_labeled_lines(content)
        pairs.update(text_pairs)
        return pairs

    def _parse_html_tables(self, content: str) -> Dict[str, str]:
        """
        Parse HTML tables with column-based section tracking.

        Detects multi-section header rows like:
          <tr><td colspan="3">Patient</td><td colspan="2">Primary Insurance</td><td>Secondary Insurance</td></tr>

        After such a row, maps cell positions to sections:
          - cells at positions matching the "Patient" columns → Patient section
          - cells at positions matching "Primary Insurance" columns → Primary Insurance section
        """
        pairs = {}

        tables = re.findall(r'<table[^>]*>(.*?)</table>', content, re.DOTALL | re.IGNORECASE)

        for table_html in tables:
            rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL | re.IGNORECASE)

            # Column-to-section mapping: built from multi-header rows
            # Maps column index → section name
            col_sections: Dict[int, str] = {}
            simple_section = ""  # fallback for single-section headers

            for row_html in rows:
                # Parse raw <td> elements with their colspan
                raw_tds = re.findall(r'<td([^>]*)>(.*?)</td>', row_html, re.DOTALL | re.IGNORECASE)

                # Build cell list with column span info
                cells_with_span = []
                col_idx = 0
                for attrs, cell_html in raw_tds:
                    text = self._clean_cell(cell_html)
                    # Extract colspan
                    colspan_match = re.search(r'colspan="?(\d+)"?', attrs)
                    colspan = int(colspan_match.group(1)) if colspan_match else 1
                    cells_with_span.append({
                        'text': text,
                        'col_start': col_idx,
                        'col_end': col_idx + colspan - 1,
                        'colspan': colspan,
                    })
                    col_idx += colspan

                non_empty = [c for c in cells_with_span if c['text'].strip()]
                if not non_empty:
                    continue

                # Check for multi-section header row
                section_cells = [c for c in non_empty
                                 if c['text'].lower().strip() in SECTION_HEADERS]

                if len(section_cells) >= 2:
                    # Multi-section header row! Build column→section mapping
                    col_sections = {}
                    for sc in section_cells:
                        for ci in range(sc['col_start'], sc['col_end'] + 1):
                            col_sections[ci] = sc['text']
                    continue

                if len(section_cells) == 1 and len(non_empty) == 1:
                    # Single section header row
                    simple_section = section_cells[0]['text']
                    col_sections = {}  # Reset column mapping
                    continue

                # Process data cells — extract key-value pairs
                # Use index-based iteration to handle adjacent-cell pairing
                i = 0
                while i < len(cells_with_span):
                    cell = cells_with_span[i]
                    text = cell['text'].strip()
                    if not text:
                        i += 1
                        continue

                    # Determine section for this cell
                    section = ""
                    if col_sections:
                        section = col_sections.get(cell['col_start'], "")
                    elif simple_section:
                        section = simple_section

                    # Strategy A: Embedded "Label: Value" in same cell
                    kv = self._split_key_value(text)
                    if kv:
                        label, value = kv
                        self._add_pair(pairs, label, value, section)
                        i += 1
                        continue

                    # Strategy B: Label-only cell → next cell is value
                    # "Name:" or "Date of birth:" (ends with colon, no value)
                    if text.endswith(':') and len(text) < 60:
                        label = text[:-1].strip()
                        # Find next non-empty cell
                        j = i + 1
                        while j < len(cells_with_span) and not cells_with_span[j]['text'].strip():
                            j += 1
                        if j < len(cells_with_span):
                            value = cells_with_span[j]['text'].strip()
                            if label and value:
                                self._add_pair(pairs, label, value, section)
                                i = j + 1
                                continue

                    # Strategy C: Short title-case cell → next cell might be value
                    # e.g. <td>Name</td><td>John Doe</td> (no colon)
                    words = text.split()
                    if (1 <= len(words) <= 5 and text[0:1].isupper()
                            and i + 1 < len(cells_with_span)):
                        next_text = cells_with_span[i + 1]['text'].strip()
                        # Only pair if next cell doesn't also look like a label
                        if (next_text and not next_text.endswith(':')
                                and next_text.lower().strip() not in SECTION_HEADERS):
                            self._add_pair(pairs, text, next_text, section)
                            i += 2
                            continue

                    i += 1

        return pairs

    def _split_key_value(self, text: str) -> Optional[Tuple[str, str]]:
        """Split 'Label: Value' from text. Returns (label, value) or None."""
        match = re.match(r'^([^:]{1,50}):\s*(.+)$', text)
        if match:
            label = match.group(1).strip()
            value = match.group(2).strip()
            if label and value and not re.match(r'^(http|https|ftp)$', label, re.IGNORECASE):
                return label, value
        return None

    def _add_pair(self, pairs: Dict[str, str], label: str, value: str, section: str):
        """Add key-value pair with both section-scoped and flat versions."""
        label = label.strip().rstrip(':').strip()
        if not label or not value:
            return
        if section:
            scoped = f"{section} > {label}"
            pairs[scoped] = value
        if label not in pairs:
            pairs[label] = value

    def _parse_labeled_lines(self, content: str) -> Dict[str, str]:
        """Extract key-value pairs from plain text (outside HTML tables)."""
        pairs = {}
        text = re.sub(r'<table[^>]*>.*?</table>', '', content, flags=re.DOTALL | re.IGNORECASE)
        for line in text.replace("\r\n", "\n").split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Standard "Label: Value" pattern
            match = re.match(r'^(.+?)\s*[:：]\s*(.+)$', line)
            if match:
                label = match.group(1).strip()
                value = match.group(2).strip()
                if label and value and len(label) < 60:
                    if label not in pairs:
                        pairs[label] = value

            # Special: "signed by ... on MM/DD/YYYY" pattern
            date_match = re.search(
                r'signed\s+by\s+.+?\s+on\s+(\d{1,2}/\d{1,2}/\d{2,4})',
                line, re.IGNORECASE
            )
            if date_match:
                pairs['Signed Date'] = date_match.group(1)

        return pairs

    def _clean_cell(self, cell_html: str) -> str:
        """Clean HTML cell: strip tags, decode entities, normalize whitespace."""
        text = re.sub(r'<[^>]+>', '', cell_html)
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # ──────────────────────────────────────────────────
    # Field matching
    # ──────────────────────────────────────────────────

    def _match_field(self, field: str, pairs: Dict[str, str]) -> str:
        """
        Match requested field to discovered key-value pairs.

        Priority:
        1. Alias table (maps "Patient Name" → "Patient > Name")
        2. Exact match (case-insensitive)
        3. Section-scoped match
        4. Substring match
        5. Fuzzy match (cutoff=0.55)

        Post-processes composite fields (Sex/DOB/Age).
        """
        field_lower = field.lower().strip()
        field_clean = re.sub(r'[:\s]+$', '', field_lower)

        # 1. Alias lookup — try each alias in order, skip if extraction is empty
        if field_clean in FIELD_ALIASES:
            for alias in FIELD_ALIASES[field_clean]:
                for label, value in pairs.items():
                    if label.lower().strip() == alias.lower():
                        # Try composite extraction first
                        extracted = self._extract_from_composite(field_clean, label, value)
                        if extracted is not None:
                            if extracted:  # Non-empty → use it
                                return extracted
                            else:  # Empty string → skip to next alias
                                break
                        # No composite extraction needed → use raw value
                        if value:
                            return value
                        break  # Value is empty → try next alias

        # 2. Exact match
        for label, value in pairs.items():
            lc = re.sub(r'[:\s]+$', '', label.lower().strip())
            if lc == field_clean:
                return value

        # 3. Section-scoped match ("* > field_name")
        for label, value in pairs.items():
            ll = label.lower().strip()
            if " > " in ll:
                _, lp = ll.rsplit(" > ", 1)
                lp = re.sub(r'[:\s]+$', '', lp)
                if lp == field_clean:
                    return value

        # 4. Substring match
        for label, value in pairs.items():
            lc = re.sub(r'[:\s]+$', '', label.lower().strip())
            if " > " in lc:
                _, lc = lc.rsplit(" > ", 1)
            if field_clean in lc or lc in field_clean:
                extracted = self._extract_from_composite(field_clean, label, value)
                if extracted is not None:
                    return extracted
                return value

        # 5. Fuzzy match
        label_list = list(pairs.keys())
        clean_map = {}
        for l in label_list:
            cl = re.sub(r'[:\s]+$', '', l.strip())
            if " > " in cl:
                _, cl = cl.rsplit(" > ", 1)
            clean_map[cl] = l
        clean_labels = list(clean_map.keys())
        if clean_labels:
            matches = difflib.get_close_matches(field, clean_labels, n=1, cutoff=0.55)
            if matches:
                original = clean_map[matches[0]]
                value = pairs[original]
                extracted = self._extract_from_composite(field_clean, original, value)
                if extracted is not None:
                    return extracted
                return value

        return ""

    def _extract_from_composite(self, field: str, label: str, value: str) -> Optional[str]:
        """
        Extract sub-value from composite fields.

        "Sex/DOB/Age: Female 01/02/1938 87 Years"
          - field="dob" → "01/02/1938"
          - field="sex" → "Female"
          - field="age" → "87"

        "Collected Date: Fasting: Yes Room #:"
          - field="fasting" → "Yes"
          - field="date" → (date or empty)
        """
        label_clean = label.lower().strip()
        if " > " in label_clean:
            _, label_clean = label_clean.rsplit(" > ", 1)

        # Sex/DOB/Age composite
        if "sex/dob/age" in label_clean or "sex/dob" in label_clean:
            if "dob" in field or "date of birth" in field:
                m = re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', value)
                return m.group() if m else None
            elif "sex" in field or "gender" in field:
                m = re.match(r'(Male|Female|M|F)\b', value, re.IGNORECASE)
                return m.group() if m else None
            elif "age" in field:
                m = re.search(r'(\d+)\s*(?:Years?|yrs?|Y)', value, re.IGNORECASE)
                return m.group(1) if m else None

        # Fasting embedded in Collected Date
        if "fasting" in field:
            m = re.search(r'Fasting:\s*(Yes|No|Y|N)', value, re.IGNORECASE)
            return m.group(1) if m else None

        # Date field pointing to "Collected Date: Fasting: Yes Room #:" — extract date only
        if "date" in field and "fasting" in value.lower():
            m = re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', value)
            return m.group() if m else ""

        return None

    def get_last_signals(self) -> Dict[str, dict]:
        return self._last_signals

    def get_last_meta(self) -> Dict:
        return self._last_meta
