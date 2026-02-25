# PDF AI — Deep Improvement Roadmap

> Context: OCR didn't work well (switched to VLM), confidence scores were broken (to be fixed), HITL review queue exists, currently building table/checkbox/layout detection. Next: fine-tuning.

---

## 🧠 The Fundamental Problem with Pure VLM

Your system sends the **full page image** to VLM for every extraction. This is like hiring a PhD to read every line of a form — powerful but slow and sometimes overthinks simple things.

**What the best systems do:** They use VLM as the **brain**, not the **eyes**. Separate the two:

```
EYES (fast, cheap)              BRAIN (slow, expensive)
├── PaddleOCR text + boxes      ├── Handwriting recognition
├── Layout detection regions    ├── Complex table parsing
├── Checkbox pixel analysis     ├── Ambiguous field resolution
└── Simple field matching       └── Cross-field reasoning
```

---

## 🔬 7 Techniques to Dramatically Improve Accuracy

### 1. Self-Verification Loop (Biggest Win, Easiest to Build)

Instead of extracting once and hoping it's right, **extract → verify → re-extract if wrong**:

```
Step 1: VLM extracts "Date: 01/01/2024"
Step 2: VLM re-reads the image with a NEW prompt:
        "The extracted value for 'Date' is '01/01/2024'.
         Look at the document — is this correct?
         If wrong, what is the actual value?"
Step 3: If VLM says "correct" → keep it
        If VLM says "wrong, it's 01/02/2024" → use corrected value
```

**Why it works:** The VLM is better at _verifying_ answers than _generating_ them from scratch. It's like how you catch more typos when proofreading someone else's work.

**Cost:** ~2x inference time, but accuracy jumps significantly.

**Research shows:** 15-20% accuracy improvement with self-verification.

---

### 2. Prompt Chaining (Break Complex → Simple)

Instead of one mega-prompt "extract all 15 fields from this page", break it into steps:

```
Chain 1: "What type of document is this?" → "Invoice"
Chain 2: "List ALL field labels visible in this invoice" → ["Invoice To", "Date", ...]
Chain 3: "For the field 'Invoice To', what is the value?" → "JOHANNA DOE"
Chain 4: "Verify: Is 'JOHANNA DOE' correct for 'Invoice To'?" → "Yes"
```

You already do batch → per-field fallback. The improvement: add **step 0 (classify)** and **step N+1 (verify)**.

---

### 3. Visual Grounding (Know WHERE the answer came from)

Current: VLM says `Date = "01/01/2024"` but you don't know WHERE in the image it found this.

**Upgrade:** Ask VLM to also return the **bounding box coordinates** of the answer:

```json
{
  "field": "Date",
  "value": "01/01/2024",
  "bbox": [420, 180, 580, 200],
  "confidence": "high"
}
```

**Qwen2.5-VL natively supports this** — it can output bounding boxes. You just need to ask:

> "Extract the value and the bounding box coordinates of where you found it."

**Why it matters:**

- You can SHOW the user exactly where the value was found (highlight on PDF)
- You can VERIFY it against OCR text in that region
- You can BUILD templates from bounding boxes for future extractions
- You can DETECT errors (if bbox is in the wrong area of the page)

---

### 4. Hybrid OCR + VLM (Speed + Accuracy)

Your OCR "didn't work" — but that doesn't mean it's useless. OCR is perfect for **validation and cross-checking**:

```
┌─ PaddleOCR extracts ALL text + positions (fast, 1-2s) ─┐
│  "JOHANNA DOE" at (150, 200)                            │
│  "01/01/2024" at (420, 180)                             │
│  "$10" at (500, 350)                                    │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─ VLM extracts field values (slow, 15-30s) ──────────────┐
│  Date = "01/01/2024"                                     │
│  Invoice To = "JOHANNA DOE"                              │
└──────────────────────────────────────────────────────────┘
                    ↓
┌─ Cross-check: Does VLM answer exist in OCR text? ───────┐
│  "01/01/2024" found in OCR? → YES ✅ High confidence     │
│  "JOHANNA DOE" found in OCR? → YES ✅ High confidence    │
│  "Handwritten signature" in OCR? → NO → VLM-only 📐     │
└──────────────────────────────────────────────────────────┘
```

**This gives you REAL confidence scores** — not fake ones. If VLM and OCR agree = high confidence. If they disagree = flag for review.

---

### 5. Spatial-Aware Field Matching

Instead of asking VLM "what is the value of Date?", tell it WHERE to look:

```
"In this document, the field label 'Date' appears at the
top-right area. What value is written next to or below
the label 'Date'?"
```

**Better yet:** Run OCR first to find WHERE "Date:" appears in the document, then crop that region and send ONLY that region to the VLM. This:

- Reduces image size → faster inference
- Eliminates confusion with other dates in the document
- Dramatically improves accuracy for specific fields

This is what **AWS Textract** and **Google Document AI** do internally.

---

### 6. Ensemble Extraction (Multiple Models Agree)

Run the same extraction through 2-3 different approaches and take the consensus:

```
Approach A: Full-page VLM extraction → "Date: 01/01/2024"
Approach B: Region-cropped VLM extraction → "Date: 01/01/2024"
Approach C: OCR text near "Date" label → "Date: 01/01/2024"

All 3 agree → 99% confidence ✅
2/3 agree → 80% confidence, flag for review
All disagree → Low confidence, manual review ⚠️
```

You already have voting rounds — this extends it across METHODS, not just temperature.

---

### 7. Document Memory (Learn from Past Extractions)

When the user extracts from `investment.pdf` type documents repeatedly:

```
Extraction 1: User corrects "Bank Name" → "Kotak Mahindra Bank" (was wrong)
Extraction 2: System remembers the correction → includes in prompt:
              "In similar documents, 'Bank Name' was found in the
               middle-left area near 'Purchase Request' section"
Extraction 3: System nails it first try → no correction needed
```

Store extraction history per document type. After 3-5 successful extractions, the system "knows" the layout and can guide the VLM with spatial hints.

---

## 🆚 What ChatPDF / PDF.ai Do Differently

| Feature            | ChatPDF/PDF.ai                         | Your PDF AI                          |
| ------------------ | -------------------------------------- | ------------------------------------ |
| **Core approach**  | RAG: PDF → text chunks → embed → query | VLM: PDF → image → visual extraction |
| **Table handling** | Parse text-layer tables only           | VLM reads tables from image          |
| **Handwriting**    | ❌ Cannot read                         | ✅ VLM reads handwriting             |
| **Checkboxes**     | ❌ Cannot detect                       | ✅ Visual detection                  |
| **Scanned PDFs**   | OCR first, then text search            | ✅ VLM reads directly                |
| **Multi-file**     | ✅ Compare across docs                 | ❌ Single doc                        |
| **Chat interface** | ✅ Conversational Q&A                  | ✅ Q&A + structured extraction       |

**Your advantage:** You handle the HARD cases (handwriting, checkboxes, scanned forms) that ChatPDF/PDF.ai simply cannot. They're text-search tools; you're a visual understanding tool.

**Their advantage:** Speed (text search is instant) and multi-document analysis.

---

## 🎯 Priority Roadmap (What to Build Next)

| #   | Feature                                      | Effort    | Impact   | Why                                      |
| --- | -------------------------------------------- | --------- | -------- | ---------------------------------------- |
| 1   | **Self-verification loop**                   | 2-3 days  | 🔥🔥🔥   | Biggest accuracy gain, simple to add     |
| 2   | **OCR cross-validation** (real confidence)   | 2-3 days  | 🔥🔥🔥   | Replaces broken confidence scores        |
| 3   | **Visual grounding** (bbox in response)      | 1-2 days  | 🔥🔥     | Enables template learning + UI highlight |
| 4   | **Region cropping** (spatial extraction)     | 3-5 days  | 🔥🔥🔥   | Speed + accuracy for known layouts       |
| 5   | **Fine-tuning Qwen** on your doc types       | 1-2 weeks | 🔥🔥🔥🔥 | Model learns YOUR documents              |
| 6   | **Document memory** (learn from corrections) | 1 week    | 🔥🔥     | Gets smarter over time                   |
| 7   | **Multi-document batch** + CSV export        | 3-5 days  | 🔥🔥     | Enterprise killer feature                |

---

## 💡 The Ultimate Architecture

```
┌──────────────────────────────────────────────────────────┐
│  PDF Upload                                               │
│  ↓                                                        │
│  Page → Image (you have this)                             │
│  ↓                                                        │
│  ┌──────────────┐  ┌──────────────┐                       │
│  │ PaddleOCR    │  │ Layout Det.  │  ← EYES (parallel)    │
│  │ text + bbox  │  │ regions      │                       │
│  └──────┬───────┘  └──────┬───────┘                       │
│         └────────┬────────┘                               │
│                  ↓                                        │
│  ┌─ Smart Router ─────────────────────────────┐           │
│  │ Simple text field? → OCR lookup (instant)  │           │
│  │ Checkbox? → Pixel analysis (fast)          │           │
│  │ Table? → VLM table extraction              │           │
│  │ Handwriting? → VLM with region crop        │           │
│  │ Ambiguous? → Full VLM + self-verify        │           │
│  └────────────────────────────────────────────┘           │
│                  ↓                                        │
│  Cross-validate OCR ↔ VLM → confidence score              │
│                  ↓                                        │
│  Low confidence? → HITL review queue                      │
│  High confidence? → Auto-approve                          │
└──────────────────────────────────────────────────────────┘
```

**This architecture is what Google Document AI and AWS Textract use internally.** The difference: they have billion-dollar models. You have a 2B parameter Qwen running locally — which is actually impressive and more private.

---

## 📋 Implementation Plan (Phased by Conversation)

### ✅ Conversation /2 — Current (Table/Checkbox/Layout)

> **Rule: Don't touch what's working. Finish current features first.**

- [x] Smart table extraction (full table, column, row modes)
- [x] Table pre-scan (detect table structure before extraction)
- [x] Proactive field routing (column headings → column mode, SL.N → row mode)
- [x] Frontend mini-table rendering with sticky headers
- [x] Checkbox detection (always-on)
- [ ] Fix remaining row extraction edge cases (SL.3, SL.7)
- [ ] Test table extraction across all document types (invoice, valuation, CPL, ORD)

---

### 🔜 Conversation /3 — Accuracy & Intelligence Boost

**Phase 3A: Self-Verification Loop** (2-3 days, no new dependencies)

```
Extract → Verify → Correct
VLM extracts value → VLM re-checks with verification prompt → keep or correct
```

- Add `_self_verify(image, field, value)` method to extractor
- For each extracted field, run verification pass
- If VLM says "wrong", use corrected value
- Track `verified: true/false` in signals
- **Expected impact:** 15-20% accuracy improvement

**Phase 3B: Visual Grounding / Bounding Boxes** (1-2 days)

```
VLM returns: {"value": "01/01/2024", "bbox": [420, 180, 580, 200]}
```

- Modify extraction prompts to request bbox coordinates
- Qwen2.5-VL supports this natively
- Store bbox in signals for each field
- Frontend: highlight source region on PDF preview
- **Enables:** template learning, error detection, UI source highlighting

---

### 🔮 Conversation /4+ — Hybrid & Learning

**Phase 4A: OCR Cross-Validation** (2-3 days)

- Run PaddleOCR in parallel (boxes + text, even if text is garbled)
- Fuzzy-match VLM values against OCR text → real confidence scores
- VLM + OCR agree → high confidence | disagree → flag for HITL
- Replace broken confidence system with genuine signal

**Phase 4B: Region Cropping** (3-5 days)

- Use OCR bounding boxes to detect text regions
- Crop relevant region → feed smaller image to VLM
- Faster inference + higher accuracy for specific fields
- Especially useful for dense, multi-section forms

**Phase 4C: Fine-Tuning Qwen** (1-2 weeks)

- Collect extraction + correction data from production use
- Fine-tune Qwen2-VL on YOUR specific document types
- Model learns your documents' patterns, layouts, handwriting styles
- Biggest long-term accuracy improvement

**Phase 4D: Document Memory** (1 week)

- Store extraction history per document type
- After N successful extractions → build spatial template
- Skip VLM entirely for known templates → instant extraction
