# Checkbox Extraction — Design Document

## Problem

The system extracts text fields from PDFs but cannot detect physical checkbox elements (☐ ☑ ✓ ✗). Users need to know which checkboxes are checked/unchecked in medical and administrative forms.

## Approach: VLM-Only + Zoom Verification

Use Qwen2.5-VL to detect checkboxes visually — no new dependencies. For low-confidence results, crop and re-verify at higher resolution.

## Two Workflows

### 1. User-Specified Checkbox Fields

User adds a field name (e.g., "Fasting") and marks it as checkbox type. System extracts `true`/`false` instead of text.

### 2. Auto-Detect All Checkboxes

"Find All Checkboxes" button asks VLM to scan entire document and return all physical checkboxes with labels and status.

## Output Format (Single JSON, Two Sections)

```json
{
  "fields": {
    "Client Name": "ROCKY MOUNTAIN LABORATORIES",
    "DOB": "01/02/1938"
  },
  "checkboxes": [
    { "label": "Fasting", "checked": true, "confidence": 0.95 },
    { "label": "STAT", "checked": false, "confidence": 0.88 }
  ],
  "_meta": { "extraction_model": "qwen", "time_seconds": 45.2 }
}
```

Extends later with `"tables": [...]` for table extraction.

## VLM Prompts

**Auto-detect all:**

```
Look at this document image. Find ALL physical checkboxes (☐, ☑, ✓, ✗, filled/empty squares).
For each checkbox, return its label text and whether it is checked or unchecked.
Return ONLY valid JSON array: [{"label": "...", "checked": true/false}]
Do NOT include text that says "yes/no" — only physical checkbox elements.
```

**User-specified field:**

```
Look at the checkbox next to "{field_name}" in this document.
Is it checked or unchecked? Reply with ONLY: checked OR unchecked
```

## Zoom Verification

For checkboxes with confidence < 0.7:

1. Crop a region around the checkbox (using approximate position from VLM)
2. Re-ask the VLM at 2× resolution
3. Use the higher-confidence result

## UI Changes

- New "Document Analysis" section below extraction results
- "Find All Checkboxes" button (same pattern for "Find All Tables" later)
- Checkbox results render as cards with ☑/☐ icons
- Checkbox type toggle when adding fields

## Accuracy Gate

Do **not** push to GitHub until checkbox detection achieves 90%+ accuracy on test documents.
