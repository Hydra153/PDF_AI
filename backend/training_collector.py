"""
Training Data Collector for Qwen2.5-VL Fine-tuning

Automatically collects training samples when users correct extraction results.
Each sample stores the complete extraction context in Qwen2.5-VL conversation
format, ready for QLoRA fine-tuning.

Data flow:
    User corrects field → save_sample() stores image + conversation
    Export → qwen2vl_train.json in official Qwen2.5-VL format
    Train  → python train_qwen2vl.py

Storage:
    training_data/
    ├── images/          # Page images (PNG)
    ├── samples.jsonl    # One JSON object per line (full context)
    └── metadata.json    # Stats and export info
"""

import json
import uuid
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from PIL import Image

logger = logging.getLogger(__name__)


class TrainingDataCollector:
    """
    Collects and stores training data for Qwen2.5-VL QLoRA fine-tuning.

    Each sample contains:
    - The page image (saved as PNG)
    - All fields requested for this extraction
    - The complete extraction result (with corrections merged = ground truth)
    - What was corrected (original vs corrected value)
    - Review signals, model info, voting rounds

    Multipage-ready: page_num is tracked per sample with unique image naming.
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = Path(__file__).parent / "training_data"
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.samples_path = self.data_dir / "samples.jsonl"
        self.metadata_path = self.data_dir / "metadata.json"

        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self._load_metadata()

    def _load_metadata(self):
        """Load or initialize metadata."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = self._default_metadata()
        else:
            self.metadata = self._default_metadata()

    def _default_metadata(self) -> Dict:
        return {
            "created_at": datetime.now().isoformat(),
            "total_samples": 0,
            "total_corrections": 0,
            "total_approvals": 0,
            "files": {},
            "last_export": None,
        }

    def _save_metadata(self):
        """Save metadata to disk."""
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def save_sample(
        self,
        image: Image.Image,
        filename: str,
        page_num: int,
        fields_requested: List[str],
        extraction_results: Dict[str, str],
        corrections: Dict[str, Dict[str, str]],
        signals: Dict[str, dict],
        model_used: str = "qwen",
        voting_rounds: int = 1,
        user_id: str = "default",
    ) -> str:
        """
        Save a complete training sample with maximum context.

        Args:
            image: PIL Image of the document page
            filename: Source PDF filename (e.g. "CPL Form 2.pdf")
            page_num: Page number (1-indexed, multipage-ready)
            fields_requested: List of field names that were extracted
            extraction_results: Dict of {field: value} with corrections ALREADY merged
                                (this is the ground truth for training)
            corrections: Dict of corrections made, e.g.
                         {"Patient Name": {"original": "LENOORE", "corrected": "LENORE"}}
                         Empty dict if all values were approved
            signals: Dict of {field: signal_dict}
            model_used: "qwen" or "paddleocr"
            voting_rounds: Number of voting rounds used

        Returns:
            Sample ID (UUID string)
        """
        sample_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()

        # Save page image with unique name
        safe_name = filename.replace(" ", "_").replace(".pdf", "")
        image_filename = f"{safe_name}_p{page_num}_{sample_id}.png"
        image_path = self.images_dir / image_filename

        try:
            image.save(str(image_path), "PNG")
        except Exception as e:
            logger.error(f"Failed to save training image: {e}")
            return ""

        # Build sample record
        sample = {
            "id": sample_id,
            "timestamp": timestamp,
            "source_pdf": filename,
            "page_num": page_num,
            "image_path": str(image_path.relative_to(self.data_dir.parent)),
            "fields_requested": fields_requested,
            "model_used": model_used,
            "voting_rounds": voting_rounds,
            "extraction_results": extraction_results,
            "corrections": corrections,
            "signals": signals,
            "is_corrected": len(corrections) > 0,
            "user_id": user_id,
        }

        # Append to JSONL
        try:
            with open(self.samples_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to save training sample: {e}")
            return ""

        # Update metadata
        self.metadata["total_samples"] += 1
        if corrections:
            self.metadata["total_corrections"] += len(corrections)
        else:
            self.metadata["total_approvals"] += 1
        self.metadata["files"][filename] = self.metadata["files"].get(filename, 0) + 1
        self._save_metadata()

        logger.info(f"Saved training sample {sample_id} for {filename} p{page_num}")
        return sample_id

    def export_for_training(self, output_path: str = None) -> str:
        """
        Convert samples.jsonl → qwen2vl_train.json in official Qwen2.5-VL format.

        The official format (from Context7/Qwen2.5-VL docs):
        [
            {
                "image": "path/to/image.png",
                "conversations": [
                    {"from": "human", "value": "<image>\n...prompt..."},
                    {"from": "gpt", "value": "{...json results...}"}
                ]
            },
            ...
        ]

        The prompt mirrors exactly what the extractor sends during inference,
        so the model learns to produce the correct output for the same prompt.
        """
        if output_path is None:
            output_path = str(self.data_dir / "qwen2vl_train.json")

        samples = self._load_samples()
        if not samples:
            logger.warning("No training samples to export")
            return ""

        training_data = []
        for sample in samples:
            # Build the exact same prompt the extractor uses
            fields = sample["fields_requested"]
            field_list = ", ".join(fields)

            user_prompt = (
                f"<image>\n"
                f"Extract these fields from the document image and return as JSON: {field_list}. "
                f"Return a JSON object with exactly these keys. "
                f"For each field, find the matching label in the document and copy its value exactly as written. "
                f"Include complete multi-line values (e.g. full addresses with city, state, zip). "
                f"If a field is not found, set its value to empty string.\n\n"
                f"Example output format:\n"
                f'{{"Patient Name": "DOE, JOHN", "DOB": "01/15/1980", "Address": "123 MAIN ST, SPRINGFIELD, IL 62701"}}\n\n'
                f"Now extract from this document:"
            )

            # Ground truth = extraction_results (with corrections already merged)
            gpt_response = json.dumps(
                sample["extraction_results"], ensure_ascii=False
            )

            training_entry = {
                "image": sample["image_path"],
                "conversations": [
                    {"from": "human", "value": user_prompt},
                    {"from": "gpt", "value": gpt_response},
                ],
            }
            training_data.append(training_entry)

        # Save in official format
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)

        self.metadata["last_export"] = datetime.now().isoformat()
        self.metadata["last_export_count"] = len(training_data)
        self._save_metadata()

        logger.info(f"Exported {len(training_data)} training samples to {output_path}")
        print(f"📚 Exported {len(training_data)} training samples → {output_path}")
        return output_path

    def get_stats(self) -> Dict:
        """Get training data statistics."""
        self._load_metadata()
        samples = self._load_samples()

        # Count unique documents
        unique_docs = set()
        unique_fields = set()
        corrected_fields = set()
        per_user = {}  # user_id → sample count
        for s in samples:
            unique_docs.add(s.get("source_pdf", ""))
            unique_fields.update(s.get("fields_requested", []))
            for field in s.get("corrections", {}):
                corrected_fields.add(field)
            uid = s.get("user_id", "default")
            per_user[uid] = per_user.get(uid, 0) + 1

        return {
            "total_samples": len(samples),
            "total_corrections": self.metadata.get("total_corrections", 0),
            "total_approvals": self.metadata.get("total_approvals", 0),
            "unique_documents": len(unique_docs),
            "unique_fields": len(unique_fields),
            "most_corrected_fields": list(corrected_fields),
            "per_user": per_user,
            "files": self.metadata.get("files", {}),
            "last_export": self.metadata.get("last_export"),
            "ready_for_training": len(samples) >= 20,
            "recommendation": self._training_recommendation(len(samples)),
        }

    def _training_recommendation(self, count: int) -> str:
        """Provide guidance on when to train based on sample count."""
        if count == 0:
            return "No samples yet. Start extracting and correcting documents."
        elif count < 20:
            return f"{count}/20 samples. Keep correcting — need at least 20 for meaningful training."
        elif count < 50:
            return f"{count} samples. Can start training now for moderate improvement (+5-8%)."
        elif count < 100:
            return f"{count} samples. Good amount for strong improvement (+10-12%). Ready to train."
        else:
            return f"{count} samples. Excellent dataset. Training will yield optimal results (+12-15%)."

    def get_samples(self) -> List[Dict]:
        """Get all training samples."""
        return self._load_samples()

    def clear(self):
        """Clear all training data (use with caution)."""
        if self.samples_path.exists():
            self.samples_path.unlink()
        if self.images_dir.exists():
            shutil.rmtree(self.images_dir)
            self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = self._default_metadata()
        self._save_metadata()
        logger.info("Training data cleared")
        print("🗑️ Training data cleared")

    def _load_samples(self) -> List[Dict]:
        """Load all samples from JSONL file."""
        samples = []
        if self.samples_path.exists():
            with open(self.samples_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            samples.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        return samples


# ─── Global Singleton ───
_collector = None


def get_training_collector() -> TrainingDataCollector:
    """Get the global training data collector instance."""
    global _collector
    if _collector is None:
        _collector = TrainingDataCollector()
    return _collector
