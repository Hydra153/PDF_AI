
import json
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HITLManager:
    """
    Manages the Human-in-the-Loop review queue.
    Stores items in a JSON file for simplicity and persistence.
    """
    
    def __init__(self, storage_path: str = "hitl_queue.json"):
        self.storage_path = Path(storage_path)
        self.queue: List[Dict] = []
        self._load_queue()
        
    def _load_queue(self):
        """Load queue from disk"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    self.queue = json.load(f)
                logger.info(f"Loaded {len(self.queue)} items from HITL queue")
            except Exception as e:
                logger.error(f"Failed to load HITL queue: {e}")
                self.queue = []
        else:
            self.queue = []
            
    def _save_queue(self):
        """Save queue to disk"""
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.queue, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save HITL queue: {e}")
            
    def add_item(self, 
                 filename: str, 
                 field_name: str, 
                 ai_value: Any, 
                 confidence: float, 
                 page_num: int = 1,
                 bbox: Optional[List[float]] = None,
                 context_image: Optional[str] = None) -> str:
        """
        Add an item to the review queue
        Returns: Item ID
        """
        item_id = str(uuid.uuid4())
        
        item = {
            "id": item_id,
            "created_at": datetime.now().isoformat(),
            "status": "pending",
            "filename": filename,
            "page_num": page_num,
            "field_name": field_name,
            "ai_value": ai_value,
            "confidence": round(confidence, 3),
            "bbox": bbox,
            "context_image": context_image  # Base64 string of the region
        }
        
        self.queue.append(item)
        self._save_queue()
        logger.info(f"Added item {item_id} to review queue (Conf: {confidence})")
        return item_id
        
    def get_pending_items(self) -> List[Dict]:
        """Get all items waiting for review"""
        return [item for item in self.queue if item["status"] == "pending"]
        
    def resolve_item(self, item_id: str, action: str, corrected_value: Any = None) -> bool:
        """
        Resolve an item (approve/correct)
        action: 'approve' or 'correct'
        """
        for item in self.queue:
            if item["id"] == item_id:
                item["status"] = "resolved"
                item["resolved_at"] = datetime.now().isoformat()
                item["action"] = action
                if action == "correct":
                    item["final_value"] = corrected_value
                else:
                    item["final_value"] = item["ai_value"]
                    
                self._save_queue()
                logger.info(f"Resolved item {item_id} with action {action}")
                return True
                
        logger.warning(f"Item {item_id} not found")
        return False
        
    def get_stats(self) -> Dict:
        """Get queue statistics"""
        total = len(self.queue)
        pending = len(self.get_pending_items())
        resolved = total - pending
        return {
            "total": total,
            "pending": pending,
            "resolved": resolved
        }
    
    def get_corrections(self) -> List[Dict]:
        """Get all corrected items (for training data)"""
        return [
            {
                "filename": item["filename"],
                "field_name": item["field_name"],
                "original_value": item["ai_value"],
                "corrected_value": item["final_value"],
                "was_corrected": item["action"] == "correct",
                "timestamp": item.get("resolved_at", "")
            }
            for item in self.queue 
            if item["status"] == "resolved"
        ]
    
    def export_training_data(self, output_path: str = None) -> str:
        """Export corrections in a format ready for model fine-tuning"""
        corrections = self.get_corrections()
        
        training_data = {
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "total_corrections": len(corrections),
            "corrections": corrections
        }
        
        if output_path is None:
            output_path = str(self.storage_path.parent / "training_data.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2)
        
        logger.info(f"Exported {len(corrections)} corrections to {output_path}")
        return output_path

# Global instance
_hitl_manager = None

def get_hitl_manager() -> HITLManager:
    global _hitl_manager
    if _hitl_manager is None:
        # Save in the backend directory
        storage_file = Path(__file__).parent.parent / "hitl_store.json"
        _hitl_manager = HITLManager(str(storage_file))
    return _hitl_manager
