"""
Qwen2.5-VL QLoRA Fine-tuning Script

Standalone training script that uses HITL-collected training data
to fine-tune Qwen2.5-VL with QLoRA for improved document extraction.

Usage:
    # Export training data and show stats
    python train_qwen2vl.py --stats-only

    # Export + run training
    python train_qwen2vl.py

    # Custom options
    python train_qwen2vl.py --epochs 5 --lr 1e-5 --output adapters/custom

Requirements:
    pip install peft trl accelerate bitsandbytes

Data format:
    Uses the official Qwen2.5-VL training format:
    [{"image": "path.png", "conversations": [{"from": "human", ...}, {"from": "gpt", ...}]}]

References:
    - Qwen2.5-VL fine-tuning: https://github.com/qwenlm/qwen2.5-vl/blob/main/qwen-vl-finetune/README.md
    - PEFT LoRA docs: https://huggingface.co/docs/peft/main/en/conceptual_guides/lora
    - TRL SFTTrainer: https://huggingface.co/docs/trl/main/en/sft_trainer
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-VL with QLoRA using HITL training data"
    )
    parser.add_argument(
        "--data-dir", type=str, default="training_data",
        help="Directory containing training data (default: training_data)"
    )
    parser.add_argument(
        "--output", type=str, default="adapters/qwen2vl_lora",
        help="Output directory for LoRA adapter (default: adapters/qwen2vl_lora)"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Per-device batch size (default: 1, GPU constrained)"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=8,
        help="Gradient accumulation steps (default: 8, effective batch = batch_size * grad_accum)"
    )
    parser.add_argument(
        "--lora-r", type=int, default=8,
        help="LoRA rank (default: 8, from Qwen2.5-VL official config)"
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=16,
        help="LoRA alpha (default: 16, from Qwen2.5-VL official config)"
    )
    parser.add_argument(
        "--max-length", type=int, default=2048,
        help="Max sequence length (default: 2048)"
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="Only show training data stats, don't train"
    )
    parser.add_argument(
        "--export-only", action="store_true",
        help="Only export training data to Qwen2.5-VL format, don't train"
    )
    return parser.parse_args()


def show_stats():
    """Show training data statistics."""
    from training_collector import get_training_collector
    collector = get_training_collector()
    stats = collector.get_stats()

    print("\n" + "=" * 60)
    print("📊 Training Data Statistics")
    print("=" * 60)
    print(f"  Total samples:      {stats['total_samples']}")
    print(f"  Total corrections:  {stats['total_corrections']}")
    print(f"  Total approvals:    {stats['total_approvals']}")
    print(f"  Unique documents:   {stats['unique_documents']}")
    print(f"  Unique fields:      {stats['unique_fields']}")
    print(f"  Last export:        {stats['last_export'] or 'Never'}")
    print(f"  Ready for training: {'✅ Yes' if stats['ready_for_training'] else '❌ Not yet'}")
    print(f"\n  💡 {stats['recommendation']}")

    if stats['files']:
        print(f"\n  Documents:")
        for fname, count in stats['files'].items():
            print(f"    • {fname}: {count} samples")

    if stats['most_corrected_fields']:
        print(f"\n  Most corrected fields:")
        for field in stats['most_corrected_fields']:
            print(f"    • {field}")

    print("=" * 60 + "\n")
    return stats


def export_data(data_dir: str) -> str:
    """Export training data to Qwen2.5-VL format."""
    from training_collector import get_training_collector
    collector = get_training_collector()
    output_path = collector.export_for_training()
    return output_path


def train(args):
    """Run QLoRA fine-tuning."""
    import torch

    # ─── Pre-flight checks ───
    if not torch.cuda.is_available():
        print("❌ CUDA not available. QLoRA training requires a GPU.")
        print("   Options: Colab, RunPod, Lambda Labs, or a machine with NVIDIA GPU.")
        sys.exit(1)

    vram_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB VRAM)")

    if vram_gb < 4.0:
        print("⚠️  Less than 4GB VRAM. Training may fail with OOM errors.")
        print("   Consider using Google Colab (free T4 has 16GB).")

    # ─── Load training data ───
    train_file = Path(args.data_dir) / "qwen2vl_train.json"
    if not train_file.exists():
        print(f"❌ Training data not found at {train_file}")
        print("   Run: python train_qwen2vl.py --export-only")
        sys.exit(1)

    with open(train_file, "r", encoding="utf-8") as f:
        training_data = json.load(f)

    print(f"📦 Training data: {len(training_data)} samples from {train_file}")

    if len(training_data) < 5:
        print("⚠️  Very few samples. Results may be unreliable.")
        print("   Recommend at least 20 samples for meaningful improvement.")

    # ─── Import heavy dependencies ───
    print("📥 Loading libraries...")
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig
    from config import QWEN2VL_MODEL

    # Use correct model class
    if "Qwen2.5" in QWEN2VL_MODEL:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    else:
        from transformers import Qwen2VLForConditionalGeneration as ModelClass

    # ─── Load base model with 4-bit quantization ───
    print(f"📥 Loading base model: {QWEN2VL_MODEL}...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = ModelClass.from_pretrained(
        QWEN2VL_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    processor = AutoProcessor.from_pretrained(QWEN2VL_MODEL)
    tokenizer = processor.tokenizer

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"   ✅ Base model loaded ({torch.cuda.memory_allocated() / 1024**2:.0f} MB VRAM)")

    # ─── Apply LoRA ───
    print(f"🔧 Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ─── Prepare dataset ───
    print("📋 Preparing dataset...")

    def format_sample(sample):
        """Convert a training sample to the format SFTTrainer expects."""
        conversations = sample["conversations"]
        # Build chat-style text
        text = ""
        for msg in conversations:
            role = "user" if msg["from"] == "human" else "assistant"
            content = msg["value"]
            # Remove <image> tag from text (handled separately by processor)
            content = content.replace("<image>\n", "")
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return {"text": text}

    formatted_data = [format_sample(s) for s in training_data]

    from datasets import Dataset
    dataset = Dataset.from_list(formatted_data)

    print(f"   ✅ Dataset ready: {len(dataset)} samples")

    # ─── Training config ───
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        max_seq_length=args.max_length,
        gradient_checkpointing=True,  # Critical for 4GB VRAM
        optim="adamw_torch",
        report_to="none",  # Disable wandb/tensorboard
    )

    # ─── Train ───
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Batch size: {args.batch_size} (effective: {args.batch_size * args.grad_accum})")
    print(f"   LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"   Output: {output_dir}")
    print()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    train_result = trainer.train()

    # ─── Save adapter ───
    print(f"\n💾 Saving LoRA adapter to {output_dir}...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save training metadata
    train_meta = {
        "trained_at": datetime.now().isoformat(),
        "base_model": QWEN2VL_MODEL,
        "training_samples": len(training_data),
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "train_loss": train_result.training_loss,
        "total_steps": train_result.global_step,
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(train_meta, f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"   Final loss: {train_result.training_loss:.4f}")
    print(f"   Total steps: {train_result.global_step}")
    print(f"   Adapter saved to: {output_dir}")
    print(f"\n   The adapter will be automatically loaded on next server start.")


def main():
    args = parse_args()

    # Always show stats first
    stats = show_stats()

    if args.stats_only:
        return

    # Export training data
    print("📤 Exporting training data to Qwen2.5-VL format...")
    export_path = export_data(args.data_dir)
    if not export_path:
        print("❌ No training data to export. Collect more corrections first.")
        return

    print(f"   ✅ Exported to {export_path}")

    if args.export_only:
        return

    # Check minimum samples
    if stats["total_samples"] < 5:
        print(f"\n⚠️  Only {stats['total_samples']} samples. Need at least 5 to train.")
        print("   Keep extracting documents and correcting errors.")
        return

    # Run training
    train(args)


if __name__ == "__main__":
    main()
