# PDF AI — Intelligent Document Extraction

> Upload a PDF form → Get structured data extracted via AI-powered document understanding.

## Quick Start

```bash
.\start.bat
```

- **Backend**: http://localhost:8000
- **Frontend**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs

## Architecture

```
PDF → Image → Preprocessing → OCR (Paddle|Surya) → AI Extraction → JSON
```

### OCR Methods

| Method        | Speed     | Accuracy | Use Case                |
| :------------ | :-------- | :------- | :---------------------- |
| **PaddleOCR** | ⚡ Fast   | Good     | Standard forms          |
| **Surya**     | 🐢 Slower | Best     | Complex layouts, tables |

### AI Extraction (Planned)

| Method          | Description                                         |
| :-------------- | :-------------------------------------------------- |
| **Document QA** | Pre-trained LayoutLM question answering             |
| **Ollama LLM**  | Local LLM (Phi-3/Qwen2.5) for structured extraction |

## Project Structure

```
PDF AI/
├── backend/
│   ├── server.py              # FastAPI server
│   ├── config.py              # Configuration
│   ├── hitl_manager.py        # HITL review queue
│   ├── train_layoutlm.py      # Model training
│   ├── training_collector.py   # Training data
│   └── models/
│       ├── paddleocr_extractor.py  # PaddleOCR
│       ├── surya_extractor.py      # Surya OCR
│       ├── layoutlmv3.py           # LayoutLMv3
│       ├── preprocessor.py         # Image preprocessing
│       └── validators.py           # Field validation
├── src/
│   ├── index.js               # Frontend UI
│   └── components/
│       └── review_queue.js    # HITL review UI
├── Form/                      # Test PDF forms
└── start.bat                  # Launcher
```

## Requirements

- Python 3.10+
- Node.js 18+
- NVIDIA GPU (4GB+ VRAM)
