# PDF AI — Intelligent Document Extraction

> Upload a PDF form → AI extracts structured data → Returns clean JSON.

## Features

- **Field Extraction** — Specify fields, AI reads the document and extracts values
- **Document Q&A** — Chat panel: ask natural language questions about any PDF
- **Smart Scan** — Auto-detect what fields exist in a document
- **Checkbox Detection** — Detect checked/unchecked items in forms
- **HITL Review** — Flag uncertain fields for human review
- **Field Presets** — Save/load commonly used field sets
- **Majority Voting** — Run extraction N times, take consensus (Qwen only)
- **Field Resend** — Re-extract individual fields with one click
- **Image Enhancement** — Adaptive thresholding, border removal, smart crop
- **In-Memory Cache** — Processed images cached per PDF (faster Q&A)

## Quick Start

### Docker (Recommended)

```bash
docker compose build
docker compose up -d
```

### Local Development

```bash
.\start.bat          # Windows
./start.sh           # Linux
```

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Architecture

```
PDF → Image Enhancement → VLM Inference → Validation → JSON
         ↑ cached                ↑ GPU semaphore
```

### Models

| Model             | VRAM           | Accuracy   | Auto-Selected When |
| :---------------- | :------------- | :--------- | :----------------- |
| **Qwen2.5-VL-7B** | ~6GB (4-bit)   | 95% DocVQA | VRAM ≥ 10GB        |
| **Qwen2.5-VL-3B** | ~2GB (4-bit)   | 92% DocVQA | VRAM ≥ 3.5GB       |
| **Qwen2-VL-2B**   | ~1.5GB (4-bit) | 88% DocVQA | Fallback           |
| **PaddleOCR-VL**  | ~2GB           | Good       | Manual selection   |

Model auto-selects based on GPU VRAM. Override with env var:

```yaml
environment:
  - QWEN_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
```

## API Endpoints

| Method          | Endpoint                 | Description                   |
| :-------------- | :----------------------- | :---------------------------- |
| POST            | `/api/extract`           | Extract fields from PDF       |
| POST            | `/api/ask`               | Document Q&A (chat)           |
| POST            | `/api/re-extract`        | Re-extract single field       |
| POST            | `/api/detect-checkboxes` | Detect checkboxes             |
| POST            | `/api/smart-scan`        | Auto-detect fields            |
| GET             | `/api/health`            | Backend health + model status |
| GET/POST/DELETE | `/api/reviews/*`         | HITL review queue             |

## Project Structure

```
PDF AI/
├── backend/
│   ├── server.py                  FastAPI server (all endpoints)
│   ├── config.py                  VRAM-based model auto-selection
│   ├── hitl_manager.py            Review queue manager
│   ├── training_collector.py      HITL training data collector
│   ├── train_qwen2vl.py           LoRA fine-tuning script
│   ├── models/
│   │   ├── qwen2vl_extractor.py   Qwen2.5-VL extractor (core)
│   │   ├── paddleocr_extractor.py PaddleOCR-VL extractor
│   │   └── validators.py         Field validation & normalization
│   └── utils/
│       ├── pdf_processor.py       PDF → image conversion
│       └── image_enhancer.py      Image preprocessing pipeline
├── src/
│   ├── index.js                   Main frontend UI
│   ├── components/
│   │   ├── chat.js                Document Q&A panel
│   │   ├── icons.js               SVG icon library
│   │   └── review_queue.js        HITL review UI
│   ├── services/api/
│   │   └── backend.js             API client (all fetch calls)
│   └── styles/
│       └── global.css             All application styles
├── Doc/
│   └── guide.txt                  Comprehensive project guide
├── Form/                          Test PDF forms
├── Dockerfile                     Multi-stage Docker build
├── docker-compose.yml             Container config + GPU
└── start.bat                      Local launcher (Windows)
```

## Docker Volumes

| Volume          | Purpose                   | Size     |
| :-------------- | :------------------------ | :------- |
| `model-cache`   | HuggingFace model weights | 2-8GB    |
| `training-data` | HITL training samples     | Variable |
| `adapters`      | Trained LoRA adapters     | Future   |

## Notes

- **Model Updates**: `HF_HUB_OFFLINE=1` is set in `docker-compose.yml` to skip HuggingFace server checks on startup (faster boot). To update the model to a newer version, temporarily remove this line, restart the container to re-download, then add it back.
- **GPU**: Only 1 inference at a time (semaphore). If GPU is busy, requests queue up to 300s.
- **First Query**: First Q&A query takes ~30-45s (model warmup + image processing). Subsequent queries are faster due to caching.

## Requirements

- Python 3.10+
- Node.js 18+
- NVIDIA GPU (4GB+ VRAM)
- Docker Desktop + NVIDIA Container Toolkit (for Docker deployment)
