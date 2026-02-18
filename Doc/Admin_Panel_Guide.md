# Admin Panel — Production Architecture Guide

Future planning document for the PDF AI Admin Panel. This is a standalone management interface, separate from the user-facing extraction app, designed for SaaS deployment.

> **Status:** Planning only. To be implemented after the main site is complete.
> **Depends on:** Authentication system, database, cloud storage — all built at deployment time.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Two-App Separation](#2-two-app-separation)
3. [Authentication & Authorization](#3-authentication--authorization)
4. [Database Schema](#4-database-schema)
5. [Cloud Storage (Images)](#5-cloud-storage-images)
6. [Admin Panel Pages](#6-admin-panel-pages)
7. [User App Changes](#7-user-app-changes)
8. [API Design](#8-api-design)
9. [Correction Review Pipeline](#9-correction-review-pipeline)
10. [Training Pipeline (Production)](#10-training-pipeline-production)
11. [Server Monitoring](#11-server-monitoring)
12. [Security Considerations](#12-security-considerations)
13. [Deployment Architecture](#13-deployment-architecture)
14. [Migration Path](#14-migration-path)
15. [Tech Stack Recommendations](#15-tech-stack-recommendations)

---

## 1. Architecture Overview

### Current (Single-User, Local)

```
┌──────────────────────────────────────────────┐
│                  Local Machine               │
│                                              │
│  ┌──────────┐    ┌──────────┐    ┌────────┐  │
│  │ Frontend │───>│ Backend  │───>│ Local   │  │
│  │ (Vite)   │    │ (FastAPI)│    │ Storage │  │
│  └──────────┘    └──────────┘    └────────┘  │
│                       │                      │
│                  ┌────┴────┐                  │
│                  │  GPU    │                  │
│                  │  Model  │                  │
│                  └─────────┘                  │
└──────────────────────────────────────────────┘
```

### Production (SaaS, Multi-Tenant)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLOUD INFRASTRUCTURE                          │
│                                                                             │
│  ┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐   │
│  │   USER APP       │     │   ADMIN PANEL     │     │   TRAINING         │   │
│  │   (Next.js)      │     │   (Next.js)       │     │   SERVICE          │   │
│  │                   │     │                   │     │   (Background)     │   │
│  │  • Extract PDFs   │     │  • User mgmt      │     │  • QLoRA training  │   │
│  │  • View results   │     │  • Training data   │     │  • Adapter storage │   │
│  │  • Correct fields │     │  • Model training  │     │  • Scheduled jobs  │   │
│  │  • Download JSON  │     │  • Server monitor  │     │                    │   │
│  └────────┬──────────┘     └────────┬──────────┘     └────────┬───────────┘   │
│           │                         │                          │               │
│           └────────────┬────────────┘                          │               │
│                        ▼                                       │               │
│              ┌──────────────────┐                              │               │
│              │   API GATEWAY    │◄─────────────────────────────┘               │
│              │   (FastAPI)      │                                               │
│              │                  │                                               │
│              │  • Auth (JWT)    │                                               │
│              │  • Rate limiting │                                               │
│              │  • Role checking │                                               │
│              └────────┬─────────┘                                               │
│                       │                                                         │
│         ┌─────────────┼──────────────┐                                          │
│         ▼             ▼              ▼                                          │
│  ┌────────────┐ ┌──────────┐ ┌────────────────┐                                │
│  │ PostgreSQL │ │  Redis   │ │  S3 / GCS      │                                │
│  │  Database  │ │  Queue   │ │  (Images +     │                                │
│  │            │ │  Cache   │ │   Adapters)    │                                │
│  └────────────┘ └──────────┘ └────────────────┘                                │
│                       │                                                         │
│                  ┌────┴────┐                                                    │
│                  │  GPU    │                                                    │
│                  │ Server  │                                                    │
│                  └─────────┘                                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Two-App Separation

### User App (what exists today, cleaned up)

```
Visible to: All authenticated users
URL:        app.pdfai.com
Purpose:    PDF extraction and field correction
```

**Features the user sees:**

- Upload PDF
- Select/add fields
- Extract data
- View results with confidence scores
- Correct fields (corrections logged, NOT auto-trained)
- Download JSON/CSV
- Presets management

**Features REMOVED from user app:**

- Fine-tune / Export Data button
- View Training Data panel
- Training status/stats
- Any reference to model training

### Admin Panel (new, separate)

```
Visible to: Admin users only
URL:        admin.pdfai.com
Purpose:    Full system management
```

**Features:**

- Dashboard (overview stats)
- User management (view, enable/disable, usage)
- Correction review queue (approve/reject before training)
- Training data management (view, delete, export)
- Model training (trigger, monitor, history)
- Server monitoring (GPU, memory, queue, active jobs)
- Extraction logs (who extracted what, when)
- System settings (confidence threshold, model selection)

---

## 3. Authentication & Authorization

### Role Model

```
ROLES:
  ├── super_admin    — Full access, can create other admins
  ├── admin          — All admin features except creating admins
  └── user           — Extract + correct only, no admin access
```

### Auth Flow

```
User App:
  1. User logs in (email + password, or OAuth)
  2. JWT issued with role: "user"
  3. Token sent with every API request
  4. Backend checks role before allowing access

Admin Panel:
  1. Admin logs in (same auth system, different login page)
  2. JWT issued with role: "admin" or "super_admin"
  3. Additional 2FA required for admin login
  4. Admin-only endpoints check for admin role
```

### JWT Token Structure

```json
{
  "sub": "user-uuid-12345",
  "email": "admin@company.com",
  "role": "admin",
  "tenant_id": "tenant-abc",
  "iat": 1708000000,
  "exp": 1708086400
}
```

### Middleware

```python
# Backend route protection (pseudocode)

def require_auth(role: str = "user"):
    """Decorator that checks JWT and role."""
    def decorator(func):
        async def wrapper(request):
            token = request.headers.get("Authorization")
            payload = verify_jwt(token)

            if payload["role"] not in ROLE_HIERARCHY[role]:
                raise HTTPException(403, "Insufficient permissions")

            request.state.user = payload
            return await func(request)
        return wrapper
    return decorator

# Usage:
@app.post("/api/extract")
@require_auth("user")          # Any authenticated user
async def extract_fields(): ...

@app.post("/api/training/approve")
@require_auth("admin")         # Admin only
async def approve_corrections(): ...

@app.post("/api/admin/users/disable")
@require_auth("super_admin")   # Super admin only
async def disable_user(): ...
```

---

## 4. Database Schema

### Why a Database

Currently everything is stored in JSON files and local disk. For SaaS:

- **Users** need persistent accounts
- **Corrections** need an audit trail with user attribution
- **Training samples** need approval state tracking
- **Extraction logs** need per-user history

### Recommended: PostgreSQL

```sql
-- ════════════════════════════════════════
--  USERS & AUTH
-- ════════════════════════════════════════

CREATE TABLE tenants (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name          VARCHAR(255) NOT NULL,           -- "Acme Medical Group"
    plan          VARCHAR(50) DEFAULT 'free',      -- free, pro, enterprise
    status        VARCHAR(20) DEFAULT 'active',    -- active, suspended, deleted
    created_at    TIMESTAMP DEFAULT NOW(),
    settings      JSONB DEFAULT '{}'               -- tenant-specific config
);

CREATE TABLE users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id     UUID REFERENCES tenants(id),
    email         VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role          VARCHAR(20) DEFAULT 'user',      -- user, admin, super_admin
    status        VARCHAR(20) DEFAULT 'active',    -- active, disabled, deleted
    display_name  VARCHAR(255),
    created_at    TIMESTAMP DEFAULT NOW(),
    last_login    TIMESTAMP,
    extraction_count INTEGER DEFAULT 0,            -- total extractions made
    correction_count INTEGER DEFAULT 0             -- total corrections made
);

CREATE INDEX idx_users_tenant ON users(tenant_id);
CREATE INDEX idx_users_email ON users(email);

-- ════════════════════════════════════════
--  EXTRACTIONS
-- ════════════════════════════════════════

CREATE TABLE extractions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID REFERENCES users(id),
    tenant_id     UUID REFERENCES tenants(id),
    filename      VARCHAR(500) NOT NULL,
    page_count    INTEGER DEFAULT 1,
    model_used    VARCHAR(50),                     -- qwen, paddleocr
    voting_rounds INTEGER DEFAULT 1,
    results       JSONB NOT NULL,                  -- {field: value}
    confidences   JSONB,                           -- {field: score}
    fields_requested TEXT[] NOT NULL,              -- ARRAY of field names
    duration_ms   INTEGER,                         -- extraction time
    status        VARCHAR(20) DEFAULT 'completed', -- completed, failed, timeout
    created_at    TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_extractions_user ON extractions(user_id);
CREATE INDEX idx_extractions_tenant ON extractions(tenant_id);
CREATE INDEX idx_extractions_created ON extractions(created_at DESC);

-- ════════════════════════════════════════
--  CORRECTIONS (user-submitted, pending review)
-- ════════════════════════════════════════

CREATE TABLE corrections (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    extraction_id   UUID REFERENCES extractions(id),
    user_id         UUID REFERENCES users(id),
    tenant_id       UUID REFERENCES tenants(id),
    field_name      VARCHAR(255) NOT NULL,
    original_value  TEXT,                           -- what the AI predicted
    corrected_value TEXT NOT NULL,                  -- what the user entered
    confidence      FLOAT,                         -- AI's confidence on this field

    -- Review state (admin controls this)
    review_status   VARCHAR(20) DEFAULT 'pending',  -- pending, approved, rejected
    reviewed_by     UUID REFERENCES users(id),     -- which admin reviewed
    reviewed_at     TIMESTAMP,
    rejection_reason TEXT,                          -- why admin rejected

    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_corrections_tenant ON corrections(tenant_id);
CREATE INDEX idx_corrections_status ON corrections(review_status);
CREATE INDEX idx_corrections_created ON corrections(created_at DESC);

-- ════════════════════════════════════════
--  TRAINING SAMPLES (admin-approved only)
-- ════════════════════════════════════════

CREATE TABLE training_samples (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID REFERENCES tenants(id),
    extraction_id   UUID REFERENCES extractions(id),
    image_storage_key VARCHAR(500) NOT NULL,        -- S3 key: "tenants/abc/images/xxx.png"
    fields_requested TEXT[] NOT NULL,
    ground_truth    JSONB NOT NULL,                 -- {field: corrected_value}
    corrections     JSONB DEFAULT '{}',             -- {field: {original, corrected}}
    confidences     JSONB,
    model_used      VARCHAR(50),
    is_corrected    BOOLEAN DEFAULT FALSE,

    -- Lineage
    approved_by     UUID REFERENCES users(id),
    approved_at     TIMESTAMP,
    source_correction_ids UUID[],                   -- which corrections contributed

    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_training_tenant ON training_samples(tenant_id);

-- ════════════════════════════════════════
--  TRAINING RUNS (model training history)
-- ════════════════════════════════════════

CREATE TABLE training_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID REFERENCES tenants(id),
    triggered_by    UUID REFERENCES users(id),     -- which admin started it
    status          VARCHAR(20) DEFAULT 'queued',  -- queued, running, completed, failed
    sample_count    INTEGER,
    epochs          INTEGER DEFAULT 3,
    learning_rate   FLOAT DEFAULT 2e-5,
    lora_rank       INTEGER DEFAULT 8,
    lora_alpha      INTEGER DEFAULT 16,

    -- Results
    final_loss      FLOAT,
    adapter_storage_key VARCHAR(500),               -- S3 key for saved adapter
    duration_seconds INTEGER,
    error_message   TEXT,                           -- if failed

    started_at      TIMESTAMP,
    completed_at    TIMESTAMP,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_runs_tenant ON training_runs(tenant_id);
CREATE INDEX idx_runs_status ON training_runs(status);

-- ════════════════════════════════════════
--  ACTIVE ADAPTERS (which adapter is live)
-- ════════════════════════════════════════

CREATE TABLE active_adapters (
    tenant_id       UUID PRIMARY KEY REFERENCES tenants(id),
    training_run_id UUID REFERENCES training_runs(id),
    adapter_storage_key VARCHAR(500) NOT NULL,       -- S3 key
    activated_at    TIMESTAMP DEFAULT NOW(),
    activated_by    UUID REFERENCES users(id)
);
```

### Entity Relationship Diagram

```
tenants ─────┬──── users
             │       │
             │       ├──── extractions
             │       │         │
             │       │         └──── corrections (pending review)
             │       │                    │
             │       │                    ▼ (admin approves)
             │       │              training_samples
             │       │                    │
             │       │                    ▼ (admin triggers)
             │       └──── training_runs
             │                    │
             └──── active_adapters ◄──────┘
```

---

## 5. Cloud Storage (Images)

### Why Cloud Storage

| Local (current)                   | Cloud (production)      |
| --------------------------------- | ----------------------- |
| Images in `training_data/images/` | Images in S3/GCS bucket |
| Single machine, one disk          | Unlimited, distributed  |
| No tenant isolation               | Prefixed by tenant ID   |
| Lost if server dies               | Durable, replicated     |

### S3 Bucket Structure

```
s3://pdfai-data/
├── tenants/
│   ├── tenant-abc/
│   │   ├── extractions/
│   │   │   ├── 2026-02-18/
│   │   │   │   ├── extraction-uuid-1-p1.png
│   │   │   │   └── extraction-uuid-2-p1.png
│   │   │   └── 2026-02-19/
│   │   │       └── ...
│   │   ├── training/
│   │   │   ├── images/                     # Approved training images
│   │   │   │   ├── sample-uuid-1.png
│   │   │   │   └── sample-uuid-2.png
│   │   │   └── exports/                    # Exported training files
│   │   │       └── qwen2vl_train_v3.json
│   │   └── adapters/
│   │       ├── v1/                         # Training run 1
│   │       │   ├── adapter_config.json
│   │       │   ├── adapter_model.safetensors
│   │       │   └── training_meta.json
│   │       └── v2/                         # Training run 2 (latest)
│   │           └── ...
│   └── tenant-xyz/
│       └── ... (same structure)
└── system/
    └── base_models/                        # Shared base model cache
```

### Upload Flow

```python
# Pseudocode for image storage

import boto3

s3 = boto3.client('s3', ...)
BUCKET = 'pdfai-data'

def store_extraction_image(tenant_id: str, extraction_id: str, page: int, image: Image):
    """Store page image in S3 for this tenant."""
    date = datetime.now().strftime('%Y-%m-%d')
    key = f"tenants/{tenant_id}/extractions/{date}/{extraction_id}-p{page}.png"

    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)

    s3.upload_fileobj(buffer, BUCKET, key)
    return key

def store_training_image(tenant_id: str, sample_id: str, image: Image):
    """Store approved training image."""
    key = f"tenants/{tenant_id}/training/images/{sample_id}.png"
    # ... same upload logic
    return key

def store_adapter(tenant_id: str, run_id: str, adapter_dir: str):
    """Upload trained adapter to S3."""
    for file in Path(adapter_dir).iterdir():
        key = f"tenants/{tenant_id}/adapters/{run_id}/{file.name}"
        s3.upload_file(str(file), BUCKET, key)
    return f"tenants/{tenant_id}/adapters/{run_id}/"

def load_adapter(tenant_id: str, run_id: str, local_dir: str):
    """Download adapter from S3 to local disk for inference."""
    prefix = f"tenants/{tenant_id}/adapters/{run_id}/"
    # Download all files in prefix to local_dir
    # ...
```

### Retention Policy

| Data Type              | Retention                  | Reason                                   |
| ---------------------- | -------------------------- | ---------------------------------------- |
| Extraction images      | 30 days                    | Only needed for correction review window |
| Training images        | Permanent (per tenant)     | Needed for retraining                    |
| Adapters               | Last 3 versions per tenant | Rollback capability                      |
| Exported training data | Last 3 exports             | Audit trail                              |

---

## 6. Admin Panel Pages

### Page 1: Dashboard

```
URL: /admin/dashboard
Purpose: Overview of the entire system at a glance
```

**Sections:**

| Section       | Content                                           |
| ------------- | ------------------------------------------------- |
| System Status | Server health, GPU utilization, memory, uptime    |
| Active Users  | Users currently online, active extraction jobs    |
| Queue         | Pending corrections awaiting review               |
| Training      | Last training run status, current adapter version |
| Usage (24h)   | Extractions today, corrections today, error rate  |

**Key Metrics (cards at top):**

- Total users (active / disabled)
- Extractions today / this week / this month
- Pending corrections to review
- Current model version + training loss
- GPU utilization %
- Server uptime

### Page 2: User Management

```
URL: /admin/users
Purpose: View, manage, enable/disable user accounts
```

**Table columns:**

- User name / email
- Role (user / admin)
- Status (active / disabled)
- Tenant
- Extractions count (total)
- Corrections count (total)
- Last active
- Actions (disable, delete, change role)

**Features:**

- Search / filter by name, email, role, status
- Bulk actions (disable multiple users)
- Click user → detailed usage history
- Create new user (invite by email)
- Change user role (user ↔ admin)
- View user's extraction history

**User Detail Page:**

```
URL: /admin/users/:id
```

- Profile info
- Activity timeline (extractions, corrections)
- Usage charts (extractions per day/week)
- Current active sessions
- Button: Disable account / Reset password

### Page 3: Correction Review Queue

```
URL: /admin/corrections
Purpose: Review user-submitted corrections before they enter training data
```

**This is the anti-data-poisoning layer.**

**Table/card view:**

- Field name
- Original AI value
- User's corrected value
- Confidence score
- Source PDF name
- Submitted by (user)
- Submitted at (timestamp)
- Actions: Approve / Reject / Edit

**Features:**

- Filter by status: pending, approved, rejected
- Filter by field name, user, date range
- Bulk approve (for trusted users)
- Side-by-side view: show the document image + correction
- Auto-flag suspicious corrections:
  - Correction is drastically different from AI prediction
  - User has submitted many corrections in a short time
  - Correction contains unusual characters or patterns
- Rejection requires a reason (logged for audit)

**Approval Flow:**

```
User submits correction
        │
        ▼
  ┌─────────────┐
  │   PENDING    │  ← Visible in admin panel
  └──────┬──────┘
         │
    Admin reviews
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│APPROVED│ │ REJECTED │
└───┬────┘ └────┬─────┘
    │           │
    ▼           ▼
 Enters      Logged with
 training    rejection
 data        reason
```

### Page 4: Training Data

```
URL: /admin/training
Purpose: View, manage, and curate the training dataset
```

**Stats panel:**

- Total approved samples
- Corrections vs approvals ratio
- Unique documents
- Most corrected fields (bar chart)
- Sample count over time (line chart)

**Sample list:**

- Image thumbnail
- Source PDF
- Fields extracted
- Corrections applied (highlighted)
- Approved by / date
- Actions: Delete / View detail

**Features:**

- Filter by document, field, date, corrected vs approved
- Delete individual samples
- Bulk delete
- Export to Qwen2.5-VL format (download JSON)
- Sample quality indicators (flagged if suspicious)

### Page 5: Model Training

```
URL: /admin/training/runs
Purpose: Trigger training, monitor progress, review history
```

**New Training Run form:**

- Epochs (default 3, range 1-10)
- Learning rate (default 2e-5)
- LoRA rank (default 8)
- LoRA alpha (default 16)
- Estimated duration
- Sample count to use
- Button: Start Training

**Training Runs table (history):**

- Run ID
- Started at
- Status: queued / running / completed / failed
- Samples used
- Final loss
- Duration
- Triggered by (admin name)
- Actions: View details, Activate adapter, Delete

**Active run (if in progress):**

- Real-time loss chart
- Current epoch / step
- Estimated time remaining
- GPU utilization
- Button: Cancel training

**Adapter management:**

- Currently active adapter (version, training date, loss)
- Button: Rollback to previous version
- Button: Deactivate adapter (use base model only)

### Page 6: Server Monitoring

```
URL: /admin/server
Purpose: Real-time server health and resource monitoring
```

**Metrics (live-updating):**

- CPU usage %
- RAM usage (used / total)
- GPU utilization %
- GPU VRAM (used / total)
- Disk usage
- Active connections
- Request rate (req/sec)

**Active Jobs:**

- Table of currently running extractions
- Username, file name, model, started at, duration
- Button: Kill job (force stop)

**Extraction Queue:**

- Pending extractions waiting for GPU
- Queue depth
- Average wait time

**Logs (live tail):**

- Last 100 server log lines
- Filter by level (INFO / WARNING / ERROR)
- Search

**Alerts:**

- GPU VRAM > 90% for > 5 minutes
- Error rate > 5% in last hour
- Extraction queue depth > 10
- Disk usage > 80%

### Page 7: System Settings

```
URL: /admin/settings
Purpose: Configure system-wide parameters
```

**Settings:**

- Confidence threshold (default 0.7)
- Max file size (default 10 MB)
- Allowed file types
- Default model (qwen / paddleocr)
- Voting rounds (default 3)
- Rate limiting (extractions per user per hour)
- Enable/disable image enhancement
- Maintenance mode toggle (blocks all user requests)

---

## 7. User App Changes

### What to Remove from User App

When migrating to the two-app model, remove these from the current frontend:

```javascript
// REMOVE from review_queue.js:
// - "Export Data" button
// - "View Data" button
// - Training panel entirely (training-info, readiness badge, stats)
//
// KEEP:
// - Review cards (approve / correct / delete)
// - Stats badges (pending / resolved count)
```

### What to Add to User App

- Login page (email + password)
- User profile (change password, view usage)
- "My Extractions" history page
- Logout button

### User Correction Flow (Modified)

```
CURRENT:
  User corrects → sample saved to training_data/ immediately

PRODUCTION:
  User corrects → correction saved to DB with status "pending"
                → user's extraction result updated immediately (for their download)
                → correction appears in admin panel for review
                → admin approves → enters training data
```

---

## 8. API Design

### Public API (User App)

```
Authentication:
  POST   /api/auth/login              — Login, get JWT
  POST   /api/auth/logout             — Invalidate token
  GET    /api/auth/me                 — Get current user info

Extraction:
  POST   /api/extract                 — Upload PDF + extract fields
  GET    /api/extractions             — List user's extraction history
  GET    /api/extractions/:id         — Get single extraction result

Review (user's own flagged fields):
  GET    /api/reviews                 — Get user's pending reviews
  POST   /api/reviews/:id/resolve    — Approve or correct a field

Profile:
  GET    /api/profile                 — User profile
  PUT    /api/profile/password       — Change password
```

### Admin API (Admin Panel)

```
Dashboard:
  GET    /api/admin/dashboard         — System overview stats
  GET    /api/admin/metrics/live     — Real-time metrics (WebSocket)

Users:
  GET    /api/admin/users             — List all users (paginated)
  GET    /api/admin/users/:id        — User detail + activity
  PUT    /api/admin/users/:id/status — Enable / disable user
  PUT    /api/admin/users/:id/role   — Change user role
  DELETE /api/admin/users/:id        — Delete user account
  POST   /api/admin/users/invite     — Invite new user by email

Corrections:
  GET    /api/admin/corrections       — List all corrections (filterable)
  GET    /api/admin/corrections/:id  — Correction detail with image
  POST   /api/admin/corrections/:id/approve   — Approve into training data
  POST   /api/admin/corrections/:id/reject    — Reject with reason
  POST   /api/admin/corrections/bulk-approve  — Bulk approve

Training Data:
  GET    /api/admin/training/samples  — List approved training samples
  DELETE /api/admin/training/samples/:id — Delete a sample
  POST   /api/admin/training/export  — Export to Qwen2.5-VL format
  GET    /api/admin/training/stats   — Training data statistics

Training Runs:
  POST   /api/admin/training/start   — Start a new training run
  GET    /api/admin/training/runs    — List training run history
  GET    /api/admin/training/runs/:id — Training run detail
  POST   /api/admin/training/runs/:id/cancel — Cancel running training
  POST   /api/admin/training/runs/:id/activate — Set adapter as active
  POST   /api/admin/training/runs/:id/rollback — Rollback to this version

Server:
  GET    /api/admin/server/status    — Server health + resource usage
  GET    /api/admin/server/jobs      — Active extraction jobs
  POST   /api/admin/server/jobs/:id/kill — Force-stop a job
  GET    /api/admin/server/logs      — Recent server logs
  POST   /api/admin/server/maintenance — Toggle maintenance mode

Settings:
  GET    /api/admin/settings         — Get all settings
  PUT    /api/admin/settings         — Update settings
```

---

## 9. Correction Review Pipeline

### The Complete Flow (Production)

```
Step 1: User extracts PDF
        └─→ Extraction saved to DB (extractions table)
        └─→ Image saved to S3 (tenants/{id}/extractions/...)
        └─→ Low-confidence fields auto-flagged in user's review queue

Step 2: User corrects a field
        └─→ Correction saved to DB with status "pending"
        └─→ User's result updated immediately (they can download corrected JSON)
        └─→ NOT added to training data yet

Step 3: Admin opens Admin Panel → Corrections page
        └─→ Sees all pending corrections across all users
        └─→ Can view the document image side-by-side with the correction
        └─→ Can see the user who made the correction
        └─→ Can see auto-flagged suspicious corrections

Step 4: Admin reviews each correction
        └─→ APPROVE: Correction + image promoted to training_samples table
        └─→ REJECT: Logged with reason, correction not used for training
        └─→ EDIT: Admin fixes the correction, then approves the edited version

Step 5: When enough approved samples exist, admin triggers training
        └─→ Training job queued (runs on GPU server, not web server)
        └─→ Progress visible in admin panel
        └─→ On completion, adapter saved to S3

Step 6: Admin activates the new adapter
        └─→ Server downloads adapter from S3
        └─→ All future extractions use the improved model
```

### Anti-Poisoning Safeguards

| Safeguard                   | Description                                             |
| --------------------------- | ------------------------------------------------------- |
| Admin-only approval         | No correction enters training without admin review      |
| Suspicious correction flags | Auto-flag if value is very different from AI prediction |
| User reputation scoring     | Users with many rejected corrections get lower trust    |
| Correction rate limiting    | Max corrections per user per hour                       |
| Audit trail                 | Every action is logged with user ID and timestamp       |
| Rollback capability         | Admin can revert to previous adapter version            |

---

## 10. Training Pipeline (Production)

### How Training Changes for SaaS

| Current (local)                           | Production (SaaS)                     |
| ----------------------------------------- | ------------------------------------- |
| `python train_qwen2vl.py` on same machine | Background job on separate GPU server |
| Immediate, blocking                       | Async, queued                         |
| One adapter for everyone                  | Per-tenant adapters                   |
| Local disk storage                        | S3 for adapters                       |
| No auth                                   | Admin-only trigger                    |

### Training Job Flow

```
Admin clicks "Start Training"
        │
        ▼
  ┌─────────────┐
  │   QUEUED     │  Job created in DB
  └──────┬──────┘
         │
         ▼ (GPU available)
  ┌─────────────┐
  │   RUNNING    │  Download images from S3
  │              │  Load base model + QLoRA
  │              │  Train for N epochs
  │              │  Report progress via WebSocket
  └──────┬──────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────┐
│COMPLETE│ │FAILED│
│        │ │      │
│Upload  │ │Log   │
│adapter │ │error │
│to S3   │ │      │
└───┬────┘ └──────┘
    │
    ▼
Admin activates
new adapter
```

### Per-Tenant Adapter Loading

```python
# On each extraction request:

async def get_model_for_tenant(tenant_id: str):
    """Load base model with tenant's LoRA adapter."""

    # Check if tenant has an active adapter
    active = db.query(active_adapters).filter_by(tenant_id=tenant_id).first()

    if active:
        # Download adapter from S3 if not cached locally
        local_path = f"/tmp/adapters/{tenant_id}/{active.training_run_id}"
        if not os.path.exists(local_path):
            download_adapter_from_s3(active.adapter_storage_key, local_path)

        # Load base model + merge LoRA
        model = load_base_model()
        model = PeftModel.from_pretrained(model, local_path)
        return model

    # No adapter — use base model
    return load_base_model()
```

---

## 11. Server Monitoring

### Metrics to Track

```
System:
  - CPU usage (%)
  - RAM used / total (GB)
  - Disk used / total (GB)
  - Network I/O (MB/s)

GPU:
  - Utilization (%)
  - VRAM used / total (GB)
  - Temperature (°C)
  - Power draw (W)

Application:
  - Active connections
  - Requests per second
  - Average response time (ms)
  - Error rate (%)
  - Active extraction jobs
  - Queue depth (waiting for GPU)

Business:
  - Extractions per hour/day
  - Corrections per hour/day
  - Unique active users
  - Pending corrections count
```

### Implementation

```python
# GPU metrics (using nvidia-smi or pynvml)
import pynvml

def get_gpu_metrics():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

    return {
        "gpu_utilization": util.gpu,
        "vram_used_gb": mem.used / 1024**3,
        "vram_total_gb": mem.total / 1024**3,
        "temperature_c": temp,
    }

# System metrics (using psutil)
import psutil

def get_system_metrics():
    return {
        "cpu_percent": psutil.cpu_percent(),
        "ram_used_gb": psutil.virtual_memory().used / 1024**3,
        "ram_total_gb": psutil.virtual_memory().total / 1024**3,
        "disk_used_gb": psutil.disk_usage('/').used / 1024**3,
        "disk_total_gb": psutil.disk_usage('/').total / 1024**3,
    }
```

### Live Updates (WebSocket)

Admin panel connects via WebSocket for real-time metrics:

```
ws://admin.pdfai.com/api/admin/metrics/live

Server sends every 2 seconds:
{
  "timestamp": "2026-02-18T12:00:00",
  "cpu": 45,
  "ram_used": 8.2,
  "gpu_util": 78,
  "vram_used": 3.8,
  "active_jobs": 2,
  "queue_depth": 1,
  "req_per_sec": 12
}
```

---

## 12. Security Considerations

### Authentication

- Passwords hashed with bcrypt (cost factor 12+)
- JWT tokens with short expiry (15 min access, 7 day refresh)
- Admin login requires 2FA (TOTP via authenticator app)
- Rate limit login attempts (5 per minute per IP)

### Authorization

- Tenant isolation: users can NEVER access another tenant's data
- SQL queries ALWAYS filter by tenant_id
- S3 keys are prefixed by tenant_id
- Admin API endpoints require admin role JWT

### Data Security

- All API traffic over HTTPS (TLS 1.3)
- S3 bucket private, no public access
- Database encrypted at rest
- PII fields (names, DOB, addresses) encrypted in extraction results
- Document images treated as sensitive data
- GDPR: user can request data deletion

### Rate Limiting

```
User API:
  - Extractions: 30 per hour per user
  - Corrections: 100 per hour per user
  - Downloads: 60 per hour per user

Admin API:
  - Training starts: 5 per day per tenant
  - Bulk operations: 10 per hour

Global:
  - 1000 requests per minute per IP (DDoS protection)
```

---

## 13. Deployment Architecture

### Option A: Single VPS (Small Scale)

```
1 server (8 vCPU, 32 GB RAM, 1x T4 GPU)

  ┌────────────────────────────┐
  │  Nginx (reverse proxy)      │
  │  ├── app.pdfai.com → :5173  │
  │  └── admin.pdfai.com → :5174│
  │                              │
  │  FastAPI backend (:8000)     │
  │  PostgreSQL (:5432)          │
  │  Redis (:6379)               │
  │  Celery worker (training)    │
  └────────────────────────────┘
  + S3 bucket for images/adapters
```

Best for: < 50 users, < 100 extractions/day

### Option B: Managed Services (Medium Scale)

```
  ┌────────────────┐   ┌─────────────────┐
  │ Vercel/CF Pages│   │ Vercel/CF Pages  │
  │ (User App)     │   │ (Admin Panel)    │
  └───────┬────────┘   └────────┬─────────┘
          │                      │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────┐
          │  Cloud Run / ECS  │  ← Auto-scaling API
          │  (FastAPI)        │
          └────────┬──────────┘
                   │
     ┌─────────────┼──────────────┐
     ▼             ▼              ▼
 ┌────────┐  ┌──────────┐  ┌──────────┐
 │ Cloud  │  │ Managed  │  │   S3     │
 │  SQL   │  │  Redis   │  │  Bucket  │
 └────────┘  └──────────┘  └──────────┘
                   │
          ┌────────┴──────────┐
          │  GPU Instance     │  ← Training only (on-demand)
          │  (RunPod / Lambda)│
          └───────────────────┘
```

Best for: 50-500 users, pay-as-you-go GPU for training

### Option C: Kubernetes (Large Scale)

For 500+ users, multi-region, enterprise. Full k8s orchestration with
GPU node pools for inference and training, auto-scaling pods, etc.

---

## 14. Migration Path

### Phase 1: Current → Auth + DB (Week 1-2)

```
KEEP: Everything as-is
ADD:
  - PostgreSQL database
  - User auth (JWT)
  - Login page on user app
  - Store extractions + corrections in DB

STILL LOCAL:
  - Images on disk
  - Training on same machine
  - No admin panel yet
```

### Phase 2: Admin Panel MVP (Week 3-4)

```
ADD:
  - Admin panel (separate Next.js app)
  - Dashboard page
  - User management page
  - Correction review page (the critical one)

CHANGE:
  - Remove training buttons from user app
  - Corrections go to "pending" state
  - Admin approves before training
```

### Phase 3: Cloud Storage (Week 5)

```
CHANGE:
  - Images → S3
  - Adapters → S3
  - Remove local training_data/images/

ADD:
  - S3 upload/download utilities
  - Image cleanup cron job
```

### Phase 4: Production Training (Week 6)

```
CHANGE:
  - Training runs as background job (Celery + Redis)
  - Admin triggers from panel
  - Real-time progress via WebSocket

ADD:
  - Training runs page
  - Adapter version management
  - Rollback capability
```

### Phase 5: Monitoring + Polish (Week 7-8)

```
ADD:
  - Server monitoring page
  - Live metrics WebSocket
  - Alerting
  - System settings page
  - Rate limiting
  - 2FA for admins
```

---

## 15. Tech Stack Recommendations

### Admin Panel Frontend

| Choice               | Why                                                |
| -------------------- | -------------------------------------------------- |
| **Next.js 15**       | Server components, API routes, fast builds         |
| **shadcn/ui**        | Professional component library, fully customizable |
| **Recharts**         | Charts for training stats, usage graphs            |
| **TanStack Table**   | Powerful data tables for user/correction lists     |
| **Socket.io client** | Real-time metrics display                          |

### Backend Additions

| Choice              | Why                                        |
| ------------------- | ------------------------------------------ |
| **PostgreSQL 16**   | Relational DB, JSONB for flexible fields   |
| **Redis**           | Job queue (Celery), caching, rate limiting |
| **Celery**          | Background training jobs                   |
| **boto3**           | S3 storage                                 |
| **python-jose**     | JWT token handling                         |
| **passlib[bcrypt]** | Password hashing                           |
| **pyotp**           | 2FA TOTP generation                        |
| **pynvml**          | GPU metrics                                |
| **psutil**          | System metrics                             |

### Infrastructure

| Choice        | Why                                   |
| ------------- | ------------------------------------- |
| **AWS / GCP** | S3/GCS, Cloud SQL, GPU instances      |
| **RunPod**    | On-demand GPU for training (cheapest) |
| **Vercel**    | Frontend hosting (both apps)          |
| **Docker**    | Containerized backend                 |
| **Nginx**     | Reverse proxy, SSL termination        |
