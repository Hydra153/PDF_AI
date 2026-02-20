import { sanitizeInput } from "./services/utils.js";
import { icons } from "./components/icons.js";
import {
  extractFields,
  autoFindFields as autoFindFieldsAPI,
  checkBackendHealth,
  detectCheckboxes,
} from "./services/api/backend.js";
import { ReviewQueue } from "./components/review_queue.js";
import {
  extractLayoutFromPDF,
  pdfPageToImage,
} from "./services/pdf_helpers.js";

// Fields to Extract (starts empty - use Auto-Find or add manually)
let CURRENT_FIELDS = [];

function renderApp() {
  const root = document.getElementById("app");
  if (!root) return;

  root.innerHTML = `
    <div class="app">
      <header class="hero">
        <h1>PDF Data Extractor</h1>
        <p class="sub">Upload a form to extract structured data. Add custom fields or use Smart Scan.</p>
        
        <!-- Navigation Tabs -->
        <div class="nav-tabs" style="margin-top: 24px;">
          <button id="tab-extract" class="active">Extract</button>
          <button id="tab-review">Review Queue</button>
        </div>
      </header>
      
      <!-- Extract View -->
      <div id="view-extract">

      <section class="panel" id="fields-panel">
        <label class="label">Fields to Extract</label>
        <button id="auto-find-btn" disabled style="width: 100%; margin-bottom: 12px; background: #27ae60;">${icons.search(14)} Auto-Find Fields</button>
        <div class="field-list" id="field-list"></div>
        <div class="add-field-row">
            <input type="text" id="new-field-input" placeholder="Add new field (e.g. Allergies)..." />
            <button id="add-field-btn" type="button">+ Add</button>
            <div class="presets-wrapper">
              <button id="presets-btn" type="button" class="presets-btn">⚙ Presets</button>
              <div id="presets-dropdown" class="presets-dropdown" style="display:none;"></div>
            </div>
        </div>
      </section>

      <section class="panel">
        <div class="upload-area" id="drop-zone">
          <input type="file" id="file-input" accept="application/pdf" style="display: none" />
          <button id="upload-btn">Choose PDF File</button>
          <p id="file-name">No file selected</p>
        </div>

        <div style="margin: 20px 0; padding: 12px 16px; background: rgba(39, 174, 96, 0.08); border: 1px solid rgba(39, 174, 96, 0.2); border-radius: 8px; display: flex; align-items: center; gap: 10px;">
          <span style="font-size: 1.3rem;">${icons.eye(20)}</span>
          <div>
            <span id="model-badge-name" style="font-weight: 600; color: #27ae60;">Loading model...</span>
            <small id="model-badge-sub" style="display: block; color: #888; font-size: 11px;">Vision Language Model • OCR-free</small>
          </div>
        </div>

        <div class="model-selector" style="margin: 0 0 16px 0; padding: 10px 14px; background: rgba(74, 144, 226, 0.06); border: 1px solid rgba(74, 144, 226, 0.15); border-radius: 8px;">
          <span style="font-weight: 600; font-size: 12px; color: #999; text-transform: uppercase; letter-spacing: 0.5px; display: block; margin-bottom: 8px;">Extraction Engine</span>
          <div style="display: flex; gap: 16px;">
            <label id="label-paddleocr" style="display: flex; align-items: center; gap: 6px; cursor: pointer; padding: 6px 12px; border-radius: 6px; border: 1px solid rgba(39, 174, 96, 0.3); background: rgba(39, 174, 96, 0.08);">
              <input type="radio" name="model-select" value="paddleocr" checked style="accent-color: #27ae60;" />
              <span style="font-weight: 500;">PaddleOCR-VL 1.5</span>
              <small style="color: #888; font-size: 10px;">(0.9B)</small>
            </label>
            <label id="label-qwen" style="display: flex; align-items: center; gap: 6px; cursor: pointer; padding: 6px 12px; border-radius: 6px; border: 1px solid rgba(74, 144, 226, 0.2); background: transparent;">
              <input type="radio" name="model-select" value="qwen" style="accent-color: #4a90e2;" />
              <span style="font-weight: 500;">Qwen2.5-VL-3B</span>
              <small style="color: #888; font-size: 10px;">(3B)</small>
            </label>
          </div>
        </div>

        <div id="voting-option" style="margin: 0 0 16px 0; padding: 8px 14px; background: rgba(74, 144, 226, 0.04); border: 1px solid rgba(74, 144, 226, 0.12); border-radius: 8px; display: none;">
          <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
            <input type="checkbox" id="voting-checkbox" style="accent-color: #4a90e2; width: 16px; height: 16px;" />
            <span style="font-weight: 500; font-size: 13px;">Accuracy Boost</span>
            <small style="color: #888; font-size: 11px;">(3× voting — slower but more accurate)</small>
          </label>
        </div>

        <div class="actions" style="display: flex; align-items: center; gap: 16px;">
           <button id="extract-btn" disabled>Extract Data</button>
           <label id="checkbox-toggle-label" style="display: flex; align-items: center; gap: 8px; cursor: pointer; padding: 6px 14px; border-radius: 8px; border: 1px solid rgba(142, 68, 173, 0.25); background: rgba(142, 68, 173, 0.06); transition: all 0.2s; user-select: none;">
             <input type="checkbox" id="checkbox-toggle" style="accent-color: #8e44ad; width: 16px; height: 16px;" />
             <span style="font-weight: 500; font-size: 13px; color: #8e44ad;">☑ Checkbox Mode</span>
           </label>
        </div>
      </section>

      <section class="panel output-panel">
        <div class="status">
          <span data-status>Idle</span>
        </div>
        <div id="results-container" class="results-grid"></div>
        
        <div id="scan-results-section" style="display:none; margin-top: 24px;">
            <p class="label">Smart Scan (Detected Patterns)</p>
            <div id="scan-results-container" class="results-grid"></div>
        </div>
      </section>

      <section class="panel" id="analysis-panel">
        <label class="label">Document Analysis</label>
        <p style="font-size: 0.8rem; color: var(--muted); margin: 0 0 12px 0;">Detect structured elements beyond text fields</p>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
          <button id="detect-checkboxes-btn" disabled style="background: #8e44ad; color: white; border: none; padding: 10px 18px; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 0.85rem; display: flex; align-items: center; gap: 6px; transition: all 0.2s;">☑ Find All Checkboxes</button>
        </div>
        <div id="checkbox-results-container" style="margin-top: 16px;"></div>
      </section>

      <section class="panel">
        <label class="label">Document Preview</label>
        <div id="layout-preview-container" style="margin-top: 16px; background: #fff; padding: 20px; border-radius: 8px;">
            <div id="no-preview" style="text-align: center; padding: 40px; color: #999;">
                No Preview
            </div>
            <img id="pdf-preview-img" style="display: none; max-width: 100%; border: 2px solid #3498db; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" alt="Document Preview" />
        </div>
      </section>
      </div><!-- End Extract View -->
      
      <!-- Review View -->
      <div id="view-review" style="display: none;">
        <div id="review-queue-container"></div>
      </div>
    </div>
  `;

  // UI References
  const fileInput = document.getElementById("file-input");
  const uploadBtn = document.getElementById("upload-btn");
  const fileNameDisplay = document.getElementById("file-name");
  const extractBtn = document.getElementById("extract-btn");
  const bboxCanvas = document.getElementById("pdf-preview-img");
  const noPreview = document.getElementById("no-preview");

  const statusEl = document.querySelector("[data-status]");
  const resultsContainer = document.getElementById("results-container");
  const scanResultsSection = document.getElementById("scan-results-section");
  const scanResultsContainer = document.getElementById(
    "scan-results-container",
  );
  const fieldListEl = document.getElementById("field-list");
  const newFieldInput = document.getElementById("new-field-input");
  const addFieldBtn = document.getElementById("add-field-btn");
  const autoFindBtn = document.getElementById("auto-find-btn");
  const presetsBtn = document.getElementById("presets-btn");
  const presetsDropdown = document.getElementById("presets-dropdown");
  const detectCheckboxesBtn = document.getElementById("detect-checkboxes-btn");
  const checkboxResultsContainer = document.getElementById("checkbox-results-container");

  let selectedFile = null;
  let lastCheckboxResults = null;  // Store for JSON export
  let checkboxEnabled = false;  // Checkbox mode toggle state

  // Tab Navigation
  const tabExtract = document.getElementById("tab-extract");
  const tabReview = document.getElementById("tab-review");
  const viewExtract = document.getElementById("view-extract");
  const viewReview = document.getElementById("view-review");
  const reviewContainer = document.getElementById("review-queue-container");

  // Initialize ReviewQueue
  const reviewQueue = new ReviewQueue(reviewContainer);

  // Clear queue on page load - ensures consistent state
  // Reviews only make sense with a PDF loaded, so start fresh
  // Guard: backend may not be ready yet at page load time
  reviewQueue.clearQueue().catch(() => {});

  // ─── Startup: fetch model names from backend & configure selector ───
  (async () => {
    try {
      const health = await checkBackendHealth();
      const badge = document.getElementById("model-badge-name");
      const badgeSub = document.getElementById("model-badge-sub");

      if (health && health.models) {
        // New multi-model health format
        const models = health.models;

        // Gray out unavailable models
        if (!models.paddleocr?.available) {
          const labelP = document.getElementById("label-paddleocr");
          if (labelP) {
            labelP.style.opacity = "0.4";
            labelP.style.pointerEvents = "none";
            const radio = labelP.querySelector("input");
            if (radio) radio.disabled = true;
          }
          // Switch to qwen if paddle unavailable
          const qwenRadio = document.querySelector('input[name="model-select"][value="qwen"]');
          if (qwenRadio) qwenRadio.checked = true;
        }

        if (!models.qwen?.available) {
          const labelQ = document.getElementById("label-qwen");
          if (labelQ) {
            labelQ.style.opacity = "0.4";
            labelQ.style.pointerEvents = "none";
            const radio = labelQ.querySelector("input");
            if (radio) radio.disabled = true;
          }
        }

        // Update badge with currently selected model
        const selected = document.querySelector('input[name="model-select"]:checked')?.value || "paddleocr";
        updateModelBadge(selected, models);
      } else if (health && badge) {
        badge.textContent = health.model || "Qwen VL";
      } else if (badge) {
        badge.textContent = "Backend offline";
        badge.style.color = "#e74c3c";
      }
    } catch {
      const badge = document.getElementById("model-badge-name");
      if (badge) {
        badge.textContent = "Backend offline";
        badge.style.color = "#e74c3c";
      }
    }
  })();

  // ─── Model selector radio button change handler ───
  document.querySelectorAll('input[name="model-select"]').forEach((radio) => {
    radio.addEventListener("change", (e) => {
      const selected = e.target.value;
      const labelP = document.getElementById("label-paddleocr");
      const labelQ = document.getElementById("label-qwen");

      if (selected === "paddleocr") {
        labelP.style.border = "1px solid rgba(39, 174, 96, 0.3)";
        labelP.style.background = "rgba(39, 174, 96, 0.08)";
        labelQ.style.border = "1px solid rgba(74, 144, 226, 0.2)";
        labelQ.style.background = "transparent";
      } else {
        labelQ.style.border = "1px solid rgba(74, 144, 226, 0.3)";
        labelQ.style.background = "rgba(74, 144, 226, 0.08)";
        labelP.style.border = "1px solid rgba(39, 174, 96, 0.2)";
        labelP.style.background = "transparent";
      }

      // Update badge
      updateModelBadge(selected);

      // Show/hide voting checkbox (Qwen only)
      const votingDiv = document.getElementById("voting-option");
      if (votingDiv) {
        votingDiv.style.display = selected === "qwen" ? "block" : "none";
        // Uncheck when hiding
        if (selected !== "qwen") {
          const cb = document.getElementById("voting-checkbox");
          if (cb) cb.checked = false;
        }
      }
    });
  });

  // ─── Checkbox Mode Toggle Handler ───
  const checkboxToggle = document.getElementById("checkbox-toggle");
  const analysisPanel = document.getElementById("analysis-panel");

  // Start with analysis panel hidden (checkbox mode OFF by default)
  if (analysisPanel) analysisPanel.style.display = "none";

  checkboxToggle.addEventListener("change", (e) => {
    checkboxEnabled = e.target.checked;
    const toggleLabel = document.getElementById("checkbox-toggle-label");

    if (checkboxEnabled) {
      if (analysisPanel) analysisPanel.style.display = "block";
      if (toggleLabel) {
        toggleLabel.style.background = "rgba(142, 68, 173, 0.15)";
        toggleLabel.style.borderColor = "rgba(142, 68, 173, 0.5)";
      }
    } else {
      if (analysisPanel) analysisPanel.style.display = "none";
      if (toggleLabel) {
        toggleLabel.style.background = "rgba(142, 68, 173, 0.06)";
        toggleLabel.style.borderColor = "rgba(142, 68, 173, 0.25)";
      }
      // Clear checkbox results when disabling
      checkboxResultsContainer.innerHTML = "";
      lastCheckboxResults = null;
    }
  });

  function updateModelBadge(selected, models) {
    const badge = document.getElementById("model-badge-name");
    const badgeSub = document.getElementById("model-badge-sub");
    if (!badge) return;

    if (selected === "paddleocr") {
      badge.textContent = models?.paddleocr?.name || "PaddleOCR-VL 1.5";
      badge.style.color = "#27ae60";
      if (badgeSub) badgeSub.textContent = "Document Parsing Model • 0.9B params • 109 languages";
    } else {
      badge.textContent = models?.qwen?.name || "Qwen2.5-VL-3B";
      badge.style.color = "#4a90e2";
      if (badgeSub) badgeSub.textContent = "Vision Language Model • OCR-free";
    }
  }

  function switchTab(tab) {
    if (tab === "extract") {
      tabExtract.classList.add("active");
      tabReview.classList.remove("active");
      viewExtract.style.display = "block";
      viewReview.style.display = "none";
    } else {
      tabReview.classList.add("active");
      tabExtract.classList.remove("active");
      viewExtract.style.display = "none";
      viewReview.style.display = "block";
      reviewQueue.init(); // Refresh when switching to review
    }
  }

  tabExtract.addEventListener("click", () => switchTab("extract"));
  tabReview.addEventListener("click", () => switchTab("review"));

  // Render Fields
  function renderFields() {
    fieldListEl.innerHTML = CURRENT_FIELDS.map(
      (f, i) => `
        <div class="field-tag">
            <span>${f.key}</span>
            <button class="remove-field" data-index="${i}">×</button>
        </div>
      `,
    ).join("");

    // Re-attach listeners
    document.querySelectorAll(".remove-field").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        const idx = parseInt(e.target.dataset.index);
        CURRENT_FIELDS.splice(idx, 1);
        renderFields();
      });
    });
  }

  // Add Field Logic
  addFieldBtn.addEventListener("click", () => {
    const val = newFieldInput.value.trim();
    if (val) {
      CURRENT_FIELDS.push({ key: val, question: `What is the ${val}?` });
      newFieldInput.value = "";
      renderFields();
    }
  });

  // ─── Presets Logic (localStorage) ───
  const PRESETS_KEY = "pdf_extractor_presets";

  // Default presets – seeded on first run
  const DEFAULT_PRESETS = [
    {
      name: "CPL Form",
      fields: [
        "Fasting", "Client Name", "Client Address", "Client Phone",
        "Patient Name", "Patient Address", "Patient Phone", "DOB",
        "Age", "Primary Insurance Name", "Primary Insurance Address",
        "Physician", "Diagnosis Codes", "Date"
      ]
    }
  ];

  function loadPresets() {
    try {
      const raw = localStorage.getItem(PRESETS_KEY);
      if (raw) return JSON.parse(raw);
    } catch { /* corrupt data */ }
    // First run – seed defaults
    savePresetsToStorage(DEFAULT_PRESETS);
    return DEFAULT_PRESETS;
  }

  function savePresetsToStorage(presets) {
    localStorage.setItem(PRESETS_KEY, JSON.stringify(presets));
  }

  function renderPresetsDropdown() {
    const presets = loadPresets();
    const hasFields = CURRENT_FIELDS.length > 0;

    let html = `<div class="presets-dropdown-header">
      <span>Saved Presets</span>
      <button class="preset-save-btn" id="preset-save-trigger" ${!hasFields ? 'disabled title="Add fields first"' : ''}>+ Save Current</button>
    </div>`;

    if (presets.length === 0) {
      html += `<div class="preset-empty">
        <div class="preset-empty-icon">${icons.clipboard(24)}</div>
        No presets yet. Add fields and click "Save Current".
      </div>`;
    } else {
      html += `<div class="preset-list">`;
      presets.forEach((p, idx) => {
        html += `<div class="preset-item" data-preset-idx="${idx}">
          <span class="preset-item-name" data-load-idx="${idx}">${p.name}</span>
          <span class="preset-item-count">${p.fields.length} fields</span>
          <div class="preset-item-actions">
            <button class="preset-action-btn rename" data-rename-idx="${idx}" title="Rename">${icons.edit(12)}</button>
            <button class="preset-action-btn delete" data-delete-idx="${idx}" title="Delete">✕</button>
          </div>
        </div>`;
      });
      html += `</div>`;
    }

    presetsDropdown.innerHTML = html;

    // ── Attach event handlers ──

    // Save Current trigger
    const saveTrigger = document.getElementById("preset-save-trigger");
    if (saveTrigger && hasFields) {
      saveTrigger.addEventListener("click", (e) => {
        e.stopPropagation();
        showSaveRow();
      });
    }

    // Load preset on name click
    presetsDropdown.querySelectorAll("[data-load-idx]").forEach(el => {
      el.addEventListener("click", (e) => {
        e.stopPropagation();
        const idx = parseInt(el.dataset.loadIdx);
        applyPreset(idx);
      });
    });

    // Rename
    presetsDropdown.querySelectorAll("[data-rename-idx]").forEach(btn => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const idx = parseInt(btn.dataset.renameIdx);
        startRename(idx);
      });
    });

    // Delete
    presetsDropdown.querySelectorAll("[data-delete-idx]").forEach(btn => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const idx = parseInt(btn.dataset.deleteIdx);
        deletePreset(idx);
      });
    });
  }

  function showSaveRow() {
    // Remove existing save row if any
    const existing = presetsDropdown.querySelector(".preset-save-row");
    if (existing) { existing.remove(); return; }

    const row = document.createElement("div");
    row.className = "preset-save-row";
    row.innerHTML = `<input type="text" placeholder="Preset name..." id="preset-name-input" />
      <button id="preset-confirm-save">Save</button>`;
    presetsDropdown.appendChild(row);

    const nameInput = document.getElementById("preset-name-input");
    nameInput.focus();

    const doSave = () => {
      const name = nameInput.value.trim();
      if (!name) return;
      const presets = loadPresets();
      presets.push({
        name,
        fields: CURRENT_FIELDS.map(f => f.key)
      });
      savePresetsToStorage(presets);
      renderPresetsDropdown();
    };

    document.getElementById("preset-confirm-save").addEventListener("click", (e) => {
      e.stopPropagation();
      doSave();
    });
    nameInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") doSave();
      e.stopPropagation();
    });
  }

  function applyPreset(idx) {
    const presets = loadPresets();
    if (idx < 0 || idx >= presets.length) return;
    const preset = presets[idx];
    CURRENT_FIELDS = preset.fields.map(key => ({ key, question: `What is the ${key}?` }));
    renderFields();
    presetsDropdown.style.display = "none";
  }

  function deletePreset(idx) {
    const presets = loadPresets();
    presets.splice(idx, 1);
    savePresetsToStorage(presets);
    renderPresetsDropdown();
  }

  function startRename(idx) {
    const presets = loadPresets();
    const item = presetsDropdown.querySelector(`[data-preset-idx="${idx}"]`);
    if (!item) return;

    const nameSpan = item.querySelector(".preset-item-name");
    const oldName = presets[idx].name;

    // Replace name span with input
    const input = document.createElement("input");
    input.className = "preset-rename-input";
    input.value = oldName;
    nameSpan.replaceWith(input);
    input.focus();
    input.select();

    const finishRename = () => {
      const newName = input.value.trim();
      if (newName && newName !== oldName) {
        presets[idx].name = newName;
        savePresetsToStorage(presets);
      }
      renderPresetsDropdown();
    };

    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") finishRename();
      if (e.key === "Escape") renderPresetsDropdown();
      e.stopPropagation();
    });
    input.addEventListener("blur", finishRename);
  }

  // Toggle dropdown
  presetsBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    const open = presetsDropdown.style.display !== "none";
    if (open) {
      presetsDropdown.style.display = "none";
    } else {
      renderPresetsDropdown();
      presetsDropdown.style.display = "block";
    }
  });

  // Close dropdown on click outside
  document.addEventListener("click", (e) => {
    if (!presetsDropdown.contains(e.target) && e.target !== presetsBtn) {
      presetsDropdown.style.display = "none";
    }
  });

  // Initial render
  renderFields();

  // Clear any stale results on page load
  resultsContainer.innerHTML = `
    <div class="empty-results animate-fadeUp">
      <div style="font-size: 2rem; margin-bottom: 8px; color: var(--muted);">${icons.file(32)}</div>
      <p>Upload a PDF to extract data</p>
    </div>
  `;

  // Event Handlers
  uploadBtn.addEventListener("click", () => fileInput.click());

  // Drag & Drop Support
  const dropZone = document.getElementById("drop-zone");

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", async (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type === "application/pdf") {
      await handleFileSelect(files[0]);
    }
  });

  async function handleFileSelect(file) {
    selectedFile = file;
    fileNameDisplay.innerHTML = `${icons.file(14)} <strong>${file.name}</strong>`;
    extractBtn.disabled = false;
    autoFindBtn.disabled = false;
    detectCheckboxesBtn.disabled = false;
    statusEl.textContent = "Ready to process";

    // Clear review queue when new PDF selected
    await reviewQueue.clearQueue();

    // Clear previous results
    resultsContainer.innerHTML = "";
    checkboxResultsContainer.innerHTML = "";
    lastCheckboxResults = null;

    await renderLayoutPreview();
  }

  fileInput.addEventListener("change", async (e) => {
    if (e.target.files.length > 0) {
      await handleFileSelect(e.target.files[0]);
    }
  });

  // Auto-Find Fields Handler
  autoFindBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    try {
      statusEl.textContent = "Checking backend connection...";
      autoFindBtn.disabled = true;

      // Check if backend is available
      const health = await checkBackendHealth();

      if (!health) {
        throw new Error(
          "Backend server not running. Please start the backend first.",
        );
      }

      statusEl.textContent = "Analyzing document for fields...";

      // Call backend API
      const detectedFields = await autoFindFieldsAPI(selectedFile);

      if (!detectedFields || detectedFields.length === 0) {
        statusEl.textContent = "No fields detected. Try manual entry.";
        autoFindBtn.disabled = false;
        return;
      }

      // Replace current fields with detected fields
      CURRENT_FIELDS = detectedFields;
      renderFields();

      statusEl.textContent = `Found ${detectedFields.length} fields! Ready to extract.`;
      console.log("Auto-detected fields:", detectedFields);
    } catch (err) {
      console.error("Auto-find error:", err);
      statusEl.textContent = `Error: ${err.message}`;
    } finally {
      autoFindBtn.disabled = false;
    }
  });

  // ─── Checkbox Detection Handler ───
  detectCheckboxesBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    try {
      detectCheckboxesBtn.disabled = true;
      detectCheckboxesBtn.textContent = "⏳ Scanning...";
      statusEl.textContent = "Detecting checkboxes...";
      checkboxResultsContainer.innerHTML = `
        <div style="text-align: center; padding: 20px; color: var(--muted);">
          <div class="spinner" style="margin: 0 auto 12px;"></div>
          Scanning document for checkboxes...
        </div>`;

      const result = await detectCheckboxes(selectedFile);

      if (!result.checkboxes || result.checkboxes.length === 0) {
        checkboxResultsContainer.innerHTML = `
          <div style="text-align: center; padding: 20px; color: var(--muted);">
            ☐ No physical checkboxes found in this document
          </div>`;
        statusEl.textContent = "No checkboxes found";
        lastCheckboxResults = [];
        return;
      }

      lastCheckboxResults = result.checkboxes;
      renderCheckboxResults(result.checkboxes, result.time_seconds);
      statusEl.textContent = `Found ${result.checkboxes.length} checkboxes in ${result.time_seconds}s`;

    } catch (err) {
      console.error("Checkbox detection error:", err);
      checkboxResultsContainer.innerHTML = `
        <div style="color: #ef4444; padding: 12px;">❌ ${err.message}</div>`;
      statusEl.textContent = `Error: ${err.message}`;
    } finally {
      detectCheckboxesBtn.disabled = false;
      detectCheckboxesBtn.textContent = "☑ Find All Checkboxes";
    }
  });

  function renderCheckboxResults(checkboxes, timeSec) {
    const checked = checkboxes.filter(c => c.checked).length;
    const unchecked = checkboxes.length - checked;

    let html = `
      <div style="display: flex; gap: 12px; margin-bottom: 12px; font-size: 0.8rem; color: var(--muted); align-items: center;">
        <span>☑ ${checked} checked</span>
        <span>☐ ${unchecked} unchecked</span>
        <span style="margin-left: auto;">${timeSec}s</span>
      </div>
      <div style="display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap;">
        <button id="merge-checkboxes-btn" class="btn-secondary" style="font-size: 0.78rem; padding: 6px 14px;">🔗 Merge into Results</button>
        <button id="copy-checkbox-json" class="btn-secondary" style="font-size: 0.78rem; padding: 6px 14px;">📋 Copy JSON</button>
        <button id="download-checkbox-json" class="btn-secondary" style="font-size: 0.78rem; padding: 6px 14px;">⬇ Download</button>
      </div>
      <div class="results-grid">`;

    checkboxes.forEach((cb, i) => {
      const icon = cb.checked ? "☑" : "☐";
      const statusText = cb.checked ? "Checked" : "Unchecked";
      const statusColor = cb.checked ? "#22c55e" : "#94a3b8";
      const cbSignal = cb.signal || {};
      const cbSource = cbSignal.source || "batch";
      const sourceIcon = cbSource === "checkbox_vqa" ? "🔍" : "📋";

      html += `
        <div class="result-card animate-fadeUp cb-card" data-cb-index="${i}" data-cb-checked="${cb.checked}" style="animation-delay: ${i * 0.03}s; cursor: pointer;" title="Click to toggle">
          <div class="card-header">
            <span class="field-name">${cb.label}</span>
            <span class="signal-badge">${sourceIcon} ${cbSource}</span>
          </div>
          <div class="field-value" style="display: flex; align-items: center; gap: 8px;">
            <span class="cb-icon" style="font-size: 1.4rem; color: ${statusColor};">${icon}</span>
            <span class="cb-status">${statusText}</span>
          </div>
        </div>`;
    });

    html += `</div>`;
    checkboxResultsContainer.innerHTML = html;

    // Wire up toggle on each checkbox card
    checkboxResultsContainer.querySelectorAll(".cb-card").forEach((card) => {
      card.addEventListener("click", () => {
        const idx = parseInt(card.dataset.cbIndex);
        const cb = lastCheckboxResults[idx];
        if (!cb) return;

        // Toggle state
        cb.checked = !cb.checked;
        card.dataset.cbChecked = cb.checked;

        // Update visual
        const iconEl = card.querySelector(".cb-icon");
        const statusEl = card.querySelector(".cb-status");
        iconEl.textContent = cb.checked ? "☑" : "☐";
        iconEl.style.color = cb.checked ? "#22c55e" : "#94a3b8";
        statusEl.textContent = cb.checked ? "Checked" : "Unchecked";

        // Flash animation
        card.style.transform = "scale(0.97)";
        setTimeout(() => { card.style.transform = ""; }, 150);
      });
    });

    // Merge into Results button
    document.getElementById("merge-checkboxes-btn")?.addEventListener("click", () => {
      if (!lastCheckboxResults || lastCheckboxResults.length === 0) return;

      // Build merged data: existing extraction + checkbox states
      const mergedData = {};
      // Add existing extraction data first
      if (typeof currentExtractionData !== "undefined") {
        for (const [k, v] of Object.entries(currentExtractionData)) {
          if (k !== "_meta") mergedData[k] = v;
        }
      }
      // Add checkbox results
      lastCheckboxResults.forEach(cb => {
        mergedData[cb.label] = cb.checked ? "Checked" : "Unchecked";
      });

      // Re-render main results with merged data
      currentExtractionData = mergedData;
      renderResults(resultsContainer, mergedData);

      // Visual feedback
      const btn = document.getElementById("merge-checkboxes-btn");
      if (btn) {
        btn.textContent = "✓ Merged!";
        btn.style.background = "rgba(34, 197, 94, 0.15)";
        btn.style.color = "#22c55e";
        setTimeout(() => {
          btn.textContent = "🔗 Merge into Results";
          btn.style.background = "";
          btn.style.color = "";
        }, 2000);
      }

      statusEl.textContent = `Merged ${lastCheckboxResults.length} checkboxes into results`;
    });

    // JSON export data
    const getExportData = () => lastCheckboxResults.map(c => ({
      label: c.label,
      checked: c.checked,
    }));

    // Copy JSON
    document.getElementById("copy-checkbox-json")?.addEventListener("click", () => {
      navigator.clipboard.writeText(JSON.stringify(getExportData(), null, 2));
      const btn = document.getElementById("copy-checkbox-json");
      btn.textContent = "✓ Copied!";
      setTimeout(() => { btn.textContent = "📋 Copy JSON"; }, 2000);
    });

    // Download JSON
    document.getElementById("download-checkbox-json")?.addEventListener("click", () => {
      const blob = new Blob([JSON.stringify(getExportData(), null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `checkboxes_${selectedFile?.name?.replace('.pdf', '') || 'doc'}.json`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }


  // Render Document Preview (simple PDF page image)
  async function renderLayoutPreview() {
    const previewImg = document.getElementById("pdf-preview-img");
    if (!selectedFile) {
      console.log("No file selected for preview");
      noPreview.style.display = "block";
      previewImg.style.display = "none";
      return;
    }

    try {
      console.log("Rendering PDF preview...");
      noPreview.style.display = "none";

      // Render PDF page as image at good quality
      const imageDataUrl = await pdfPageToImage(selectedFile, 1, 2.0);
      previewImg.src = imageDataUrl;
      previewImg.style.display = "block";

      console.log("Document preview rendered");
    } catch (err) {
      console.error("✗ Preview error:", err);
      noPreview.style.display = "block";
      noPreview.textContent = `Error: ${err.message}`;
      previewImg.style.display = "none";
    }
  }

  // Extract Button Handler
  extractBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    await extractWithVision();
  });

  // AI Vision Extraction (PaddleOCR-VL / Qwen2-VL)
  async function extractWithVision() {
    try {
      statusEl.textContent = "Checking backend connection...";
      extractBtn.disabled = true;
      resultsContainer.innerHTML = "";

      // Check if backend is available
      const health = await checkBackendHealth();

      if (!health) {
        throw new Error(
          "Backend server not running. Please start the backend with: cd backend && python server.py",
        );
      }

      // Read selected model from radio buttons
      const selectedModel =
        document.querySelector('input[name="model-select"]:checked')?.value || "paddleocr";

      // Validate model availability
      const models = health.models || {};
      const modelInfo = models[selectedModel];
      if (modelInfo && !modelInfo.available) {
        throw new Error(`${modelInfo.name || selectedModel} not available on backend`);
      }

      // Backward compat: check old health format too
      if (!health.models && !health.model_available) {
        throw new Error(`${health.model || "AI model"} not loaded on backend`);
      }

      const modelName = modelInfo?.name || health.model || "AI Model";

      // Check voting checkbox
      const votingChecked = document.getElementById("voting-checkbox")?.checked && selectedModel === "qwen";
      const votingRounds = votingChecked ? 3 : 1;
      if (votingChecked) {
        statusEl.textContent = `Extracting with ${modelName} (3× accuracy boost)...`;
      } else {
        statusEl.textContent = `Extracting with ${modelName}...`;
      }

      const extractedData = await extractFields(selectedFile, CURRENT_FIELDS, selectedModel, votingRounds, checkboxEnabled);

      console.log(
        `--- ${modelName.toUpperCase()} OUTPUT (JSON) ---`,
      );
      console.log(JSON.stringify(extractedData, null, 2));
      console.log("------------------------------");

      const timeSec = extractedData?._meta?.time_seconds;
      renderResults(resultsContainer, extractedData);
      statusEl.textContent = `Extraction complete!${timeSec ? ` (${timeSec}s)` : ""}`;
    } catch (err) {
      console.error("Extraction error:", err);
      statusEl.textContent = `Error: ${err.message}`;
    } finally {
      extractBtn.disabled = false;
    }
  }

  // NOTE: extractWithText is disabled — it relied on client-side functions
  // (extractTextFromPDF, performOCR, extractStructuredData, scanForKeyValuePairs)
  // that were never implemented. All extraction now goes through extractWithVision
  // which delegates to the Python backend.
  // To re-enable, implement the missing functions and uncomment below.
  /*
  async function extractWithText() {
    if (!selectedFile) return;
    try {
      statusEl.textContent = "Parsing PDF text...";
      extractBtn.disabled = true;
      resultsContainer.innerHTML = "";
      scanResultsContainer.innerHTML = "";
      scanResultsSection.style.display = "none";
      let text = await extractTextFromPDF(selectedFile);
      if (!text || text.trim().length === 0) {
        statusEl.textContent = "Scanned document detected. Running OCR...";
        text = await performOCR(selectedFile, (status) => { statusEl.textContent = status; });
      }
      if (!text || text.trim().length === 0) {
        throw new Error("Could not extract text from document (OCR failed).");
      }
      const sanitizedText = sanitizeInput(text);
      statusEl.textContent = "Extracting requested fields...";
      const aiData = await extractStructuredData({ text: sanitizedText, fields: CURRENT_FIELDS });
      statusEl.textContent = "Running Smart Scan...";
      const scanData = scanForKeyValuePairs(sanitizedText);
      const mergedData = { ...scanData, ...aiData };
      Object.keys(aiData).forEach((key) => {
        const aiVal = aiData[key];
        const isInvalid = !aiVal || aiVal.length < 2 || aiVal.toLowerCase().includes("not found");
        if (isInvalid) {
          const matchKey = Object.keys(scanData).find((k) => k.toLowerCase().includes(key.toLowerCase()));
          if (matchKey && scanData[matchKey]) { mergedData[key] = scanData[matchKey]; }
        }
      });
      console.log("--- FINAL EXTRACTION OUTPUT (JSON) ---");
      console.log(JSON.stringify(mergedData, null, 2));
      console.log("--------------------------------------");
      renderResults(resultsContainer, mergedData);
      statusEl.textContent = "Complete (Check console for JSON)";
    } catch (err) {
      console.error(err);
      statusEl.textContent = `Error: ${err.message}`;
    } finally {
      extractBtn.disabled = false;
    }
  }
  */

  // Current extraction data (for JSON display)
  let currentExtractionData = {};

  function renderResults(container, data) {
    currentExtractionData = { ...data };

    if (!data || Object.keys(data).length === 0) {
      container.innerHTML = `
        <div class="empty-results animate-fadeUp">
          <div style="font-size: 2rem; margin-bottom: 8px;">📭</div>
          <p>No data extracted yet</p>
        </div>
      `;
      return;
    }

    // Extract _meta for signals and normalized values
    const meta = data._meta || {};
    const normalizedValues = meta.normalized_values || {};
    const metaSignals = meta.signals || {};
    const flaggedFields = meta.flagged_fields || [];
    const hasNormalized = Object.keys(normalizedValues).length > 0;

    // Track format state
    let showFormatted = false;

    // Toggle button (only shows if there are normalized values)
    let html = "";
    if (hasNormalized) {
      html += `
        <div class="format-toggle-bar animate-fadeUp" style="display: flex; align-items: center; gap: 10px; margin-bottom: 14px; padding: 8px 14px; background: var(--surface, #f8f9fa); border-radius: 10px; border: 1px solid var(--border, #e2e8f0);">
          <span style="font-size: 0.8rem; color: var(--text-muted, #64748b);">Output Format:</span>
          <button id="format-toggle-btn" class="btn-secondary" style="padding: 5px 14px; font-size: 0.78rem; border-radius: 8px; display: flex; align-items: center; gap: 6px; transition: all 0.2s ease;">
            <span id="format-toggle-icon" style="font-size: 0.9rem;">${icons.file(14)}</span>
            <span id="format-toggle-label">Raw</span>
          </button>
          <span id="format-hint" style="font-size: 0.72rem; color: var(--text-muted, #94a3b8); margin-left: auto;">${Object.keys(normalizedValues).length} field(s) can be formatted</span>
        </div>
      `;
    }

    html += '<div class="results-grid">';
    let idx = 0;
    for (const [key, value] of Object.entries(data)) {
      // Skip _meta — it's not a field
      if (key === "_meta") continue;

      const displayValue = value && value !== "None" ? value : "—";
      const safeValue = value && value !== "None" ? value : "";
      
      // Build signal display
      const fieldSignal = metaSignals[key] || {};
      const source = fieldSignal.source || "";
      const flags = fieldSignal.flags || [];
      const isFlagged = flaggedFields.includes(key);
      
      // Source badge icon mapping
      const sourceIcons = {
        batch: "⚡", fallback: "🔍", voting: "🗳️", paddleocr: "📄",
        checkbox_batch: "☑", checkbox_vqa: "🔍", unknown: "❓",
      };
      const sourceIcon = sourceIcons[source] || "";
      const sourceBadge = source ? `<span class="signal-badge">${sourceIcon} ${source}</span>` : "";
      
      // Flag pills
      const flagPills = flags.map(f => {
        const flagIcons = {
          empty_value: "⭕", fallback_recovery: "🔄", voting_disagreed: "⚖️",
          validation_error: "❌", vqa_fallback: "🔍", not_found: "❓",
          fuzzy_match: "≈",
        };
        return `<span class="flag-pill">${flagIcons[f] || "⚠"} ${f.replace(/_/g, " ")}</span>`;
      }).join("");

      // Check if this field has a different normalized value
      const normalizedVal = normalizedValues[key];
      const hasFormat = normalizedVal && normalizedVal !== value;
      const formatIndicator = hasFormat
        ? `<span class="format-indicator" style="display: none; font-size: 0.65rem; color: var(--accent, #6366f1); margin-left: 6px; opacity: 0.7;" title="Formatted by validator">${icons.sparkle(11)}</span>`
        : "";

      html += `
        <div class="result-card animate-fadeUp${isFlagged ? ' flagged' : ''}" data-field="${key}" data-raw="${safeValue}" data-normalized="${hasFormat ? normalizedVal : safeValue}" style="animation-delay: ${idx * 0.03}s;">
          <div class="card-header">
            <span class="field-name">${key}${formatIndicator}</span>
            ${sourceBadge}
          </div>
          ${flagPills ? `<div class="flag-pills">${flagPills}</div>` : ""}
          <div class="field-value">${displayValue}</div>
          <div class="card-footer">
            <button class="btn-icon btn-flag" data-field="${key}" data-value="${safeValue}" data-signal="${source || 'manual_flag'}" title="Flag for review">${icons.alertCircle(14)}</button>
            <button class="btn-icon btn-delete-result" data-field="${key}" title="Remove">✕</button>
          </div>
        </div>
      `;
      idx++;
    }
    html += "</div>";

    // JSON Actions (hidden JSON, just buttons)
    // Exclude _meta from the JSON export
    const exportData = {};
    for (const [k, v] of Object.entries(currentExtractionData)) {
      if (k !== "_meta") exportData[k] = v;
    }
    html += `
      <div class="json-actions animate-fadeUp" style="animation-delay: ${idx * 0.03}s;">
        <button id="copy-json" class="btn-secondary">${icons.copy(14)} Copy JSON</button>
        <button id="download-json" class="btn-secondary">${icons.download(14)} Download JSON</button>
        <button id="toggle-json" class="btn-secondary">${icons.eye(14)} View JSON</button>
      </div>
      <pre id="json-output" class="json-code" style="display: none;">${JSON.stringify(exportData, null, 2)}</pre>
    `;

    container.innerHTML = html;

    // Format toggle button
    const toggleBtn = document.getElementById("format-toggle-btn");
    if (toggleBtn) {
      toggleBtn.addEventListener("click", () => {
        showFormatted = !showFormatted;
        const icon = document.getElementById("format-toggle-icon");
        const label = document.getElementById("format-toggle-label");
        const hint = document.getElementById("format-hint");

        if (showFormatted) {
          icon.innerHTML = icons.sparkle(14);
          label.textContent = "Formatted";
          toggleBtn.style.background = "var(--accent, #6366f1)";
          toggleBtn.style.color = "white";
          if (hint) hint.textContent = "Showing validated & normalized values";
        } else {
          icon.innerHTML = icons.file(14);
          label.textContent = "Raw";
          toggleBtn.style.background = "";
          toggleBtn.style.color = "";
          if (hint) hint.textContent = `${Object.keys(normalizedValues).length} field(s) can be formatted`;
        }

        // Swap values on all cards
        container.querySelectorAll(".result-card").forEach((card) => {
          const field = card.dataset.field;
          const raw = card.dataset.raw;
          const normalized = card.dataset.normalized;
          const valueEl = card.querySelector(".field-value");
          const indicator = card.querySelector(".format-indicator");

          if (raw !== normalized) {
            valueEl.textContent = showFormatted ? (normalized || raw || "—") : (raw || "—");
            // Show sparkle indicator when formatted
            if (indicator) indicator.style.display = showFormatted ? "inline" : "none";
          }
        });

        // Update JSON preview
        const jsonOut = document.getElementById("json-output");
        if (jsonOut) {
          const jsonData = {};
          container.querySelectorAll(".result-card").forEach((card) => {
            const field = card.dataset.field;
            const raw = card.dataset.raw;
            const normalized = card.dataset.normalized;
            jsonData[field] = showFormatted ? (normalized || raw) : raw;
          });
          jsonOut.textContent = JSON.stringify(jsonData, null, 2);
        }
      });
    }

    // Flag buttons
    container.querySelectorAll(".btn-flag").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const fieldName = btn.dataset.field;
        const aiValue = btn.dataset.value;
        const signal = btn.dataset.signal || "manual_flag";
        const filename = selectedFile ? selectedFile.name : "unknown.pdf";

        try {
          btn.innerHTML = icons.loader(14);
          const res = await fetch("http://localhost:8000/api/flag", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              filename,
              field_name: fieldName,
              ai_value: aiValue,
              signal: signal,
              reason: "Manually flagged for review",
            }),
          });
          if (res.ok) {
            btn.innerHTML = icons.check(14);
            btn.disabled = true;
            btn.style.background = "var(--accent)";
            btn.style.color = "white";
          }
        } catch (err) {
          console.error("Failed to flag:", err);
          btn.innerHTML = icons.alertCircle(14);
        }
      });
    });

    // Delete buttons
    container.querySelectorAll(".btn-delete-result").forEach((btn) => {
      btn.addEventListener("click", () => {
        const field = btn.dataset.field;
        const item = btn.closest(".result-card");
        item.style.animation = "slideOut 0.3s ease-out forwards";
        setTimeout(() => {
          delete currentExtractionData[field];
          const jsonOut = document.getElementById("json-output");
          if (jsonOut)
            jsonOut.textContent = JSON.stringify(
              currentExtractionData,
              null,
              2,
            );
          item.remove();
        }, 300);
      });
    });

    // Copy JSON
    document.getElementById("copy-json")?.addEventListener("click", () => {
      navigator.clipboard.writeText(
        JSON.stringify(currentExtractionData, null, 2),
      );
      const btn = document.getElementById("copy-json");
      btn.innerHTML = `${icons.check(14)} Copied!`;
      setTimeout(() => (btn.innerHTML = `${icons.copy(14)} Copy JSON`), 2000);
    });

    // Download JSON
    document.getElementById("download-json")?.addEventListener("click", () => {
      const blob = new Blob([JSON.stringify(currentExtractionData, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download =
        (selectedFile?.name.replace(".pdf", "") || "extraction") + ".json";
      a.click();
      URL.revokeObjectURL(url);
    });

    // Toggle JSON view
    document.getElementById("toggle-json")?.addEventListener("click", () => {
      const jsonOut = document.getElementById("json-output");
      const btn = document.getElementById("toggle-json");
      if (jsonOut.style.display === "none") {
        jsonOut.style.display = "block";
        btn.innerHTML = `${icons.x(14)} Hide JSON`;
      } else {
        jsonOut.style.display = "none";
        btn.innerHTML = `${icons.eye(14)} View JSON`;
      }
    });
  }

  // Step Loading Display
  function updateSteps(steps) {
    let html = '<div class="step-list">';
    steps.forEach((step, i) => {
      const icon = step.done ? icons.checkCircle(14) : step.active ? icons.loader(14) : icons.circle(14);
      const cls = step.done ? "done" : step.active ? "active" : "";
      html += `<div class="step-item ${cls}">${icon} ${step.label}</div>`;
    });
    html += "</div>";
    statusEl.innerHTML = html;
  }
}

if (typeof document !== "undefined") {
  renderApp();
}
