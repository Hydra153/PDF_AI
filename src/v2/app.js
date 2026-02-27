import { sanitizeInput } from "../services/utils.js";
import { icons } from "../components/icons.js";
import {
  extractFields,
  autoFindFields as autoFindFieldsAPI,
  checkBackendHealth,
  detectCheckboxes,
  findTables,
  askQuestion,
  reExtractField,
  classifyDocument,
  exportCSV,
} from "../services/api/backend.js";
import { ReviewQueue } from "../components/review_queue.js";
import { createChat } from "../components/chat.js";
import {
  extractLayoutFromPDF,
  pdfPageToImage,
  getPdfPageCount,
} from "../services/pdf_helpers.js";

// Fields to Extract (starts empty - use Auto-Find or add manually)
let CURRENT_FIELDS = [];

function renderApp() {
  const root = document.getElementById("app");
  if (!root) return;

  root.innerHTML = `
    <!-- Header Bar -->
    <header class="rd-header">
      <div class="rd-header-logo">
        <div class="rd-logo-icon">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="3" y="1" width="15" height="20" rx="2.5" fill="rgba(255,255,255,0.12)" stroke="white" stroke-width="1.4"/>
            <rect x="6" y="3" width="15" height="20" rx="2.5" fill="rgba(255,255,255,0.2)" stroke="white" stroke-width="1.4"/>
            <path d="M10 9h8M10 12.5h6" stroke="white" stroke-width="1.3" stroke-linecap="round"/>
            <line x1="9" y1="16.5" x2="19" y2="16.5" stroke="#4ecca3" stroke-width="2" stroke-linecap="round" opacity="0.9"/>
          </svg>
        </div>
        <span>ReaDox</span>
        <button id="theme-toggle" class="theme-toggle" title="Toggle dark mode">
          <svg id="theme-icon-sun" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
          <svg id="theme-icon-moon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="display:none;"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
        </button>
      </div>

      <nav class="rd-header-nav">
        <button id="tab-extract" class="rd-nav-btn active">${icons.clipboard(14)} Extract</button>
        <button id="tab-review" class="rd-nav-btn">${icons.eye(14)} Review Queue</button>
      </nav>

      <div class="rd-header-right">
        <span id="rd-status-dot" class="rd-dot"></span>
        <span id="rd-status-label">Loading...</span>
      </div>
    </header>

    <div class="app">
      
      <!-- Extract View -->
      <div id="view-extract">

      <section class="panel">
        <div class="upload-area" id="drop-zone">
          <input type="file" id="file-input" accept="application/pdf,image/png,image/jpeg,image/jpg,image/tiff,image/bmp,image/webp" style="display: none" />
          <button id="upload-btn">Choose File</button>
          <p id="file-name">No file selected</p>
        </div>

        <!-- Hidden status elements (synced to header bar) -->
        <div id="status-indicator" style="display:none;">
          <span id="status-dot"></span>
          <span id="status-text"></span>
          <small id="status-sub"></small>
        </div>
      </section>

      <!-- Global Settings (applies to ALL extractions) -->
      <div id="global-settings" class="panel" style="padding: 8px 14px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; background: var(--surface, #f8fafc); border: 1px solid var(--border, #e2e8f0); border-radius: 10px; margin-bottom: 2px;">
        <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
          <input type="checkbox" id="voting-checkbox" style="accent-color: #4a90e2; width: 16px; height: 16px;" />
          <span style="font-weight: 500; font-size: 13px;">Accuracy Boost</span>
          <span class="info-tooltip" style="position: relative; display: inline-flex; align-items: center; cursor: help;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#888" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
            <span class="info-tooltip-text">3× voting passes — slower but more accurate</span>
          </span>
        </label>
        <label style="display: flex; align-items: center; gap: 6px; cursor: pointer;">
          <input type="checkbox" id="raw-mode-checkbox" style="accent-color: #e67e22; width: 14px; height: 14px;" />
          <span style="font-weight: 500; font-size: 11px;">🎨 Color Document</span>
          <span class="info-tooltip" style="position: relative; display: inline-flex; align-items: center; cursor: help;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#888" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
            <span class="info-tooltip-text">Enable for color forms, photos, or highlighted docs. Disables B&W conversion.</span>
          </span>
        </label>
      </div>

      <div id="extraction-wrapper" class="panel" style="padding: 10px; overflow: visible; display: flex; flex-direction: column; gap: 2px;">

      <!-- Extraction Tab Bar -->
      <nav class="extract-tabs">
        <button class="extract-tab active" data-tab="fields">${icons.clipboard(14)} Fields</button>
        <button class="extract-tab" data-tab="checkboxes">${icons.checkCircle(14)} Checkboxes</button>
        <button class="extract-tab" data-tab="tables">${icons.table ? icons.table(14) : '⊞'} Tables</button>
      </nav>

      <!-- Fields Tab Content -->
      <div id="tab-content-fields" class="extract-tab-content">
        <section class="panel" id="fields-panel" style="box-shadow: none;">
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
          <div class="actions" style="margin-top: 12px;">
             <button id="extract-btn" disabled>Extract Data</button>
          </div>
        </section>

        <section class="panel output-panel" style="box-shadow: none;">
          <div class="status">
            <span data-status>Idle</span>
          </div>
          <div id="results-container" class="results-grid"></div>
          
          <div id="scan-results-section" style="display:none; margin-top: 24px;">
              <p class="label">Smart Scan (Detected Patterns)</p>
              <div id="scan-results-container" class="results-grid"></div>
          </div>
        </section>
      </div>

      <!-- Checkboxes Tab Content -->
      <div id="tab-content-checkboxes" class="extract-tab-content" style="display: none;">
        <section class="panel" style="box-shadow: none;">
          <label class="label">Checkbox Detection</label>
          <p style="font-size: 0.8rem; color: var(--muted); margin: 0 0 12px 0;">Detect all checkboxes and their states in the document</p>
          <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <button id="detect-checkboxes-btn" disabled style="background: #8e44ad; color: white; border: none; padding: 10px 18px; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 0.85rem; display: flex; align-items: center; gap: 6px; transition: all 0.2s;">${icons.checkCircle(14)} Find All Checkboxes</button>
          </div>
        </section>
        <section class="panel" style="box-shadow: none;">
          <div id="checkbox-results-container"></div>
        </section>
      </div>

      <!-- Tables Tab Content -->
      <div id="tab-content-tables" class="extract-tab-content" style="display: none;">
        <section class="panel" style="box-shadow: none;">
          <label class="label">Table Detection</label>
          <p style="font-size: 0.8rem; color: var(--muted); margin: 0 0 12px 0;">Scan all pages and detect data tables — shows column headers and row structure</p>
          <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <button id="find-tables-btn" disabled style="background: #16a085; color: white; border: none; padding: 10px 18px; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 0.85rem; display: flex; align-items: center; gap: 6px; transition: all 0.2s;">⊞ Find All Tables</button>
          </div>
        </section>
        <section class="panel" style="box-shadow: none;">
          <div id="table-scan-results-container"></div>
        </section>
      </div>

      </div>

      <!-- Unified Final Export Bar (below tabs, above doc preview) -->
      <div id="final-export-bar" class="panel" style="padding: 10px 14px; display: none; align-items: center; gap: 10px; flex-wrap: wrap; background: var(--surface, #f8fafc); border: 1px solid var(--border, #e2e8f0); border-radius: 10px; margin-bottom: 2px;">
        <button id="copy-final-json" class="btn-secondary" style="font-weight: 600; padding: 7px 16px; font-size: 0.82rem; display: flex; align-items: center; gap: 6px;">${icons.copy(14)} Copy Final JSON</button>
        <div style="position: relative; display: inline-block;">
          <button id="final-export-dropdown-btn" class="btn-secondary" style="background: #27ae60; color: #fff; padding: 7px 16px; font-size: 0.82rem; font-weight: 600;">${icons.download(14)} Export ▾</button>
          <div id="final-export-dropdown-menu" style="display: none; position: absolute; bottom: 100%; left: 0; z-index: 100; background: var(--surface, #fff); border: 1px solid var(--border, #e2e8f0); border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); min-width: 140px; margin-bottom: 4px; overflow: hidden;">
            <button id="final-download-json" style="display: block; width: 100%; text-align: left; padding: 8px 14px; border: none; background: none; cursor: pointer; font-size: 0.82rem; color: var(--text, #1e293b);" onmouseover="this.style.background='var(--surface-hover, #f1f5f9)'" onmouseout="this.style.background='none'">${icons.file(14)} JSON</button>
            <button id="final-export-csv" style="display: block; width: 100%; text-align: left; padding: 8px 14px; border: none; background: none; cursor: pointer; font-size: 0.82rem; color: var(--text, #1e293b);" onmouseover="this.style.background='var(--surface-hover, #f1f5f9)'" onmouseout="this.style.background='none'">${icons.table ? icons.table(14) : '▇'} CSV</button>
          </div>
        </div>
        <button id="toggle-final-json" class="btn-secondary" style="padding: 7px 14px; font-size: 0.82rem;">${icons.eye(14)} View JSON</button>
        <span id="final-export-hint" style="font-size: 0.72rem; color: var(--text-muted, #94a3b8); margin-left: auto;">Combines fields + checkboxes + tables</span>
      </div>
      <pre id="final-json-output" class="json-code" style="display: none;"></pre>

      <section class="panel">
        <label class="label">Document Preview</label>
        <div id="layout-preview-container" style="margin-top: 16px; background: #fff; padding: 20px; border-radius: 8px; position: relative;">
            <div id="no-preview" style="text-align: center; padding: 40px; color: #999;">
                No Preview
            </div>
            <img id="pdf-preview-img" style="display: none; max-width: 100%; border: 2px solid #3498db; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" alt="Document Preview" />
            <div id="page-nav" style="display: none; margin-top: 10px; display: none; align-items: center; justify-content: center; gap: 12px; padding: 8px 0;">
              <button id="prev-page-btn" class="btn-secondary" style="padding: 4px 14px; font-size: 0.85rem; border-radius: 6px; min-width: 36px;" disabled>←</button>
              <span id="page-indicator" style="font-size: 0.82rem; color: var(--text-muted, #64748b); user-select: none;">Page 1 of 1</span>
              <button id="next-page-btn" class="btn-secondary" style="padding: 4px 14px; font-size: 0.85rem; border-radius: 6px; min-width: 36px;" disabled>→</button>
            </div>
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
  let lastTableResults = null;     // Store for table JSON export
  let checkboxEnabled = true;  // Checkbox detection always on
  let currentPage = 1;         // Current preview page (1-based)
  let totalPageCount = 1;      // Total pages in current document

  // ─── Build Final Combined JSON ───
  function buildFinalJSON() {
    const output = {};

    // Fields section
    const fieldsData = {};
    for (const [k, v] of Object.entries(currentExtractionData || {})) {
      if (k === "_meta") continue;
      fieldsData[k] = v;
    }
    if (Object.keys(fieldsData).length > 0) {
      output.fields = fieldsData;
    }

    // Formatted section (decomposed fields)
    const formattedData = {};
    for (const [k, v] of Object.entries(currentExtractionData || {})) {
      if (k === "_meta") continue;
      const decomposed = decomposeField(k, v);
      if (decomposed) {
        const sub = {};
        for (const part of decomposed) {
          sub[part.label] = part.value;
        }
        formattedData[k] = sub;
      }
    }
    if (Object.keys(formattedData).length > 0) {
      output.formatted = formattedData;
    }

    // Checkboxes section
    if (lastCheckboxResults && lastCheckboxResults.length > 0) {
      output.checkboxes = {};
      lastCheckboxResults.forEach(cb => {
        output.checkboxes[cb.label] = cb.checked ? "Checked" : "Unchecked";
      });
    }

    // Tables section
    if (lastTableResults && lastTableResults.length > 0) {
      output.tables = lastTableResults.map(t => ({
        name: t.table_name || t.name || "Table",
        columns: t.columns || [],
        rows: t.rows_json || t.rows || [],
      }));
    }

    return output;
  }

  // Show/hide the final export bar based on whether data exists
  function updateFinalExportBar() {
    const bar = document.getElementById("final-export-bar");
    if (!bar) return;
    const hasFields = currentExtractionData && Object.keys(currentExtractionData).some(k => k !== "_meta");
    const hasCheckboxes = lastCheckboxResults && lastCheckboxResults.length > 0;
    const hasTables = lastTableResults && lastTableResults.length > 0;

    if (hasFields || hasCheckboxes || hasTables) {
      bar.style.display = "flex";

      // Update hint with section counts
      const parts = [];
      if (hasFields) parts.push("fields");
      if (hasCheckboxes) parts.push("checkboxes");
      if (hasTables) parts.push("tables");
      const hint = document.getElementById("final-export-hint");
      if (hint) hint.textContent = `Combines: ${parts.join(" + ")}`;
    } else {
      bar.style.display = "none";
    }
  }

  // Wire Final Export bar buttons
  (function wireFinalExportBar() {
    // Copy Final JSON
    document.getElementById("copy-final-json")?.addEventListener("click", () => {
      const data = buildFinalJSON();
      navigator.clipboard.writeText(JSON.stringify(data, null, 2));
      const btn = document.getElementById("copy-final-json");
      btn.innerHTML = `${icons.check(14)} Copied!`;
      setTimeout(() => (btn.innerHTML = `${icons.copy(14)} Copy Final JSON`), 2000);
    });

    // Export dropdown toggle
    const expBtn = document.getElementById("final-export-dropdown-btn");
    const expMenu = document.getElementById("final-export-dropdown-menu");
    if (expBtn && expMenu) {
      expBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        expMenu.style.display = expMenu.style.display === "none" ? "block" : "none";
      });
      document.addEventListener("click", () => { expMenu.style.display = "none"; });
    }

    // Download Final JSON
    document.getElementById("final-download-json")?.addEventListener("click", () => {
      const data = buildFinalJSON();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = (selectedFile?.name?.replace(/\.[^.]+$/, '') || "extraction") + "_final.json";
      a.click();
      URL.revokeObjectURL(url);
    });

    // Final CSV export (flatten all sections)
    document.getElementById("final-export-csv")?.addEventListener("click", async () => {
      const btn = document.getElementById("final-export-csv");
      try {
        btn.textContent = "Exporting...";
        const flat = {};
        const data = buildFinalJSON();
        // Flatten fields
        if (data.fields) Object.assign(flat, data.fields);
        // Flatten formatted
        if (data.formatted) {
          for (const [field, sub] of Object.entries(data.formatted)) {
            for (const [label, val] of Object.entries(sub)) {
              flat[`${field} - ${label}`] = val;
            }
          }
        }
        // Flatten checkboxes
        if (data.checkboxes) {
          for (const [label, val] of Object.entries(data.checkboxes)) {
            flat[`Checkbox: ${label}`] = val;
          }
        }
        await exportCSV(flat, (selectedFile?.name || "extraction") + "_final");
        btn.innerHTML = `${icons.check(14)} Exported!`;
        setTimeout(() => { btn.innerHTML = `${icons.table ? icons.table(14) : '▇'} CSV`; }, 2000);
      } catch (err) {
        btn.textContent = "Failed";
        setTimeout(() => { btn.innerHTML = `${icons.table ? icons.table(14) : '▇'} CSV`; }, 2000);
      }
    });

    // Toggle Final JSON view
    document.getElementById("toggle-final-json")?.addEventListener("click", () => {
      const jsonOut = document.getElementById("final-json-output");
      const btn = document.getElementById("toggle-final-json");
      if (jsonOut.style.display === "none") {
        jsonOut.textContent = JSON.stringify(buildFinalJSON(), null, 2);
        jsonOut.style.display = "block";
        btn.innerHTML = `${icons.x(14)} Hide JSON`;
      } else {
        jsonOut.style.display = "none";
        btn.innerHTML = `${icons.eye(14)} View JSON`;
      }
    });
  })();

  // Tab Navigation (Extract / Review)
  const tabExtract = document.getElementById("tab-extract");
  const tabReview = document.getElementById("tab-review");
  const viewExtract = document.getElementById("view-extract");
  const viewReview = document.getElementById("view-review");
  const reviewContainer = document.getElementById("review-queue-container");

  // ─── Extraction Sub-Tab Navigation (Fields / Checkboxes / Tables) ───
  const extractTabs = document.querySelectorAll(".extract-tab");
  const extractTabContents = {
    fields: document.getElementById("tab-content-fields"),
    checkboxes: document.getElementById("tab-content-checkboxes"),
    tables: document.getElementById("tab-content-tables"),
  };

  const findTablesBtn = document.getElementById("find-tables-btn");
  const tableScanResultsContainer = document.getElementById("table-scan-results-container");

  extractTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.tab;
      // Toggle active class
      extractTabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      // Toggle content visibility
      Object.entries(extractTabContents).forEach(([key, el]) => {
        el.style.display = key === target ? "" : "none";
      });
    });
  });

  // Initialize ReviewQueue
  const reviewQueue = new ReviewQueue(reviewContainer);

  // Initialize Chat Panel
  const chatPanel = createChat(document.body);

  // Open chat by default and pin it
  chatPanel.open();
  chatPanel.pin();

  // ─── Dark/Light Theme Toggle ───
  const themeToggle = document.getElementById("theme-toggle");
  const sunIcon = document.getElementById("theme-icon-sun");
  const moonIcon = document.getElementById("theme-icon-moon");

  function applyTheme(dark) {
    document.body.classList.toggle("dark", dark);
    sunIcon.style.display = dark ? "none" : "";
    moonIcon.style.display = dark ? "" : "none";
    localStorage.setItem("rd-theme", dark ? "dark" : "light");
  }

  // Load saved preference
  const savedTheme = localStorage.getItem("rd-theme");
  if (savedTheme === "dark") applyTheme(true);

  themeToggle.addEventListener("click", () => {
    applyTheme(!document.body.classList.contains("dark"));
  });

  // Clear queue on page load - ensures consistent state
  // Reviews only make sense with a PDF loaded, so start fresh
  // Guard: backend may not be ready yet at page load time
  reviewQueue.clearQueue().catch(() => {});

  // ─── Status Indicator Helper ───
  function updateStatus(state) {
    const dot = document.getElementById("status-dot");
    const text = document.getElementById("status-text");
    const sub = document.getElementById("status-sub");
    const indicator = document.getElementById("status-indicator");
    if (!dot || !text) return;

    const states = {
      loading:    { color: "#f39c12", label: "Loading...",   sub: "Connecting to backend",          bg: "rgba(243, 156, 18, 0.08)", border: "rgba(243, 156, 18, 0.2)" },
      ready:      { color: "#27ae60", label: "Ready",        sub: "Qwen2.5-VL • Ready to extract",  bg: "rgba(39, 174, 96, 0.08)",  border: "rgba(39, 174, 96, 0.2)" },
      processing: { color: "#3498db", label: "Processing...",sub: "Extracting data from document",   bg: "rgba(52, 152, 219, 0.08)", border: "rgba(52, 152, 219, 0.2)" },
      offline:    { color: "#e74c3c", label: "Offline",      sub: "Backend not reachable",           bg: "rgba(231, 76, 60, 0.08)",  border: "rgba(231, 76, 60, 0.2)" },
    };

    const s = states[state] || states.loading;
    dot.style.background = s.color;
    text.textContent = s.label;
    text.style.color = s.color;
    if (sub) sub.textContent = s.sub;
    if (indicator) {
      indicator.style.background = s.bg;
      indicator.style.borderColor = s.border;
    }

    // Also update header bar status
    const headerDot = document.getElementById("rd-status-dot");
    const headerLabel = document.getElementById("rd-status-label");
    if (headerDot) headerDot.style.background = s.color;
    if (headerLabel) { headerLabel.textContent = s.label; headerLabel.style.color = s.color; }
  }

  // ─── Startup: check backend health (with retry polling) ───
  let _healthPollTimer = null;
  async function pollBackendHealth() {
    try {
      const health = await checkBackendHealth();
      if (health) {
        updateStatus("ready");
        if (_healthPollTimer) { clearInterval(_healthPollTimer); _healthPollTimer = null; }
        return;
      }
    } catch { /* still offline */ }
    updateStatus("offline");
  }
  // Initial check + retry every 3s until backend is up
  pollBackendHealth();
  _healthPollTimer = setInterval(pollBackendHealth, 3000);

  // Always use Qwen model for chat
  chatPanel.setModel("qwen");

  // ─── Checkbox Detection (always on) ───
  const analysisPanel = document.getElementById("analysis-panel");
  if (analysisPanel) analysisPanel.style.display = "block";



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
        <div class="field-tag" data-index="${i}" style="cursor: pointer;" title="Click to remove">
            <span>${f.key}</span>
            <button class="remove-field" data-index="${i}">×</button>
        </div>
      `,
    ).join("");

    // Re-attach listeners — whole card is clickable to remove
    document.querySelectorAll(".field-tag").forEach((tag) => {
      tag.addEventListener("click", () => {
        const idx = parseInt(tag.dataset.index);
        CURRENT_FIELDS.splice(idx, 1);
        renderFields();
      });
    });
  }

  // Add Field Logic
  const addField = () => {
    const val = newFieldInput.value.trim();
    if (val) {
      CURRENT_FIELDS.push({ key: val, question: `What is the ${val}?` });
      newFieldInput.value = "";
      renderFields();
    }
  };
  addFieldBtn.addEventListener("click", addField);
  newFieldInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") addField();
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
      <p>Upload a document to extract data</p>
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
    if (files.length > 0 && isSupportedFile(files[0])) {
      await handleFileSelect(files[0]);
    }
  });

  // Paste support — only works when hovering over the drop zone
  let _hoveringDropZone = false;
  dropZone.addEventListener("mouseenter", () => { _hoveringDropZone = true; });
  dropZone.addEventListener("mouseleave", () => { _hoveringDropZone = false; });

  document.addEventListener("paste", async (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;

    // Check if clipboard has an image/file
    let pastedFile = null;
    for (const item of items) {
      if (item.kind === "file") {
        const file = item.getAsFile();
        if (file && isSupportedFile(file)) {
          pastedFile = file;
          break;
        }
      }
    }
    if (!pastedFile) return;

    // Only accept if hovering over drop zone
    if (_hoveringDropZone) {
      e.preventDefault();
      await handleFileSelect(pastedFile);
    } else {
      // Show toast hint
      e.preventDefault();
      const toast = document.createElement("div");
      toast.textContent = "📋 Hover over the file area to paste a document";
      toast.style.cssText = "position:fixed;bottom:24px;left:50%;transform:translateX(-50%);background:#1e293b;color:#fff;padding:10px 20px;border-radius:10px;font-size:0.82rem;z-index:9999;box-shadow:0 4px 12px rgba(0,0,0,0.2);animation:fadeIn 0.2s ease;";
      document.body.appendChild(toast);
      setTimeout(() => toast.remove(), 3000);
    }
  });

  // Helper: check if file is supported (PDF or image)
  function isSupportedFile(file) {
    const supported = [
      "application/pdf",
      "image/png", "image/jpeg", "image/jpg",
      "image/tiff", "image/bmp", "image/webp"
    ];
    return supported.includes(file.type);
  }

  function isImageFile(file) {
    return file.type.startsWith("image/");
  }

  async function handleFileSelect(file) {
    selectedFile = file;
    fileNameDisplay.innerHTML = `${icons.file(14)} <strong>${file.name}</strong>`;
    extractBtn.disabled = false;
    autoFindBtn.disabled = false;
    detectCheckboxesBtn.disabled = false;
    findTablesBtn.disabled = false;
    statusEl.textContent = "Ready to process";

    // Update chat with new file — always clear history on new selection/paste
    chatPanel.clearHistory();
    chatPanel.setFile(file);

    // Clear review queue when new PDF selected
    await reviewQueue.clearQueue();

    // Clear previous results
    resultsContainer.innerHTML = "";
    checkboxResultsContainer.innerHTML = "";
    tableScanResultsContainer.innerHTML = "";
    lastCheckboxResults = null;
    lastTableResults = null;
    currentExtractionData = {};
    updateFinalExportBar();

    // Clear fields from previous document
    CURRENT_FIELDS = [];
    renderFields();

    await renderLayoutPreview();
  }

  fileInput.addEventListener("change", async (e) => {
    if (e.target.files.length > 0) {
      await handleFileSelect(e.target.files[0]);
    }
  });

  // Auto-Find Fields Handler (uses VLM classification)
  autoFindBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    try {
      statusEl.textContent = "Analyzing document...";
      autoFindBtn.disabled = true;

      const health = await checkBackendHealth();
      if (!health) throw new Error("Backend server not running.");

      const result = await classifyDocument(selectedFile);

      if (!result.success || !result.suggested_fields?.length) {
        statusEl.textContent = "No fields detected. Try manual entry.";
        autoFindBtn.disabled = false;
        return;
      }

      CURRENT_FIELDS = result.suggested_fields.map(f => ({ key: f, question: "" }));
      renderFields();

      statusEl.textContent = `${result.doc_type} — ${result.suggested_fields.length} fields found (${result.time_seconds}s)`;
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
      updateFinalExportBar();

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

  // ─── Find All Tables Handler ───
  findTablesBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    try {
      findTablesBtn.disabled = true;
      findTablesBtn.textContent = "⏳ Scanning...";
      statusEl.textContent = "Scanning for tables...";
      tableScanResultsContainer.innerHTML = `
        <div style="text-align: center; padding: 20px; color: var(--muted);">
          <div class="spinner" style="margin: 0 auto 12px;"></div>
          Scanning document for data tables...
        </div>`;

      const result = await findTables(selectedFile);

      if (!result.tables || result.tables.length === 0) {
        tableScanResultsContainer.innerHTML = `
          <div style="text-align: center; padding: 20px; color: var(--muted);">
            ⊟ No data tables found in this document
          </div>`;
        statusEl.textContent = "No tables found";
        return;
      }

      renderTableScanResults(result.tables, result.total_pages, result.time_seconds);
      lastTableResults = result.tables;
      statusEl.textContent = `Found ${result.count} table(s) across ${result.total_pages} page(s) in ${result.time_seconds}s`;
      updateFinalExportBar();

    } catch (err) {
      console.error("Table scan error:", err);
      tableScanResultsContainer.innerHTML = `
        <div style="color: #ef4444; padding: 12px;">❌ ${err.message}</div>`;
      statusEl.textContent = `Error: ${err.message}`;
    } finally {
      findTablesBtn.disabled = false;
      findTablesBtn.textContent = "⊞ Find All Tables";
    }
  });

  function renderTableScanResults(tables, totalPages, timeSec) {
    let html = `
      <div style="display: flex; gap: 12px; margin-bottom: 12px; font-size: 0.8rem; color: var(--muted); align-items: center; flex-wrap: wrap;">
        <span>📊 ${tables.length} table${tables.length !== 1 ? "s" : ""} found</span>
        <span>📄 ${totalPages} page${totalPages !== 1 ? "s" : ""} scanned</span>
        <span style="margin-left: auto;">${timeSec}s</span>
      </div>
      <div style="display: flex; flex-direction: column; gap: 16px;">`;

    tables.forEach((t, i) => {
      const cols = t.columns || [];
      const rowPrefix = t.row_index_prefix || "";
      const rowsJson = t.rows_json || null;

      // ── Try to build a real data table from rows_json ──
      // rows_json can be:
      //   - a JS array  (FastAPI auto-serialized from Python list)
      //   - a JSON string (if backend returned it as str)
      //   - null  (extraction failed)
      let tableBodyHtml = "";
      let rowCount = 0;
      let usedRealData = false;

      if (rowsJson) {
        try {
          // Normalize: accept both already-parsed array and JSON string
          const rows = Array.isArray(rowsJson)
            ? rowsJson
            : (typeof rowsJson === "string" ? JSON.parse(rowsJson) : null);

          if (rows && Array.isArray(rows) && rows.length > 0 && typeof rows[0] === "object") {
            const headers = Object.keys(rows[0]);
            rowCount = rows.length;
            usedRealData = true;

            const thCells = headers.map(h => `<th>${h}</th>`).join("");
            const dataTrs = rows.map(row => {
              const tds = headers.map(h => `<td>${row[h] ?? ""}</td>`).join("");
              return `<tr>${tds}</tr>`;
            }).join("");

            tableBodyHtml = `<thead><tr>${thCells}</tr></thead><tbody>${dataTrs}</tbody>`;
          }
        } catch (e) { /* fall through to skeleton */ }
      }

      // ── Fallback: 3-row skeleton if no real data ──
      if (!usedRealData) {
        const PREVIEW_ROWS = 3;
        const thCells = cols.map(h => `<th>${h}</th>`).join("");
        const skeletonTrs = Array.from({ length: PREVIEW_ROWS }, (_, r) => {
          const rowNum = r + 1;
          const tds = cols.map((col, ci) => {
            const val = (ci === 0 && rowPrefix) ? `${rowPrefix}${rowNum}` : "—";
            return `<td>${val}</td>`;
          }).join("");
          return `<tr>${tds}</tr>`;
        }).join("");
        tableBodyHtml = `<thead><tr>${thCells}</tr></thead><tbody>${skeletonTrs}</tbody>`;
        rowCount = PREVIEW_ROWS;
      }

      html += `
        <div class="result-card table-card animate-fadeUp" style="animation-delay: ${i * 0.05}s; padding: 0; overflow: hidden;">

          <!-- Title Bar -->
          <div class="card-header" style="padding: 12px 16px; border-bottom: 1px solid rgba(128,128,128,0.12);">
            <span class="field-name" style="display: flex; align-items: center; gap: 6px;">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#16a085" stroke-width="2.2"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="9" x2="9" y2="21"/></svg>
              Table — Page ${t.page}
            </span>
            <span class="signal-badge" style="background: rgba(22,160,133,0.12); color: #16a085;">${cols.length} cols · ${rowCount} rows</span>
          </div>

          <!-- Full Table (real data) -->
          <div class="table-value-wrap">
            <table class="mini-table">
              ${tableBodyHtml}
            </table>
            ${!usedRealData ? `<div style="padding: 5px 12px; font-size: 0.68rem; color: var(--muted); font-style: italic; border-top: 1px dashed rgba(128,128,128,0.1);">Skeleton preview — restart backend to extract real rows</div>` : ""}
          </div>

          <!-- Meta footer -->
          <div style="display: flex; gap: 20px; flex-wrap: wrap; padding: 8px 14px; border-top: 1px solid rgba(128,128,128,0.1); font-size: 0.73rem; color: var(--muted);">
            <span><strong>Page:</strong> ${t.page} of ${totalPages}</span>
            ${rowPrefix ? `<span><strong>Row index:</strong> <code style="background: rgba(0,0,0,0.06); padding: 1px 5px; border-radius: 3px;">${rowPrefix}</code></span>` : ""}
            <span style="margin-left: auto;">${usedRealData ? "✅ Real data" : "⚠️ No rows extracted"}</span>
          </div>
        </div>`;
    });

    html += `
      </div>
      <div style="display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap;">
        <button id="copy-table-json" class="btn-secondary" style="font-size: 0.78rem; padding: 6px 14px;">📋 Copy JSON</button>
        <button id="append-table-output" class="btn-secondary" style="font-size: 0.78rem; padding: 6px 14px;">🔗 Append to Output</button>
        <button id="download-table-json" class="btn-secondary" style="font-size: 0.78rem; padding: 6px 14px;">⬇ Export JSON</button>
      </div>`;
    tableScanResultsContainer.innerHTML = html;

    // ── Wire up action buttons ──
    const getTableExportData = () => (lastTableResults || []).map(t => ({
      page: t.page,
      columns: t.columns,
      row_index_prefix: t.row_index_prefix,
      rows: Array.isArray(t.rows_json)
        ? t.rows_json
        : (typeof t.rows_json === "string" ? JSON.parse(t.rows_json) : []),
    }));

    // Copy JSON
    document.getElementById("copy-table-json")?.addEventListener("click", () => {
      const btn = document.getElementById("copy-table-json");
      navigator.clipboard.writeText(JSON.stringify(getTableExportData(), null, 2));
      btn.textContent = "✓ Copied!";
      setTimeout(() => { btn.textContent = "📋 Copy JSON"; }, 2000);
    });

    // Append to Output (merge rows as named fields into main results)
    document.getElementById("append-table-output")?.addEventListener("click", () => {
      const btn = document.getElementById("append-table-output");
      if (!lastTableResults || lastTableResults.length === 0) return;
      const merged = typeof currentExtractionData !== "undefined" ? { ...currentExtractionData } : {};
      lastTableResults.forEach((t, i) => {
        const key = `Table ${i + 1} (Page ${t.page})`;
        merged[key] = JSON.stringify(Array.isArray(t.rows_json) ? t.rows_json : [], null, 2);
      });
      currentExtractionData = merged;
      renderResults(resultsContainer, merged);
      // Switch to Fields tab so user sees results
      document.querySelector(".extract-tab[data-tab='fields']")?.click();
      btn.textContent = "✓ Appended!";
      btn.style.background = "rgba(34, 197, 94, 0.15)";
      btn.style.color = "#22c55e";
      setTimeout(() => {
        btn.textContent = "🔗 Append to Output";
        btn.style.background = "";
        btn.style.color = "";
      }, 2000);
      statusEl.textContent = `Table data appended to output`;
    });

    // Export JSON download
    document.getElementById("download-table-json")?.addEventListener("click", () => {
      const blob = new Blob([JSON.stringify(getTableExportData(), null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `tables_${selectedFile?.name?.replace(/\.[^.]+$/, '') || 'doc'}.json`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }


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


  // Render Document Preview (with multi-page navigation)
  async function renderLayoutPreview(pageNum = 1) {
    const previewImg = document.getElementById("pdf-preview-img");
    const pageNav = document.getElementById("page-nav");
    const pageIndicator = document.getElementById("page-indicator");
    const prevBtn = document.getElementById("prev-page-btn");
    const nextBtn = document.getElementById("next-page-btn");

    if (!selectedFile) {
      console.log("No file selected for preview");
      noPreview.style.display = "block";
      previewImg.style.display = "none";
      if (pageNav) pageNav.style.display = "none";
      return;
    }

    try {
      console.log(`Rendering document preview (page ${pageNum})...`);
      noPreview.style.display = "none";

      if (isImageFile(selectedFile)) {
        // Image file — show directly via object URL (single page)
        const imageUrl = URL.createObjectURL(selectedFile);
        previewImg.src = imageUrl;
        previewImg.onload = () => URL.revokeObjectURL(imageUrl);
        totalPageCount = 1;
        currentPage = 1;
        if (pageNav) pageNav.style.display = "none";
      } else {
        // PDF file — get page count and render requested page
        if (pageNum === 1) {
          totalPageCount = await getPdfPageCount(selectedFile);
        }
        currentPage = Math.max(1, Math.min(pageNum, totalPageCount));
        const imageDataUrl = await pdfPageToImage(selectedFile, currentPage, 2.0);
        previewImg.src = imageDataUrl;

        // Show/hide page nav
        if (totalPageCount > 1 && pageNav) {
          pageNav.style.display = "flex";
          pageIndicator.textContent = `Page ${currentPage} of ${totalPageCount}`;
          prevBtn.disabled = currentPage <= 1;
          nextBtn.disabled = currentPage >= totalPageCount;
        } else if (pageNav) {
          pageNav.style.display = "none";
        }
      }
      previewImg.style.display = "block";

      console.log(`Document preview rendered (page ${currentPage}/${totalPageCount})`);
    } catch (err) {
      console.error("✗ Preview error:", err);
      noPreview.style.display = "block";
      noPreview.textContent = `Error: ${err.message}`;
      previewImg.style.display = "none";
      if (pageNav) pageNav.style.display = "none";
    }
  }

  // Page navigation event handlers
  document.getElementById("prev-page-btn")?.addEventListener("click", () => {
    if (currentPage > 1) renderLayoutPreview(currentPage - 1);
  });
  document.getElementById("next-page-btn")?.addEventListener("click", () => {
    if (currentPage < totalPageCount) renderLayoutPreview(currentPage + 1);
  });

  // Keyboard shortcuts for page navigation (← →)
  document.addEventListener("keydown", (e) => {
    // Don't interfere with input fields
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    if (totalPageCount <= 1) return;
    if (e.key === "ArrowLeft" && currentPage > 1) {
      e.preventDefault();
      renderLayoutPreview(currentPage - 1);
    } else if (e.key === "ArrowRight" && currentPage < totalPageCount) {
      e.preventDefault();
      renderLayoutPreview(currentPage + 1);
    }
  });

  // Extract Button Handler
  extractBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    await extractWithVision();
  });

  // ─── Global Keyboard Shortcuts ───
  document.addEventListener("keydown", (e) => {
    // Ctrl+Enter → Start extraction (works even in inputs)
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      if (selectedFile && !extractBtn.disabled) {
        extractBtn.click();
      }
      return;
    }

    // Escape → Close any open dropdown/modal
    if (e.key === "Escape") {
      // Close presets dropdown if open
      const dropdown = document.querySelector(".presets-dropdown");
      if (dropdown) dropdown.style.display = "none";
      // Close chat panel if open and not pinned
      const chatPanel = document.querySelector(".chat-panel");
      if (chatPanel && !chatPanel.classList.contains("pinned")) {
        chatPanel.style.display = "none";
      }
      return;
    }

    // Don't fire remaining shortcuts when typing in inputs
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

    // Ctrl+Shift+F → Focus "Add Field" input
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === "F") {
      e.preventDefault();
      if (newFieldInput) newFieldInput.focus();
      return;
    }

    // Ctrl+Shift+Q → Focus chat input
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === "Q") {
      e.preventDefault();
      const chatInput = document.querySelector(".chat-input");
      if (chatInput) chatInput.focus();
      return;
    }
  });

  // AI Vision Extraction (Qwen2-VL)
  async function extractWithVision() {
    let progressInterval = null;
    try {
      statusEl.textContent = "Checking backend connection...";
      extractBtn.disabled = true;
      resultsContainer.innerHTML = "";
      updateStatus("processing");

      // Check if backend is available
      const health = await checkBackendHealth();

      if (!health) {
        updateStatus("offline");
        throw new Error(
          "Backend server not running. Please start the backend with: cd backend && python server.py",
        );
      }

      // Show progress bar
      resultsContainer.innerHTML = `
        <div id="extraction-progress" class="animate-fadeUp" style="padding: 20px; text-align: center;">
          <div style="background: var(--border, #e2e8f0); border-radius: 8px; height: 8px; overflow: hidden; margin-bottom: 10px;">
            <div id="progress-fill" style="background: linear-gradient(90deg, #6366f1, #8b5cf6); height: 100%; width: 0%; border-radius: 8px; transition: width 0.5s ease;"></div>
          </div>
          <div id="progress-text" style="font-size: 0.8rem; color: var(--text-muted, #94a3b8);">Starting extraction...</div>
        </div>
      `;

      // Start polling progress
      progressInterval = setInterval(async () => {
        try {
          const res = await fetch("http://localhost:8000/api/progress");
          const prog = await res.json();
          const fill = document.getElementById("progress-fill");
          const text = document.getElementById("progress-text");
          if (fill && prog.active) {
            fill.style.width = `${prog.percent}%`;
          }
          if (text && prog.message) {
            text.textContent = `${prog.message} (${prog.percent}%)`;
          }
        } catch { /* ignore polling errors */ }
      }, 2000);

      const votingChecked = document.getElementById("voting-checkbox")?.checked;
      const votingRounds = votingChecked ? 3 : 1;
      statusEl.textContent = votingChecked
        ? "Extracting with Qwen2.5-VL (3× accuracy boost)..."
        : "Extracting with Qwen2.5-VL...";

      const rawModeChecked = document.getElementById("raw-mode-checkbox")?.checked;
      const multipageEnabled = totalPageCount > 1;
      const extractedData = await extractFields(selectedFile, CURRENT_FIELDS, "qwen", votingRounds, checkboxEnabled, rawModeChecked, multipageEnabled);

      console.log("--- QWEN OUTPUT (JSON) ---");
      console.log(JSON.stringify(extractedData, null, 2));
      console.log("------------------------------");

      const timeSec = extractedData?._meta?.time_seconds;
      const totalPages = extractedData?._meta?.total_pages || 1;
      renderResults(resultsContainer, extractedData);
      statusEl.textContent = `Extraction complete!${timeSec ? ` (${timeSec}s` : ""}${totalPages > 1 ? `, ${totalPages} pages` : ""}${timeSec ? ")" : ""}`;
      updateStatus("ready");
    } catch (err) {
      console.error("Extraction error:", err);
      statusEl.textContent = `Error: ${err.message}`;
      updateStatus("ready");
    } finally {
      if (progressInterval) clearInterval(progressInterval);
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

  // ─── Table Value Renderer ───
  function renderTableValue(jsonStr, tableType) {
    try {
      const data = JSON.parse(jsonStr);

      if (tableType === "table" && Array.isArray(data) && data.length > 0 && typeof data[0] === "object") {
        const headers = Object.keys(data[0]);
        let t = `<div class="table-value-wrap"><table class="mini-table"><thead><tr>`;
        headers.forEach(h => { t += `<th>${h}</th>`; });
        t += `</tr></thead><tbody>`;
        data.forEach(row => {
          t += `<tr>`;
          headers.forEach(h => { t += `<td>${row[h] ?? ""}</td>`; });
          t += `</tr>`;
        });
        t += `</tbody></table></div>`;
        return t;
      }

      if (tableType === "column" && Array.isArray(data)) {
        let t = `<div class="table-value-wrap"><table class="mini-table"><tbody>`;
        data.forEach((v, i) => { t += `<tr><td class="row-idx">${i + 1}</td><td>${v}</td></tr>`; });
        t += `</tbody></table></div>`;
        return t;
      }

      if (tableType === "row" && typeof data === "object" && !Array.isArray(data)) {
        const keys = Object.keys(data);
        let t = `<div class="table-value-wrap"><table class="mini-table"><thead><tr>`;
        keys.forEach(k => { t += `<th>${k}</th>`; });
        t += `</tr></thead><tbody><tr>`;
        keys.forEach(k => { t += `<td>${data[k] ?? ""}</td>`; });
        t += `</tr></tbody></table></div>`;
        return t;
      }
    } catch (e) {
      // Not valid JSON — fall through
    }
    // Fallback: show raw value
    return `<div class="field-value">${jsonStr}</div>`;
  }

  // ─── Smart Field Decomposition ───
  // Detects composite values and breaks them into visual sub-parts
  function decomposeField(fieldName, value) {
    if (!value || typeof value !== "string" || value === "—") return null;
    const fLower = fieldName.toLowerCase();
    
    // Sex/DOB/Age: "Female 01/02/1938 87 Years"
    const sexDobAgeMatch = value.match(/^(Male|Female|M|F)\s+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\s+(\d{1,3})\s*(Years?|yrs?)?$/i);
    if (sexDobAgeMatch || (fLower.includes("sex") && fLower.includes("dob"))) {
      if (sexDobAgeMatch) {
        return [
          { label: "Sex", value: sexDobAgeMatch[1] },
          { label: "DOB", value: sexDobAgeMatch[2] },
          { label: "Age", value: `${sexDobAgeMatch[3]} Years` },
        ];
      }
    }

    // Address: multi-line or "STREET CITY, STATE ZIP"
    if (fLower.includes("address") || fLower.includes("location")) {
      // Multi-line address
      const lines = value.split(/\n|\\n/).map(l => l.trim()).filter(Boolean);
      if (lines.length >= 2) {
        const lastLine = lines[lines.length - 1];
        const cityStateZip = lastLine.match(/^(.+?),?\s+([A-Z]{2})\s+(\d{5}(?:-\d{4})?)$/);
        if (cityStateZip) {
          const parts = [
            ...lines.slice(0, lines.length - 1).map((l, i) => ({ label: i === 0 ? "Street" : `Line ${i + 1}`, value: l })),
            { label: "City", value: cityStateZip[1] },
            { label: "State", value: cityStateZip[2] },
            { label: "Zip", value: cityStateZip[3] },
          ];
          return parts;
        }
        return lines.map((l, i) => ({ label: `Line ${i + 1}`, value: l }));
      }
      // Single-line: "STREET ADDRESS CITY, STATE ZIP"
      // Split on the LAST comma before STATE ZIP to separate street from city
      const singleLine = value.match(/^(.+)\s+([A-Za-z][A-Za-z .]+?)\s*,\s*([A-Z]{2})\s+(\d{5}(?:-\d{4})?)$/);
      if (singleLine) {
        return [
          { label: "Street", value: singleLine[1].replace(/,\s*$/, "") },
          { label: "City", value: singleLine[2] },
          { label: "State", value: singleLine[3] },
          { label: "Zip", value: singleLine[4] },
        ];
      }
      // Fallback: just CITY, STATE ZIP (no street)
      const cityOnly = value.match(/^([A-Za-z][A-Za-z .]+?)\s*,\s*([A-Z]{2})\s+(\d{5}(?:-\d{4})?)$/);
      if (cityOnly) {
        return [
          { label: "City", value: cityOnly[1] },
          { label: "State", value: cityOnly[2] },
          { label: "Zip", value: cityOnly[3] },
        ];
      }
    }

    // Name: "LAST, FIRST" → separate
    if (fLower.includes("name") && value.includes(",")) {
      const parts = value.split(",").map(s => s.trim());
      if (parts.length === 2 && parts[0].length > 1 && parts[1].length > 1) {
        return [
          { label: "Last Name", value: parts[0] },
          { label: "First Name", value: parts[1] },
        ];
      }
    }

    return null; // Not a composite field
  }

  // Format decomposed parts as HTML with visual markers
  function renderDecomposed(parts) {
    return `<div class="decomposed-parts" style="display: flex; flex-direction: column; gap: 5px; margin-top: 6px; padding: 8px 10px; background: var(--surface-alt, #f1f5f9); border-radius: 8px; border-left: 3px solid var(--accent, #6366f1);">${
      parts.map(p => `<div style="display: flex; gap: 8px; align-items: baseline;"><span style="font-size: 0.68rem; color: var(--accent, #6366f1); min-width: 52px; text-align: right; font-weight: 600; letter-spacing: 0.3px; text-transform: uppercase;">${p.label}</span><span style="font-size: 0.82rem; font-weight: 500; color: var(--text, #1e293b);">${p.value}</span></div>`).join("")
    }</div>`;
  }

  function renderResults(container, data) {
    currentExtractionData = { ...data };
    updateFinalExportBar();

    if (!data || Object.keys(data).length === 0) {
      container.innerHTML = `
        <div class="empty-results animate-fadeUp">
          <div style="font-size: 2rem; margin-bottom: 8px;">📭</div>
          <p>No data extracted yet</p>
        </div>
      `;
      return;
    }

    // Extract _meta for confidence and normalized values
    const meta = data._meta || {};
    const normalizedValues = meta.normalized_values || {};
    const confidenceScores = meta.confidence || {};
    const flaggedFields = meta.flagged_fields || [];
    const totalPages = meta.total_pages || 1;
    const hasNormalized = Object.keys(normalizedValues).length > 0;

    // Count fields that can be deep-parsed
    let decomposableCount = 0;
    for (const [key, value] of Object.entries(data)) {
      if (key === "_meta") continue;
      if (decomposeField(key, value)) decomposableCount++;
    }
    const hasFormattable = hasNormalized || decomposableCount > 0;
    const formattableCount = Object.keys(normalizedValues).length + decomposableCount;

    // Track format state
    let showFormatted = false;

    // Toggle button (shows if there are normalized values OR decomposable fields)
    let html = "";
    if (hasFormattable) {
      html += `
        <div class="format-toggle-bar animate-fadeUp" style="display: flex; align-items: center; gap: 10px; margin-bottom: 14px; padding: 8px 14px; background: var(--surface, #f8f9fa); border-radius: 10px; border: 1px solid var(--border, #e2e8f0);">
          <span style="font-size: 0.8rem; color: var(--text-muted, #64748b);">Output Format:</span>
          <button id="format-toggle-btn" class="btn-secondary" style="padding: 5px 14px; font-size: 0.78rem; border-radius: 8px; display: flex; align-items: center; gap: 6px; transition: all 0.2s ease;">
            <span id="format-toggle-icon" style="font-size: 0.9rem;">${icons.file(14)}</span>
            <span id="format-toggle-label">Raw</span>
          </button>
          <span id="format-hint" style="font-size: 0.72rem; color: var(--text-muted, #94a3b8); margin-left: auto;">${formattableCount} field(s) can be formatted</span>
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
      
      // Build confidence display
      const confidence = confidenceScores[key];
      const isFlagged = flaggedFields.includes(key);
      
      // Confidence dot: 🟢 ≥ 0.8, 🟡 0.5–0.79, 🔴 < 0.5
      let confDot = "";
      if (confidence !== undefined) {
        const pct = Math.round(confidence * 100);
        const color = confidence >= 0.8 ? "#22c55e" : confidence >= 0.5 ? "#eab308" : "#ef4444";
        const label = confidence >= 0.8 ? "High" : confidence >= 0.5 ? "Medium" : "Low";
        confDot = `<span class="confidence-dot" style="cursor: help;" title="${label} Confidence"><span style="width: 9px; height: 9px; border-radius: 50%; background: ${color}; display: inline-block; box-shadow: 0 0 4px ${color}40;"></span></span>`;
      }

      // Check if this field has a different normalized value
      const normalizedVal = normalizedValues[key];
      const hasFormat = normalizedVal && normalizedVal !== value;
      const formatIndicator = hasFormat
        ? `<span class="format-indicator" style="display: none; font-size: 0.65rem; color: var(--accent, #6366f1); margin-left: 6px; opacity: 0.7;" title="Formatted by validator">${icons.sparkle(11)}</span>`
        : "";

      // ─── Table rendering ───
      let valueHtml;
      let isTableField = false;
      try {
        const parsed = JSON.parse(value);
        if (Array.isArray(parsed) || typeof parsed === "object") {
          isTableField = true;
          valueHtml = renderTableValue(value, Array.isArray(parsed) ? "table" : "row");
        } else {
          valueHtml = `<div class="field-value">${displayValue}</div>`;
        }
      } catch {
        valueHtml = `<div class="field-value">${displayValue}</div>`;
      }

      html += `
        <div class="result-card animate-fadeUp${isFlagged ? ' flagged' : ''}${isTableField ? ' table-card' : ''}" data-field="${key}" data-raw="${safeValue}" data-normalized="${hasFormat ? normalizedVal : safeValue}" style="animation-delay: ${idx * 0.03}s;">
          <div class="card-header">
            <span class="field-name">${key}${formatIndicator}</span>
            ${confDot}
          </div>
          ${valueHtml}
          <div class="card-footer">
            <button class="btn-icon btn-resend" data-field="${key}" title="Re-extract">${icons.refreshCw(14)}</button>
            <button class="btn-icon btn-flag" data-field="${key}" data-value="${safeValue}" data-signal="manual_flag" title="Flag for review">${icons.alertCircle(14)}</button>
            <button class="btn-icon btn-delete-result" data-field="${key}" title="Remove">${icons.x(14)}</button>
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

    // Helper: build export data based on current format state
    function getExportData() {
      const data = {};
      for (const [k, v] of Object.entries(currentExtractionData)) {
        if (k === "_meta") continue;
        if (showFormatted) {
          const decomposed = decomposeField(k, v);
          if (decomposed) {
            // Expand decomposed parts inline: "Patient Address" → "Patient Address - Street", etc.
            for (const part of decomposed) {
              data[`${k} - ${part.label}`] = part.value;
            }
            continue;
          }
          // Use normalized value if available
          const norm = (currentExtractionData._meta?.normalized_values || {})[k];
          if (norm && norm !== v) {
            data[k] = norm;
            continue;
          }
        }
        data[k] = v;
      }
      return data;
    }

    html += `
      <div class="json-actions animate-fadeUp" style="animation-delay: ${idx * 0.03}s;">
        <button id="copy-json" class="btn-secondary">${icons.copy(14)} Copy JSON</button>
        <div style="position: relative; display: inline-block;">
          <button id="export-dropdown-btn" class="btn-secondary" style="background: #27ae60; color: #fff;">${icons.download(14)} Export ▾</button>
          <div id="export-dropdown-menu" style="display: none; position: absolute; top: 100%; left: 0; z-index: 100; background: var(--surface, #fff); border: 1px solid var(--border, #e2e8f0); border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); min-width: 140px; margin-top: 4px; overflow: hidden;">
            <button id="download-json" style="display: block; width: 100%; text-align: left; padding: 8px 14px; border: none; background: none; cursor: pointer; font-size: 0.82rem; color: var(--text, #1e293b);" onmouseover="this.style.background='var(--surface-hover, #f1f5f9)'" onmouseout="this.style.background='none'">${icons.file(14)} JSON</button>
            <button id="export-csv" style="display: block; width: 100%; text-align: left; padding: 8px 14px; border: none; background: none; cursor: pointer; font-size: 0.82rem; color: var(--text, #1e293b);" onmouseover="this.style.background='var(--surface-hover, #f1f5f9)'" onmouseout="this.style.background='none'">${icons.table ? icons.table(14) : '▇'} CSV</button>
          </div>
        </div>
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

        // Swap values on all cards + apply deep-parsing
        container.querySelectorAll(".result-card").forEach((card) => {
          const field = card.dataset.field;
          const raw = card.dataset.raw;
          const normalized = card.dataset.normalized;
          const valueEl = card.querySelector(".field-value");
          const indicator = card.querySelector(".format-indicator");

          if (showFormatted) {
            // Try deep-parsing first
            const decomposed = decomposeField(field, raw);
            if (decomposed) {
              valueEl.innerHTML = renderDecomposed(decomposed);
              card.style.borderLeft = "3px solid var(--accent, #6366f1)";
              if (indicator) indicator.style.display = "inline";
            } else if (raw !== normalized) {
              valueEl.textContent = normalized || raw || "—";
              card.style.borderLeft = "3px solid var(--accent, #6366f1)";
              if (indicator) indicator.style.display = "inline";
            }
          } else {
            // Reset to raw
            valueEl.textContent = raw || "—";
            card.style.borderLeft = "";
            // Remove any decomposed parts
            const existing = card.querySelector(".decomposed-parts");
            if (existing) existing.remove();
            if (indicator) indicator.style.display = "none";
          }
        });

        // Update JSON preview with formatted data
        const jsonOut = document.getElementById("json-output");
        if (jsonOut) {
          jsonOut.textContent = JSON.stringify(getExportData(), null, 2);
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

    // Resend buttons (re-extract single field)
    container.querySelectorAll(".btn-resend").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const fieldName = btn.dataset.field;
        if (!selectedFile) return;

        const card = btn.closest(".result-card");
        const valueEl = card.querySelector(".field-value");
        const originalHtml = btn.innerHTML;

        try {
          btn.innerHTML = icons.loader(14);
          btn.disabled = true;
          btn.querySelector("svg").style.animation = "spin 1s linear infinite";
          valueEl.style.opacity = "0.4";

          const selectedModel =
            document.querySelector('input[name="model-select"]:checked')?.value || "qwen";
          const result = await reExtractField(selectedFile, fieldName, selectedModel);

          // Update card
          const newValue = result.value || "";
          valueEl.textContent = newValue || "\u2014";
          valueEl.style.opacity = "1";
          card.dataset.raw = newValue;
          card.dataset.normalized = newValue;

          // Update data
          currentExtractionData[fieldName] = newValue;

          // Update confidence dot for re-extracted field
          const confDot = card.querySelector(".confidence-dot");
          if (confDot) {
            confDot.innerHTML = `<span style="width: 9px; height: 9px; border-radius: 50%; background: #22c55e; display: inline-block; box-shadow: 0 0 4px #22c55e40;"></span>`;
            confDot.title = "High Confidence";
          }

          // Update JSON preview
          const jsonOut = document.getElementById("json-output");
          if (jsonOut) {
            const exportData = {};
            for (const [k, v] of Object.entries(currentExtractionData)) {
              if (k !== "_meta") exportData[k] = v;
            }
            jsonOut.textContent = JSON.stringify(exportData, null, 2);
          }

          // Flash success
          btn.innerHTML = icons.check(14);
          btn.style.color = "var(--accent)";
          setTimeout(() => {
            btn.innerHTML = originalHtml;
            btn.style.color = "";
            btn.disabled = false;
          }, 1500);
        } catch (err) {
          console.error("Re-extract failed:", err);
          valueEl.style.opacity = "1";
          btn.innerHTML = icons.alertCircle(14);
          btn.style.color = "#e74c3c";
          setTimeout(() => {
            btn.innerHTML = originalHtml;
            btn.style.color = "";
            btn.disabled = false;
          }, 2000);
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

    // Copy JSON (uses formatted data when format mode is on)
    document.getElementById("copy-json")?.addEventListener("click", () => {
      navigator.clipboard.writeText(
        JSON.stringify(getExportData(), null, 2),
      );
      const btn = document.getElementById("copy-json");
      btn.innerHTML = `${icons.check(14)} Copied!`;
      setTimeout(() => (btn.innerHTML = `${icons.copy(14)} Copy JSON`), 2000);
    });

    // Export dropdown toggle
    const exportDropdownBtn = document.getElementById("export-dropdown-btn");
    const exportDropdownMenu = document.getElementById("export-dropdown-menu");
    if (exportDropdownBtn && exportDropdownMenu) {
      exportDropdownBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        exportDropdownMenu.style.display = exportDropdownMenu.style.display === "none" ? "block" : "none";
      });
      document.addEventListener("click", () => { exportDropdownMenu.style.display = "none"; });
    }

    // Download JSON (uses formatted data when format mode is on)
    document.getElementById("download-json")?.addEventListener("click", () => {
      const blob = new Blob([JSON.stringify(getExportData(), null, 2)], {
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

    // Export CSV (uses formatted data when format mode is on)
    document.getElementById("export-csv")?.addEventListener("click", async () => {
      const btn = document.getElementById("export-csv");
      try {
        btn.textContent = "Exporting...";
        await exportCSV(getExportData(), selectedFile?.name || "extraction");
        btn.innerHTML = `${icons.check(14)} Exported!`;
        setTimeout(() => { btn.innerHTML = `${icons.download(14)} Export CSV`; }, 2000);
      } catch (err) {
        console.error("CSV export error:", err);
        btn.textContent = "Export failed";
        setTimeout(() => { btn.innerHTML = `${icons.download(14)} Export CSV`; }, 2000);
      }
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
