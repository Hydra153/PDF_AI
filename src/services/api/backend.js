/**
 * Backend API Client
 * Handles communication with the Python FastAPI backend (Qwen2-VL)
 */

const BACKEND_URL = "http://localhost:8000";
const API_TIMEOUT = 300000; // 5 minutes — Qwen2-VL batched extraction

/**
 * Simple async mutex to prevent concurrent backend requests.
 * The single-threaded backend (Qwen2-VL) can only handle one
 * request at a time. This queues requests so they run sequentially.
 */
let _lockPromise = Promise.resolve();

function withLock(fn) {
  const prev = _lockPromise;
  let resolve;
  _lockPromise = new Promise(r => { resolve = r; });
  return prev.then(() => fn().finally(resolve));
}

/**
 * Check if backend is running
 */
export async function checkBackendHealth() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/health`, {
      method: "GET",
      signal: AbortSignal.timeout(10000),
    });

    if (!response.ok) {
      throw new Error("Backend unhealthy");
    }

    const data = await response.json();
    return data;
  } catch (err) {
    console.error("Backend health check failed:", err);
    return null;
  }
}

/**
 * Extract fields from PDF using Qwen model
 * @param {File} file - PDF file
 * @param {Array} fields - Array of {key, question} objects
 * @param {string} model - Model to use (default: "qwen")
 * @param {number} votingRounds - Number of voting passes (default: 1, use 3 for accuracy boost)
 * @returns {Object} Extracted field values
 */
export function extractFields(file, fields, model = "qwen", votingRounds = 1, checkboxEnabled = false, rawMode = false) {
  return withLock(async () => {
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("fields", JSON.stringify(fields));
      formData.append("model", model);
      if (votingRounds > 1) {
        formData.append("voting_rounds", votingRounds.toString());
      }
      formData.append("checkbox_enabled", checkboxEnabled ? "true" : "false");
      formData.append("raw_mode", rawMode ? "true" : "false");

      const response = await fetch(`${BACKEND_URL}/api/extract`, {
        method: "POST",
        body: formData,
        signal: AbortSignal.timeout(API_TIMEOUT),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Extraction failed");
      }

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.message || "Extraction failed");
      }

      return result.data;
    } catch (err) {
      if (err.name === "AbortError") {
        throw new Error("Request timeout - document too large or backend slow");
      }
      throw err;
    }
  });
}

/**
 * Auto-detect fields in PDF using Qwen2-VL
 * @param {File} file - PDF file
 * @returns {Array} Array of detected field objects
 */
export function autoFindFields(file) {
  return withLock(async () => {
    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${BACKEND_URL}/api/auto-find-fields`, {
        method: "POST",
        body: formData,
        signal: AbortSignal.timeout(API_TIMEOUT),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Auto-detection failed");
      }

      const result = await response.json();

      if (!result.success) {
        throw new Error("Auto-detection failed");
      }

      return result.fields;
    } catch (err) {
      if (err.name === "AbortError") {
        throw new Error("Request timeout - document too large or backend slow");
      }
      throw err;
    }
  });
}

/**
 * Detect all physical checkboxes in a PDF
 * @param {File} file - PDF file
 * @returns {Object} {checkboxes: [{label, checked, signal}], count, time_seconds}
 */
export function detectCheckboxes(file) {
  return withLock(async () => {
    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${BACKEND_URL}/api/detect-checkboxes`, {
        method: "POST",
        body: formData,
        signal: AbortSignal.timeout(API_TIMEOUT),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Checkbox detection failed");
      }

      const result = await response.json();

      if (!result.success) {
        throw new Error("Checkbox detection failed");
      }

      return result;
    } catch (err) {
      if (err.name === "AbortError") {
        throw new Error("Request timeout - document too large or backend slow");
      }
      throw err;
    }
  });
}

/**
 * Scan all pages for data tables using VLM pre-scan
 * @param {File} file - PDF file
 * @returns {Object} {tables: [{page, columns, row_index_prefix, column_count}], count, total_pages, time_seconds}
 */
export function findTables(file) {
  return withLock(async () => {
    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${BACKEND_URL}/api/find-tables`, {
        method: "POST",
        body: formData,
        signal: AbortSignal.timeout(API_TIMEOUT),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Table scan failed");
      }

      const result = await response.json();
      if (!result.success) throw new Error("Table scan failed");
      return result;
    } catch (err) {
      if (err.name === "AbortError") {
        throw new Error("Request timeout - document too large or backend slow");
      }
      throw err;
    }
  });
}

/**
 * Ask a natural language question about a PDF document
 * @param {File} file - PDF file
 * @param {string} question - Question to ask
 * @param {string} model - Model to use (default: "qwen")
 * @returns {Object} {answer, time_seconds}
 */
export function askQuestion(file, question, model = "qwen") {
  return withLock(async () => {
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("question", question);
      formData.append("model", model);

      const response = await fetch(`${BACKEND_URL}/api/ask`, {
        method: "POST",
        body: formData,
        signal: AbortSignal.timeout(API_TIMEOUT),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Question failed");
      }

      const result = await response.json();
      if (!result.success) throw new Error("Question failed");
      return result;
    } catch (err) {
      if (err.name === "AbortError") {
        throw new Error("Request timeout - try a simpler question");
      }
      throw err;
    }
  });
}

/**
 * Re-extract a single field from a PDF
 * @param {File} file - PDF file
 * @param {string} fieldName - Field to re-extract
 * @param {string} model - Model to use (default: "qwen")
 * @returns {Object} {field, value, signal, time_seconds}
 */
export function reExtractField(file, fieldName, model = "qwen") {
  return withLock(async () => {
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("field_name", fieldName);
      formData.append("model", model);

      const response = await fetch(`${BACKEND_URL}/api/re-extract`, {
        method: "POST",
        body: formData,
        signal: AbortSignal.timeout(API_TIMEOUT),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Re-extraction failed");
      }

      const result = await response.json();
      if (!result.success) throw new Error("Re-extraction failed");
      return result;
    } catch (err) {
      if (err.name === "AbortError") {
        throw new Error("Request timeout");
      }
      throw err;
    }
  });
}
