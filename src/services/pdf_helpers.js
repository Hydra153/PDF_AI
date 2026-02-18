/**
 * PDF Helper Utilities
 * Uses pdfjs-dist to extract layout data and render page images
 */

import * as pdfjsLib from "pdfjs-dist";

// Configure PDF.js worker
pdfjsLib.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";

/**
 * Extract layout data (text items with bounding boxes) from a PDF file.
 * Used for the Layout Preview canvas.
 *
 * @param {File} file - PDF file object
 * @returns {Promise<Array<{width: number, height: number, items: Array<{text: string, bbox: number[]}>}>>}
 */
export async function extractLayoutFromPDF(file) {
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

  const pages = [];

  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const viewport = page.getViewport({ scale: 1.0 });
    const textContent = await page.getTextContent();

    const items = textContent.items
      .filter((item) => item.str && item.str.trim().length > 0)
      .map((item) => {
        const tx = item.transform;
        // tx[4] = x position, tx[5] = y position from bottom-left
        // item.width and item.height give the text dimensions
        const x = tx[4];
        const y = viewport.height - tx[5]; // flip Y (PDF is bottom-up)
        const w = item.width;
        const h = item.height;

        // Normalize to 0-1000 coordinate space
        const left = Math.round((x / viewport.width) * 1000);
        const top = Math.round((y / viewport.height) * 1000);
        const right = Math.round(((x + w) / viewport.width) * 1000);
        const bottom = Math.round(((y + h) / viewport.height) * 1000);

        return {
          text: item.str,
          bbox: [
            Math.max(0, left),
            Math.max(0, top - Math.round((h / viewport.height) * 1000)),
            Math.min(1000, right),
            Math.min(1000, top),
          ],
        };
      });

    pages.push({
      width: viewport.width,
      height: viewport.height,
      items,
    });
  }

  return pages;
}

/**
 * Render a PDF page as a data URL image.
 * Used as fallback for scanned documents with no text layer.
 *
 * @param {File} file - PDF file object
 * @param {number} pageNum - 1-based page number
 * @param {number} scale - Render scale (default 1.5)
 * @returns {Promise<string>} Data URL of the rendered page
 */
export async function pdfPageToImage(file, pageNum = 1, scale = 1.5) {
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  const page = await pdf.getPage(pageNum);
  const viewport = page.getViewport({ scale });

  const canvas = document.createElement("canvas");
  canvas.width = viewport.width;
  canvas.height = viewport.height;

  const ctx = canvas.getContext("2d");
  await page.render({ canvasContext: ctx, viewport }).promise;

  return canvas.toDataURL("image/png");
}
