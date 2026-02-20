/**
 * Professional SVG Icon System
 * 
 * Inline SVG icons for a clean, professional UI.
 * All icons are 16x16 by default with currentColor for easy theming.
 * No external dependencies.
 */

const ICON_SIZE = 16;
const STROKE_WIDTH = 1.8;

function svg(content, size = ICON_SIZE, viewBox = '0 0 24 24') {
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="${viewBox}" fill="none" stroke="currentColor" stroke-width="${STROKE_WIDTH}" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; flex-shrink: 0;">${content}</svg>`;
}

export const icons = {
    // Navigation & UI
    clipboard:    (s) => svg('<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><rect x="8" y="2" width="8" height="4" rx="1" ry="1"/>', s),
    search:       (s) => svg('<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>', s),
    eye:          (s) => svg('<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>', s),
    file:         (s) => svg('<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>', s),
    download:     (s) => svg('<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>', s),
    copy:         (s) => svg('<rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>', s),
    x:            (s) => svg('<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>', s),

    // Actions
    check:        (s) => svg('<polyline points="20 6 9 17 4 12"/>', s),
    edit:         (s) => svg('<path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>', s),
    trash:        (s) => svg('<polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>', s),
    upload:       (s) => svg('<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>', s),
    externalLink: (s) => svg('<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/>', s),

    // Status
    checkCircle:  (s) => svg('<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>', s),
    alertCircle:  (s) => svg('<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>', s),
    loader:       (s) => svg('<line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"/><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/><line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"/><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"/>', s),
    circle:       (s) => svg('<circle cx="12" cy="12" r="10"/>', s),
    sparkle:      (s) => svg('<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>', s),

    // Data & Training
    barChart:     (s) => svg('<line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/>', s),
    brain:        (s) => svg('<path d="M9.5 2A5.5 5.5 0 0 0 5 7.5c0 1.03.28 1.99.77 2.82A5.5 5.5 0 0 0 4 15.5 5.5 5.5 0 0 0 9.5 21h5a5.5 5.5 0 0 0 5.5-5.5 5.5 5.5 0 0 0-1.77-4.18A5.5 5.5 0 0 0 19 7.5 5.5 5.5 0 0 0 14.5 2h-5z"/><path d="M12 2v19"/>', s),
    zap:          (s) => svg('<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>', s),
    database:     (s) => svg('<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>', s),
    arrowRight:   (s) => svg('<line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>', s),

    // Chat & Resend
    send:         (s) => svg('<line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>', s),
    refreshCw:    (s) => svg('<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>', s),
    messageSquare:(s) => svg('<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>', s),
    chevronRight: (s) => svg('<polyline points="9 18 15 12 9 6"/>', s),
};
