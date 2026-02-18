/**
 * ReviewQueue Component
 * Human-in-the-Loop review workflow with training data integration
 */

import { icons } from './icons.js';

const API_BASE = 'http://localhost:8000';

export class ReviewQueue {
    constructor(container) {
        this.container = container;
        this.items = [];
        this.resolvedItems = [];
        this.stats = { pending: 0, resolved: 0 };
        this.trainingStats = {};
        this.showResolved = false;
        this.showTrainingData = false;
        this.trainingData = [];
    }

    async init() {
        await this.fetchItems();
        await this.fetchTrainingStats();
        this.render();
    }

    async clearQueue() {
        try {
            await fetch(`${API_BASE}/api/reviews/clear`, { method: 'DELETE' });
            this.items = [];
            this.resolvedItems = [];
            this.stats = { pending: 0, resolved: 0 };
            this.trainingData = [];
            this.render();
        } catch (err) {
            console.error('Failed to clear queue:', err);
        }
    }

    async fetchItems() {
        try {
            const res = await fetch(`${API_BASE}/api/reviews`);
            if (res.ok) {
                const data = await res.json();
                this.items = data.items || [];
                this.resolvedItems = this.items.filter(i => i.status === 'resolved');
                this.stats = data.stats || { pending: 0, resolved: 0 };
            }
        } catch (err) {
            console.error('Failed to fetch reviews:', err);
        }
    }

    async fetchTrainingStats() {
        try {
            const res = await fetch(`${API_BASE}/api/training/stats`);
            if (res.ok) {
                this.trainingStats = await res.json();
            }
        } catch (err) {
            console.error('Failed to fetch training stats:', err);
        }
    }

    async fetchTrainingData() {
        try {
            const res = await fetch(`${API_BASE}/api/training/samples`);
            if (res.ok) {
                const data = await res.json();
                this.trainingData = Array.isArray(data) ? data : [];
            } else {
                this.trainingData = [];
            }
        } catch (err) {
            console.error('Failed to fetch training data:', err);
            this.trainingData = [];
        }
    }

    async resolveItem(id, action, correctedValue = null) {
        const card = document.getElementById(`card-${id}`);
        if (card) {
            card.style.opacity = '0.5';
            card.style.transform = 'scale(0.98)';
        }
        try {
            const res = await fetch(`${API_BASE}/api/reviews/${id}/resolve`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action, corrected_value: correctedValue })
            });
            if (res.ok) {
                if (card) {
                    card.style.animation = 'slideOut 0.3s ease-out forwards';
                    await new Promise(r => setTimeout(r, 300));
                }
                await this.init();
            }
        } catch (err) {
            console.error('Failed to resolve item:', err);
            if (card) {
                card.style.opacity = '1';
                card.style.transform = 'scale(1)';
            }
        }
    }

    async deleteItem(id) {
        await this.resolveItem(id, 'delete');
    }

    async exportTrainingData() {
        const btn = document.getElementById('export-btn');
        if (btn) {
            btn.innerHTML = `${icons.loader(14)} Exporting...`;
            btn.disabled = true;
        }
        try {
            const res = await fetch(`${API_BASE}/api/training/export`, { method: 'POST' });
            if (res.ok) {
                const data = await res.json();
                if (data.success) {
                    if (btn) {
                        btn.innerHTML = `${icons.checkCircle(14)} Exported`;
                        setTimeout(() => {
                            btn.innerHTML = `${icons.upload(14)} Export Data`;
                            btn.disabled = false;
                        }, 3000);
                    }
                } else {
                    if (btn) {
                        btn.innerHTML = `${icons.alertCircle(14)} No data`;
                        setTimeout(() => {
                            btn.innerHTML = `${icons.upload(14)} Export Data`;
                            btn.disabled = false;
                        }, 3000);
                    }
                }
            }
        } catch (err) {
            console.error('Failed to export training data:', err);
            if (btn) {
                btn.innerHTML = `${icons.upload(14)} Export Data`;
                btn.disabled = false;
            }
        }
    }

    _readinessBadge(stats) {
        const count = stats.total_samples || 0;
        if (count >= 50) return '<span class="readiness-badge ready">Ready to train</span>';
        if (count >= 20) return '<span class="readiness-badge moderate">Can begin training</span>';
        return '<span class="readiness-badge collecting">Collecting data</span>';
    }

    render() {
        const pendingItems = this.items.filter(i => i.status === 'pending');
        const ts = this.trainingStats;
        const sampleCount = ts.total_samples || 0;
        const correctionCount = ts.total_corrections || 0;
        const docCount = ts.unique_documents || 0;

        let html = `
            <div class="review-header animate-fadeUp">
                <div>
                    <h2>${icons.clipboard(18)} Review Queue</h2>
                    <p class="muted">Review and correct AI predictions</p>
                </div>
                <div class="header-stats">
                    <span class="stat-badge">${this.stats.pending} pending</span>
                    <span class="stat-badge">${this.stats.resolved} resolved</span>
                </div>
            </div>

            <!-- Training Status -->
            <div class="training-panel animate-fadeUp">
                <div class="training-row">
                    <div class="training-info">
                        <strong>${icons.brain(16)} Model Training</strong>
                        <span class="training-meta muted">
                            ${sampleCount} samples &middot; ${correctionCount} corrections &middot; ${docCount} documents
                        </span>
                        ${this._readinessBadge(ts)}
                    </div>
                    <div class="training-actions">
                        <button id="view-training-data" class="btn-secondary">${icons.barChart(14)} View Data</button>
                        <button id="export-btn" class="btn-primary">${icons.upload(14)} Export Data</button>
                    </div>
                </div>
                ${ts.recommendation ? `<div class="training-recommendation muted">${ts.recommendation}</div>` : ''}
            </div>
        `;

        // Training Data Panel
        if (this.showTrainingData) {
            html += `
                <div class="training-data-panel animate-fadeUp">
                    <div class="panel-header">
                        <strong>${icons.barChart(16)} Training Samples</strong>
                        <button id="close-training-data" class="btn-close">${icons.x(16)}</button>
                    </div>
                    <div class="training-samples">
                        ${(!Array.isArray(this.trainingData) || this.trainingData.length === 0) 
                            ? '<p class="muted">No training samples collected yet. Approve or correct extractions to begin.</p>' 
                            : this.trainingData.map(s => {
                                const corrections = s.corrections || {};
                                const correctionKeys = Object.keys(corrections);
                                const fieldCount = (s.fields_requested || []).length;
                                return `
                                    <div class="sample-item ${correctionKeys.length > 0 ? 'has-corrections' : ''}">
                                        <div class="sample-header">
                                            <span class="sample-file">${icons.file(12)} ${s.source_pdf || 'Unknown'}</span>
                                            <span class="sample-meta">${fieldCount} fields &middot; p${s.page_num || 1}</span>
                                        </div>
                                        ${correctionKeys.length > 0 
                                            ? correctionKeys.map(field => `
                                                <div class="sample-correction">
                                                    <span class="sample-field">${field}</span>
                                                    <span class="sample-arrow">${icons.arrowRight(12)}</span>
                                                    <span class="sample-value corrected">${corrections[field].corrected}</span>
                                                    <span class="sample-original">${corrections[field].original}</span>
                                                </div>
                                            `).join('')
                                            : '<span class="sample-approved">All fields approved</span>'
                                        }
                                    </div>
                                `;
                            }).join('')}
                    </div>
                </div>
            `;
        }

        // Pending items
        if (pendingItems.length === 0) {
            html += `
                <div class="empty-state animate-fadeUp">
                    <div class="empty-icon">${icons.checkCircle(48)}</div>
                    <h3>All caught up</h3>
                    <p class="muted">No items pending review</p>
                </div>
            `;
        } else {
            html += `<div class="review-list">`;
            pendingItems.forEach((item, idx) => {
                const confPercent = Math.round((item.confidence || 0.3) * 100);
                const confColor = confPercent >= 70 ? 'var(--accent)' : (confPercent >= 40 ? '#f59e0b' : '#ef4444');

                html += `
                    <div class="review-card animate-fadeUp" id="card-${item.id}" style="animation-delay: ${idx * 0.05}s;">
                        <div class="card-top">
                            <div class="card-info">
                                <span class="card-field">${item.field_name}</span>
                                <span class="card-file muted">${icons.file(12)} ${item.filename}</span>
                            </div>
                            <div class="card-conf" style="color: ${confColor};">${confPercent}%</div>
                            <button class="btn-delete-card" data-id="${item.id}" title="Remove">${icons.trash(14)}</button>
                        </div>
                        <div class="card-prediction">
                            <span class="muted">AI:</span> ${item.ai_value || '<em>Empty</em>'}
                        </div>
                        <div class="card-actions">
                            <button class="btn-approve" data-id="${item.id}">${icons.check(14)} Approve</button>
                            <input type="text" id="input-${item.id}" placeholder="Correct value..." />
                            <button class="btn-correct" data-id="${item.id}">${icons.edit(14)}</button>
                        </div>
                    </div>
                `;
            });
            html += `</div>`;
        }

        this.container.innerHTML = html;
        this.attachEventListeners();
    }

    attachEventListeners() {
        // View training data
        const viewBtn = document.getElementById('view-training-data');
        if (viewBtn) {
            viewBtn.addEventListener('click', async () => {
                await this.fetchTrainingData();
                this.showTrainingData = !this.showTrainingData;
                this.render();
            });
        }

        // Close training data
        const closeBtn = document.getElementById('close-training-data');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                this.showTrainingData = false;
                this.render();
            });
        }

        // Approve buttons
        this.container.querySelectorAll('.btn-approve').forEach(btn => {
            btn.addEventListener('click', () => this.resolveItem(btn.dataset.id, 'approve'));
        });

        // Correct buttons
        this.container.querySelectorAll('.btn-correct').forEach(btn => {
            btn.addEventListener('click', () => {
                const input = document.getElementById(`input-${btn.dataset.id}`);
                const value = input?.value?.trim();
                if (value) {
                    this.resolveItem(btn.dataset.id, 'correct', value);
                } else {
                    input?.focus();
                }
            });
        });

        // Delete buttons
        this.container.querySelectorAll('.btn-delete-card').forEach(btn => {
            btn.addEventListener('click', () => this.deleteItem(btn.dataset.id));
        });

        // Export button
        const exportBtn = document.getElementById('export-btn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportTrainingData());
        }
    }
}
