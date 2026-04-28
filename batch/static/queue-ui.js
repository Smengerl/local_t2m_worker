/**
 * queue-ui.js — Queue rendering, job actions, drag-and-drop, health, lightbox.
 *
 * Loaded before scripts.js. Exports all symbols used by inline onclick
 * handlers to window so they remain accessible from HTML templates.
 *
 * pipeline_config is serialised by PipelineConfig.to_dict() — nested v2 keys:
 *   pc.backend, pc.model.repo, pc.model.gguf_file,
 *   pc.lora?.repo, pc.lora?.strength,
 *   pc.generation.steps, pc.generation.cfg_scale, pc.generation.width/height
 *
 * Complexity notes:
 *  - Stats are derived from the /api/jobs response — no separate /api/stats call.
 *  - Sortable is initialised once (setupQueueSorting) and never recreated.
 *  - isDragging is module-scoped, not on window.
 */

// ── Shared state ───────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

let expandedIds = new Set();
let isDragging = false;

// ── API helper (shared with scripts.js) ───────────────────────────────────
async function apiFetch(path, opts = {}) {
    const r = await fetch(path, {
        headers: { 'Content-Type': 'application/json' },
        ...opts
    });
    if (!r.ok) {
        const err = await r.json().catch(() => ({ detail: r.statusText }));
        throw new Error(err.detail || r.statusText);
    }
    return r.json();
}

// ── Rendering helpers ──────────────────────────────────────────────────────
function badge(status) {
    return `<span class="badge ${status}">${status}</span>`;
}

function relTime(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    const diff = Math.round((Date.now() - d) / 1000);
    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
}

function durStr(startIso, endIso) {
    if (!startIso || !endIso) return null;
    const secs = Math.round((new Date(endIso) - new Date(startIso)) / 1000);
    if (secs < 60) return `${secs}s`;
    return `${Math.floor(secs / 60)}m ${secs % 60}s`;
}

function escHtml(s) {
    return String(s ?? '')
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

// ── Stats — derived from the jobs list, no separate API call ──────────────
function updateStats(jobs) {
    const counts = { pending: 0, running: 0, done: 0, failed: 0 };
    for (const j of jobs) if (j.status in counts) counts[j.status]++;
    if ($('s-pending')) $('s-pending').textContent = counts.pending;
    if ($('s-running')) $('s-running').textContent = counts.running;
    if ($('s-done')) $('s-done').textContent = counts.done;
    if ($('s-failed')) $('s-failed').textContent = counts.failed;
}

// ── Queue Actions ──────────────────────────────────────────────────────────
async function deleteJob(id, evt) {
    if (evt) evt.stopPropagation();
    await apiFetch(`/api/jobs/${id}`, { method: 'DELETE' });
    expandedIds.delete(id);
    refresh();
}

async function retryJob(id, evt) {
    if (evt) evt.stopPropagation();
    await apiFetch(`/api/jobs/${id}/retry`, { method: 'POST' });
    refresh();
}

async function cancelJob(id, evt) {
    if (evt) evt.stopPropagation();
    if (!confirm('Really cancel this job?')) return;
    await apiFetch(`/api/jobs/${id}/cancel`, { method: 'POST' });
    refresh();
}

async function deleteImage(filename, jobId, evt) {
    if (evt) evt.stopPropagation();
    if (!confirm('Permanently delete this image?')) return;
    await apiFetch(`/outputs/${filename}`, { method: 'DELETE' });
    try { await apiFetch(`/api/jobs/${jobId}`, { method: 'DELETE' }); } catch (e) { /* ignore */ }
    refresh();
}

async function clearFinished() {
    await apiFetch('/api/clear-finished', { method: 'POST' });
    expandedIds.clear();
    refresh();
}

// ── Health Status ──────────────────────────────────────────────────────────
async function updateHealthStatus() {
    const statusEl = $('queue-status-alert');
    const statsEl = $('stats');
    if (!statusEl) return;

    try {
        const res = await fetch('/api/health');
        if (!res.ok) throw new Error('Health probe failed');
        const data = await res.json();

        if (statsEl) statsEl.classList.remove('hidden');

        const isDegraded = !data.worker_alive;
        const hasError = data.worker_error && data.worker_error !== 'None' && data.worker_error !== '';

        if (isDegraded || hasError) {
            statusEl.innerHTML = `<span class="status-alert__badge">⚠️ ${hasError ? data.worker_error : 'Worker is offline'}</span>`;
            statusEl.classList.remove('hidden');
        } else {
            statusEl.classList.add('hidden');
        }
    } catch {
        if (statsEl) statsEl.classList.add('hidden');
        statusEl.innerHTML = `<span class="status-alert__badge">⚠️ Server unreachable</span>`;
        statusEl.classList.remove('hidden');
    }
}

// ── Job Rendering ──────────────────────────────────────────────────────────
function renderJobCard(j) {
    const pc = j.pipeline_config || {};
    const cfgLabel = pc.backend ?? '—';
    const expanded = expandedIds.has(j.id);
    const isPending = j.status === 'pending';

    const dragHandle = isPending
        ? `<div class="drag-handle" title="Drag to reorder"><span class="material-icons-round">drag_indicator</span></div>`
        : '';

    // Build action buttons for this job's status
    const actions = [];
    if (isPending)
        actions.push(`<button class="btn-sm btn-danger" onclick="deleteJob('${j.id}',event)"><span class="material-icons-round">delete</span>Delete</button>`);
    if (j.status === 'failed' || j.status === 'done')
        actions.push(`<button class="btn-sm btn-retry" onclick="retryJob('${j.id}',event)"><span class="material-icons-round">replay</span>Re-run</button>`);
    if (j.status === 'failed' || (j.status === 'done' && !j.result_path))
        actions.push(`<button class="btn-sm btn-danger" onclick="deleteJob('${j.id}',event)"><span class="material-icons-round">delete</span>Delete</button>`);
    if (j.status === 'done' && j.result_path) {
        const fname = j.result_path.split('/').pop();
        actions.push(`<button class="btn-sm btn-danger" onclick="deleteImage('${fname}','${j.id}',event)"><span class="material-icons-round">hide_image</span>Delete image</button>`);
    }
    if (j.status === 'running')
        actions.push(`<button class="btn-sm btn-cancel" onclick="cancelJob('${j.id}',event)"><span class="material-icons-round">cancel</span>Cancel</button>`);

    // Progress block (running jobs only)
    const step = j.progress_step || 0;
    const total = j.progress_total || 0;
    const pct = total > 0 ? Math.round((step / total) * 100) : 0;
    const progressBlock = j.status === 'running' ? `
        <div class="job-progress">
            <div class="job-progress-status">
                <span class="spinner"></span>
                <span class="job-progress-label">${step >= total && total > 0 ? 'Completing…' : 'Generating…'}</span>
            </div>
            <progress value="${pct}" max="100"></progress>
            <div class="job-progress-steps">${total > 0 ? `${step} / ${total} Steps (${pct}%)` : '…'}</div>
        </div>` : '';

    // Thumbnail (done jobs with a saved result only)
    const thumbSrc = j.status === 'done' && j.result_path
        ? `/outputs/${j.result_path.split('/').pop()}` : null;
    const inlineThumb = thumbSrc
        ? `<img class="job-thumb" src="${thumbSrc}" onclick="event.stopPropagation(); openLightbox('${thumbSrc}')" onerror="this.style.display='none'">`
        : '';

    // Expanded detail panel
    let detail = '';
    if (expanded) {
        const modelRepo = pc.model?.repo || '—';
        const modelGguf = pc.model?.gguf_file ? ` (${pc.model.gguf_file})` : '';
        const lora = pc.lora?.repo ? `${pc.lora.repo} (Strength: ${pc.lora?.strength ?? '?'})` : '—';
        const gen = pc.generation || {};
        const steps = gen.steps ?? '—';
        const cfg = gen.cfg_scale ?? '—';
        const w = gen.width ?? '—';
        const h = gen.height ?? '—';

        detail = `
            <div class="detail">
                <div class="detail-row">
                    <span class="detail-label">Prompt:</span>
                    <span class="detail-value text-truncate">${escHtml(j.prompt)}</span>
                    <button class="btn-icon-only" title="Copy Prompt" onclick="copyPromptFromEl(this,event)" data-prompt="${encodeURIComponent(j.prompt)}">
                        <span class="material-icons-round">content_copy</span>
                    </button>
                </div>
                ${j.negative_prompt ? `
                <div class="detail-row">
                    <span class="detail-label">Negative:</span>
                    <span class="detail-value text-truncate">${escHtml(j.negative_prompt)}</span>
                    <button class="btn-icon-only" title="Copy Negative Prompt" onclick="copyPromptFromEl(this,event)" data-prompt="${encodeURIComponent(j.negative_prompt)}">
                        <span class="material-icons-round">content_copy</span>
                    </button>
                </div>` : ''}
                <div class="detail-row">
                    <span class="detail-label">Model:</span>
                    <span class="detail-value">${escHtml(modelRepo)}${escHtml(modelGguf)}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">LoRA:</span>
                    <span class="detail-value">${escHtml(lora)}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Generation:</span>
                    <span class="detail-value">${steps} steps · Guidance ${cfg} · ${w}×${h}px</span>
                </div>
                ${j.error ? `<div class="detail-error"><strong>Error:</strong><pre>${escHtml(j.error)}</pre></div>` : ''}
                ${j.log_lines?.length ? `
                <hr class="detail-divider">
                <div class="detail-log"><pre>${j.log_lines.map(escHtml).join('\n')}</pre></div>` : ''}
            </div>`;
    }

    return `<div class="job job-card${expanded ? ' expanded' : ''}" data-job-id="${j.id}" id="job-${j.id}">
        <div class="job-grid">
            <div class="job-col-left">
                <button class="btn-icon-only toggle-details" title="${expanded ? 'Collapse details' : 'Expand details'}" onclick="toggleExpand('${j.id}',event)">
                    <span class="material-icons-round">${expanded ? 'expand_less' : 'expand_more'}</span>
                </button>
                ${dragHandle}
            </div>
            <div class="job-col-middle">
                <div class="job-row-prompt">
                    <span class="job-prompt text-truncate">${escHtml(j.prompt)}</span>
                </div>
                <div class="job-row-meta">
                    added ${relTime(j.added_at)}
                    ${j.finished_at ? ` · finished ${relTime(j.finished_at)} · took ${durStr(j.added_at, j.finished_at)}` : ''}
                    · ${cfgLabel}
                </div>
                <div class="job-row-actions">${actions.join('')}</div>
            </div>
            <div class="job-col-right">
                ${j.status !== 'done' ? `<div class="job-status-badge">${badge(j.status)}</div>` : ''}
                ${progressBlock}${inlineThumb}
            </div>
        </div>
        ${detail}
    </div>`;
}

async function refreshJobs() {
    let jobs;
    try {
        jobs = await apiFetch('/api/jobs');
    } catch (e) {
        const list = $('queue-list');
        if (list) list.innerHTML = '<div class="empty">Could not load the queue.</div>';
        return;
    }
    const list = $('queue-list');
    if (!list) return;

    updateStats(jobs);

    if (jobs.length === 0) {
        list.innerHTML = '<div class="empty">Queue is empty.</div>';
        return;
    }

    const pending = jobs.filter(j => j.status === 'pending');
    const running = jobs.filter(j => j.status === 'running');
    const finished = jobs.filter(j => j.status === 'done' || j.status === 'failed').reverse();

    const parts = [];

    // ── Pending (draggable) ────────────────────────────────────────────────
    if (pending.length > 0) {
        parts.push(`<div id="queue-pending-list">${pending.map(renderJobCard).join('')}</div>`);
    } else {
        parts.push(`<div id="queue-pending-list"></div>`);
    }

    // ── Running ────────────────────────────────────────────────────────────
    if (running.length > 0) {
        parts.push(`<div class="queue-section-divider"></div>`);
        parts.push(`<div class="queue-section">${running.map(renderJobCard).join('')}</div>`);
    }

    // ── Done / Failed ──────────────────────────────────────────────────────
    if (finished.length > 0) {
        parts.push(`<div class="queue-section-divider"><span>Finished</span></div>`);
        parts.push(`<div class="queue-clear-row">
            <button class="btn-clear" onclick="clearFinished()">
                <span class="material-icons-round">delete_sweep</span>Clear finished
            </button>
        </div>`);
        parts.push(`<div class="queue-section">${finished.map(renderJobCard).join('')}</div>`);
    }

    list.innerHTML = parts.join('');

    // Re-attach Sortable to the (re-rendered) pending list
    const pendingList = $('queue-pending-list');
    if (pendingList && typeof Sortable !== 'undefined') {
        if (window.queueSortable) { window.queueSortable.destroy(); window.queueSortable = null; }
        window.queueSortable = Sortable.create(pendingList, {
            animation: 150,
            handle: '.drag-handle',
            draggable: '.job-card',
            ghostClass: 'sortable-ghost',
            onStart: () => { isDragging = true; },
            onEnd: async () => {
                const jobIds = Array.from(pendingList.querySelectorAll('.job-card'))
                    .map(el => el.getAttribute('data-job-id'))
                    .filter(Boolean);
                try {
                    const res = await fetch('/api/jobs/reorder', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ job_ids: jobIds })
                    });
                    if (!res.ok) throw new Error(`Server error: ${res.status}`);
                } catch (err) {
                    console.error('Error saving sort order:', err);
                } finally {
                    isDragging = false;
                    refresh();
                }
            }
        });
    }
}

function toggleExpand(id, event) {
    if (event) event.stopPropagation();
    expandedIds.has(id) ? expandedIds.delete(id) : expandedIds.add(id);
    refreshJobs();
}

// ── Lightbox ───────────────────────────────────────────────────────────────
function openLightbox(src) {
    const lb = $('lightbox');
    if (!lb) return;
    $('lightbox-img').src = src;
    lb.classList.add('open');
}

function closeLightbox() {
    $('lightbox')?.classList.remove('open');
}

// ── Clipboard toast ────────────────────────────────────────────────────────
async function copyPromptFromEl(btn, evt) {
    evt.stopPropagation();
    const text = decodeURIComponent(btn.getAttribute('data-prompt') || '');
    try {
        await navigator.clipboard.writeText(text);
        let t = $('copied-toast');
        if (!t) {
            t = document.createElement('div');
            t.id = 'copied-toast';
            t.className = 'copied-toast';
            document.body.appendChild(t);
        }
        t.textContent = 'Copied to clipboard';
        t.classList.add('visible');
        setTimeout(() => t.classList.remove('visible'), 1800);
    } catch (e) { console.error('Copy failed', e); }
}

// ── Main refresh — called by scripts.js bootstrap and polling ─────────────
async function refresh() {
    if (isDragging) return;
    await refreshJobs();
}

// ── window exports for inline onclick handlers in the HTML template ────────
window.deleteJob = deleteJob;
window.retryJob = retryJob;
window.cancelJob = cancelJob;
window.deleteImage = deleteImage;
window.clearFinished = clearFinished;
window.toggleExpand = toggleExpand;
window.openLightbox = openLightbox;
window.closeLightbox = closeLightbox;
window.copyPromptFromEl = copyPromptFromEl;
