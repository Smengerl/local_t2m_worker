/**
 * scripts.js — Sidebar resize/collapse, enqueue form, config loading, app bootstrap.
 *
 * Loaded after queue-ui.js. Relies on apiFetch, escHtml, refresh,
 * updateHealthStatus, setupQueueSorting, and closeLightbox defined there.
 */

// ── Sidebar resize & collapse ──────────────────────────────────────────────
function initSidebar() {
    const STORAGE_KEY_W = 'sidebar_width';
    const STORAGE_KEY_C = 'sidebar_collapsed';
    const MIN_W = 120;
    const MAX_W = 700;

    const layout = document.querySelector('.layout');
    const leftCol = document.querySelector('.left-col');
    const resizer = document.getElementById('col-resizer');
    const toggle = document.getElementById('col-toggle');
    if (!resizer || !toggle) return;

    let collapsed = localStorage.getItem(STORAGE_KEY_C) === '1';
    let width = parseInt(localStorage.getItem(STORAGE_KEY_W) || '400', 10);

    function applyState() {
        if (collapsed) {
            layout.style.gridTemplateColumns = `0 4px 1fr`;
            leftCol.classList.add('collapsed');
            toggle.textContent = '▶';
            toggle.title = 'Expand sidebar';
        } else {
            layout.style.gridTemplateColumns = `${width}px 4px 1fr`;
            leftCol.classList.remove('collapsed');
            toggle.textContent = '◀';
            toggle.title = 'Collapse sidebar';
        }
    }

    toggle.addEventListener('click', (e) => {
        e.stopPropagation();
        collapsed = !collapsed;
        localStorage.setItem(STORAGE_KEY_C, collapsed ? '1' : '0');
        applyState();
    });

    let dragging = false;
    let startX = 0;
    let startW = 0;

    resizer.addEventListener('mousedown', (e) => {
        if (e.target === toggle) return;
        if (collapsed) return;
        dragging = true;
        startX = e.clientX;
        startW = width;
        resizer.classList.add('dragging');
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!dragging) return;
        const delta = e.clientX - startX;
        width = Math.min(MAX_W, Math.max(MIN_W, startW + delta));
        layout.style.gridTemplateColumns = `${width}px 4px 1fr`;
    });

    document.addEventListener('mouseup', () => {
        if (!dragging) return;
        dragging = false;
        resizer.classList.remove('dragging');
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        localStorage.setItem(STORAGE_KEY_W, width);
    });

    applyState();
}

// ── Config loading & apply ─────────────────────────────────────────────────

// Map from config value (path string) → full config object from /api/configs
const _configMap = new Map();

/**
 * Apply a config entry to the form: set field placeholders from defaults,
 * show/hide hint icons with tooltip text, show/hide notes boxes.
 */
function applyConfig(cfg) {
    if (!cfg) return;
    const d = cfg.defaults || {};
    const h = cfg.hints || {};
    const n = cfg.notes || {};

    // ── Field placeholders from config defaults ──────────────────────────
    const placeholders = {
        'f-model-repo': d.model_repo ?? '',
        'f-model-gguf-file': d.model_gguf_file ?? '',
        'f-lora-repo': d.lora_repo ?? '',
        'f-lora-strength': d.lora_strength != null ? String(d.lora_strength) : '',
        'f-steps': d.steps != null ? String(d.steps) : '',
        'f-cfg-scale': d.cfg_scale != null ? String(d.cfg_scale) : '',
        'f-width': d.width != null ? String(d.width) : '',
        'f-height': d.height != null ? String(d.height) : '',
    };
    for (const [id, text] of Object.entries(placeholders)) {
        const el = document.getElementById(id);
        if (el) el.placeholder = text;
    }

    // ── Section hint icons ───────────────────────────────────────────────
    const hints = {
        'f-model-section-icon': { icon: 'f-model-section-icon', tip: 'f-model-section-tooltip', text: h.model },
        'f-lora-section-icon': { icon: 'f-lora-section-icon', tip: 'f-lora-section-tooltip', text: h.lora },
        'f-gen-section-icon': { icon: 'f-gen-section-icon', tip: 'f-gen-section-tooltip', text: h.generation },
    };
    for (const { icon, tip, text } of Object.values(hints)) {
        const iconEl = document.getElementById(icon);
        const tipEl = document.getElementById(tip);
        if (!iconEl) continue;
        if (text) {
            iconEl.classList.remove('hidden');
            if (tipEl) tipEl.textContent = text;
        } else {
            iconEl.classList.add('hidden');
        }
    }

    // ── Notes boxes ──────────────────────────────────────────────────────
    const aboutEl = document.getElementById('f-notes-about');
    const warnEl = document.getElementById('f-notes-warning');
    const guideEl = document.getElementById('f-notes-guide');
    const guideBody = document.getElementById('f-notes-guide-body');

    if (aboutEl) {
        aboutEl.textContent = n.about || '';
        aboutEl.classList.toggle('hidden', !n.about);
    }
    if (warnEl) {
        warnEl.textContent = n.warnings || '';
        warnEl.classList.toggle('hidden', !n.warnings);
    }
    if (guideEl && guideBody) {
        guideBody.textContent = n.prompt_guide || '';
        guideEl.classList.toggle('hidden', !n.prompt_guide);
    }

    // ── Trigger word hint on prompt label ────────────────────────────────
    const promptIcon = document.getElementById('f-prompt-icon');
    const promptTip = document.getElementById('f-prompt-tooltip');
    const trigger = cfg.extras?.trigger_word;
    if (promptIcon) {
        if (trigger) {
            promptIcon.classList.remove('hidden');
            if (promptTip) promptTip.textContent = `Trigger word: ${trigger}`;
        } else {
            promptIcon.classList.add('hidden');
        }
    }
}

async function loadConfigs() {
    try {
        const configs = await apiFetch('/api/configs');
        const sel = document.getElementById('f-config');
        if (!sel) return;

        // Populate dropdown and build lookup map
        sel.innerHTML = configs
            .map(c => `<option value="${escHtml(c.value)}">${escHtml(c.label)}</option>`)
            .join('');
        _configMap.clear();
        for (const c of configs) _configMap.set(c.value, c);

        // Apply the initially selected config
        applyConfig(_configMap.get(sel.value));

        // Re-apply whenever the user picks a different config
        sel.addEventListener('change', () => applyConfig(_configMap.get(sel.value)));
    } catch (err) {
        console.warn('Could not load configs:', err.message);
    }
}

// ── Enqueue form ───────────────────────────────────────────────────────────
function setupForm() {
    const form = document.getElementById('enqueue-form');
    if (!form) return;

    form.addEventListener('submit', async e => {
        e.preventDefault();
        const msg = document.getElementById('form-msg');
        const v = id => document.getElementById(id)?.value;

        const body = {
            config: v('f-config'),
            prompt: v('f-prompt').trim(),
            negative_prompt: v('f-negative').trim() || '',
            model_repo: v('f-model-repo').trim() || null,
            model_gguf_file: v('f-model-gguf-file').trim() || null,
            lora_repo: v('f-lora-repo').trim() || null,
            lora_strength: v('f-lora-strength') ? +v('f-lora-strength') : null,
            steps: v('f-steps') ? +v('f-steps') : null,
            cfg_scale: v('f-cfg-scale') ? +v('f-cfg-scale') : null,
            width: v('f-width') ? +v('f-width') : null,
            height: v('f-height') ? +v('f-height') : null,
        };

        try {
            const job = await apiFetch('/api/jobs', { method: 'POST', body: JSON.stringify(body) });
            msg.className = 'form-feedback form-feedback--success';
            msg.textContent = `✅ Queued id=${job.id.slice(0, 8)}`;
            document.getElementById('f-prompt').value = '';
            refresh();
        } catch (err) {
            msg.className = 'form-feedback form-feedback--error';
            msg.textContent = `❌ ${err.message}`;
        }
    });
}

// ── App bootstrap ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    initSidebar();
    setupForm();
    await loadConfigs();

    refresh();
    updateHealthStatus();

    setInterval(refresh, 3000);
    setInterval(updateHealthStatus, 5000);

    document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });
});
