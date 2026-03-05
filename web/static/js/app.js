
const API_BASE_URL = '';

document.addEventListener('DOMContentLoaded', function() {
    const queryInput = document.getElementById('query-input');
    const submitBtn = document.getElementById('submit-btn');

    queryInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            submitQuery();
        }
    });

    submitBtn.addEventListener('click', submitQuery);

    document.querySelectorAll('.collapsible-header').forEach(header => {
        header.addEventListener('click', function() {
            const targetId = this.getAttribute('data-target');
            const body = document.getElementById(targetId);
            if (!body) return;
            body.classList.toggle('collapsed');
            this.classList.toggle('expanded');
        });
    });
});

async function submitQuery() {
    const queryInput = document.getElementById('query-input');
    const topMInput = document.getElementById('top-m-input');
    const executeCheckbox = document.getElementById('execute-checkbox');
    const submitBtn = document.getElementById('submit-btn');

    const query = queryInput.value.trim();
    if (!query) {
        showError('请输入查询内容');
        return;
    }

    setLoading(true);
    resetResults();

    const payload = {
        query: query,
        top_m: parseInt(topMInput.value),
        execute: executeCheckbox.checked
    };

    try {
        updateProgress(1);

        if (!executeCheckbox.checked) {
            const response = await fetch(`${API_BASE_URL}/api/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
            updateProgress(2);
            displayQuerySpec(data.query_spec);
            displaySqlCandidates(data.sql_candidates);
        } else {
            await streamQuery(payload);
        }
    } catch (error) {
        showError(error.message);
    } finally {
        setLoading(false);
    }
}

async function streamQuery(payload) {
    const response = await fetch(`${API_BASE_URL}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });

    if (!response.ok) {
        let msg = `HTTP ${response.status}`;
        try { const d = await response.json(); msg = d.error || msg; } catch (_) {}
        throw new Error(msg);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let part4Started = false;
    const answerEl = document.getElementById('answer-text');
    const part4Card = document.getElementById('part4-card');

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = parseSSE(buffer);
        buffer = events.remainder;

        for (const evt of events.parsed) {
            if (evt.event === 'part123') {
                updateProgress(3);
                const data = JSON.parse(evt.data);
                displayQuerySpec(data.query_spec);
                displaySqlCandidates(data.sql_candidates);
                if (data.part3) displayPart3(data.part3);
            } else if (evt.event === 'token') {
                if (!part4Started) {
                    part4Started = true;
                    part4Card.hidden = false;
                    answerEl.textContent = '';
                    answerEl.classList.add('streaming');
                    updateProgress(4);
                }
                const data = JSON.parse(evt.data);
                answerEl.textContent += data.t;
            } else if (evt.event === 'done') {
                answerEl.classList.remove('streaming');
            } else if (evt.event === 'error') {
                const data = JSON.parse(evt.data);
                throw new Error(data.error || 'Stream error');
            }
        }
    }

    answerEl.classList.remove('streaming');
}

function parseSSE(text) {
    const parsed = [];
    const blocks = text.split('\n\n');
    const remainder = blocks.pop();

    for (const block of blocks) {
        if (!block.trim()) continue;
        let event = 'message';
        let data = '';
        for (const line of block.split('\n')) {
            if (line.startsWith('event: ')) {
                event = line.slice(7);
            } else if (line.startsWith('data: ')) {
                data = line.slice(6);
            }
        }
        if (data) parsed.push({ event, data });
    }

    return { parsed, remainder: remainder || '' };
}

function setLoading(loading) {
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const btnLoading = submitBtn.querySelector('.btn-loading');

    submitBtn.disabled = loading;
    btnText.hidden = loading;
    btnLoading.hidden = !loading;
}

function resetResults() {
    const resultsSection = document.getElementById('results-section');
    const errorCard = document.getElementById('error-card');
    const part3Card = document.getElementById('part3-card');
    const part4Card = document.getElementById('part4-card');

    resultsSection.hidden = false;
    errorCard.hidden = true;
    part3Card.hidden = true;
    part4Card.hidden = true;

    document.querySelectorAll('.progress-step').forEach(step => {
        step.classList.remove('active', 'completed');
    });

    document.querySelectorAll('.collapsible-body').forEach(body => {
        body.classList.add('collapsed');
    });
    document.querySelectorAll('.collapsible-header').forEach(header => {
        header.classList.remove('expanded');
    });

    document.getElementById('query-spec-content').textContent = '';
    document.getElementById('sql-candidates').innerHTML = '';
    document.getElementById('answer-text').textContent = '';
    document.getElementById('answer-text').classList.remove('streaming');
}

function updateProgress(step) {
    const steps = document.querySelectorAll('.progress-step');

    steps.forEach((s, index) => {
        s.classList.remove('active', 'completed');
        if (index < step - 1) {
            s.classList.add('completed');
        } else if (index === step - 1) {
            s.classList.add('active');
        }
    });
}

function displayQuerySpec(querySpec) {
    const content = document.getElementById('query-spec-content');
    content.textContent = JSON.stringify(querySpec, null, 2);
}

function displaySqlCandidates(sqlCandidates) {
    const container = document.getElementById('sql-candidates');
    container.innerHTML = '';

    if (!sqlCandidates || !sqlCandidates.candidates || sqlCandidates.candidates.length === 0) {
        container.innerHTML = '<p class="text-muted">没有 SQL 候选</p>';
        return;
    }

    sqlCandidates.candidates.forEach(candidate => {
        const div = document.createElement('div');
        div.className = 'sql-candidate';

        const notes = candidate.notes ? `<span class="sql-candidate-notes">${escapeHtml(candidate.notes)}</span>` : '';

        div.innerHTML = `
            <div class="sql-candidate-header">
                <span class="sql-candidate-id">候选 #${candidate.id}</span>
                ${notes}
            </div>
            <pre><code class="language-sql">${escapeHtml(candidate.sql)}</code></pre>
        `;

        container.appendChild(div);
    });

    if (typeof hljs !== 'undefined') {
        container.querySelectorAll('code').forEach(block => {
            hljs.highlightElement(block);
        });
    }
}

function displayPart3(part3) {
    const part3Card = document.getElementById('part3-card');
    const summary = document.getElementById('part3-summary');
    const selectedSql = document.getElementById('selected-sql');
    const tableBody = document.querySelector('#datasets-table tbody');

    part3Card.hidden = false;

    const passedCount = part3.passed ? part3.passed.length : 0;
    const failedCount = part3.failed ? part3.failed.length : 0;

    summary.innerHTML = `
        <div class="summary-item">
            <span class="summary-label">通过</span>
            <span class="summary-value success">${passedCount}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">失败</span>
            <span class="summary-value">${failedCount}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">选中</span>
            <span class="summary-value">#${part3.selected_id || 'N/A'}</span>
        </div>
    `;

    if (part3.selected_sql) {
        selectedSql.textContent = part3.selected_sql;
        if (typeof hljs !== 'undefined') {
            hljs.highlightElement(selectedSql);
        }
    } else {
        selectedSql.textContent = '-- 没有选中的 SQL';
    }

    tableBody.innerHTML = '';

    if (part3.datasets && part3.datasets.length > 0) {
        part3.datasets.forEach((ds, index) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${escapeHtml(String(ds.dataset_id || ''))}</td>
                <td>${escapeHtml(String(ds.dataset_name || ''))}</td>
                <td>${typeof ds.score === 'number' ? ds.score.toFixed(4) : escapeHtml(String(ds.score || ''))}</td>
            `;
            tableBody.appendChild(row);
        });
    } else {
        tableBody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">无结果</td></tr>';
    }
}

function showError(message) {
    const errorCard = document.getElementById('error-card');
    const errorMessage = document.getElementById('error-message');
    const resultsSection = document.getElementById('results-section');

    resultsSection.hidden = false;
    errorCard.hidden = false;
    errorMessage.textContent = message;

    document.querySelectorAll('.progress-step').forEach(step => {
        step.classList.remove('active');
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
