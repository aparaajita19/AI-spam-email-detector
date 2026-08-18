/**
 * AI Spam Email Detector - Frontend Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const emailInput = document.getElementById('email-input');
    const btnAnalyze = document.getElementById('btn-analyze');
    const btnClear = document.getElementById('btn-clear');
    const btnPaste = document.getElementById('btn-paste');
    const statWords = document.getElementById('stat-words');
    const statChars = document.getElementById('stat-chars');

    const stateIdle = document.getElementById('state-idle');
    const stateLoading = document.getElementById('state-loading');
    const stateActive = document.getElementById('state-active');

    const verdictBanner = document.getElementById('verdict-banner');
    const verdictIcon = document.getElementById('verdict-icon');
    const verdictBadge = document.getElementById('verdict-badge');
    const verdictSummary = document.getElementById('verdict-summary');
    const scoreConfidence = document.getElementById('score-confidence');

    const barSpam = document.getElementById('bar-spam');
    const barHam = document.getElementById('bar-ham');
    const valSpamProb = document.getElementById('val-spam-prob');
    const valHamProb = document.getElementById('val-ham-prob');
    const probDominantTag = document.getElementById('prob-dominant-tag');

    const keywordsContainer = document.getElementById('keywords-container');
    const diagUppercase = document.getElementById('diag-uppercase');
    const diagSymbols = document.getElementById('diag-symbols');
    const diagWords = document.getElementById('diag-words');

    const historyTbody = document.getElementById('history-tbody');
    const btnClearHistory = document.getElementById('btn-clear-history');
    const systemStatusIndicator = document.getElementById('system-status-indicator');
    const systemStatusText = document.getElementById('system-status-text');

    // Preset Samples
    const SAMPLE_EMAILS = {
        lottery: "Congratulations! You've won a $1,000 Amazon gift card and free cash reward! Click here to claim your prize immediately before it expires: http://claim-reward-prize.xyz",
        bank: "URGENT SECURITY NOTICE: Your online banking access has been suspended due to unauthorized login attempts. Verify your password and account details immediately to restore service: http://secure-verify-bank.com",
        crypto: "Exclusive crypto giveaway! Send 0.1 BTC to our verified wallet address and receive 0.5 BTC instantly guaranteed. Free bonus code: CRYPTO2026. Claim your payout now.",
        work: "Hi Sarah, I have finalized the quarterly performance deck and attached the sprint roadmap for your review. Let me know if you need any adjustments before tomorrow's client presentation. Thanks!",
        meeting: "Good morning team, our sprint planning sync is rescheduled to 2:30 PM today in Conference Room B. Please make sure your backlog tickets are updated."
    };

    let historyList = [];

    // Initialize
    checkServerHealth();
    loadHistoryFromStorage();
    setupEventListeners();

    /**
     * Check backend health status
     */
    async function checkServerHealth() {
        try {
            const res = await fetch('/api/health');
            if (res.ok) {
                const data = await res.json();
                systemStatusIndicator.classList.add('online');
                systemStatusIndicator.classList.remove('offline');
                systemStatusText.textContent = data.model_loaded ? 'Engine Online (Naive Bayes ML)' : 'Engine Online (No Model)';
            } else {
                throw new Error('Server returned ' + res.status);
            }
        } catch (e) {
            systemStatusIndicator.classList.add('offline');
            systemStatusIndicator.classList.remove('online');
            systemStatusText.textContent = 'Backend Offline';
        }
    }

    /**
     * Event Listeners
     */
    function setupEventListeners() {
        // Textarea input counters
        emailInput.addEventListener('input', updateTextStats);

        // Analyze button
        btnAnalyze.addEventListener('click', () => {
            const text = emailInput.value.trim();
            if (!text) {
                showToast('Please enter or paste email text to analyze.', 'warning');
                emailInput.focus();
                return;
            }
            analyzeEmail(text);
        });

        // Clear button
        btnClear.addEventListener('click', () => {
            emailInput.value = '';
            updateTextStats();
            showIdleState();
            showToast('Input cleared');
        });

        // Paste button
        btnPaste.addEventListener('click', async () => {
            try {
                const text = await navigator.clipboard.readText();
                if (text) {
                    emailInput.value = text;
                    updateTextStats();
                    showToast('Text pasted from clipboard');
                    emailInput.focus();
                } else {
                    showToast('Clipboard is empty', 'warning');
                }
            } catch (err) {
                showToast('Clipboard access denied. Use Ctrl+V / Cmd+V', 'warning');
            }
        });

        // Keyboard Shortcut: Cmd/Ctrl + Enter
        emailInput.addEventListener('keydown', (e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                e.preventDefault();
                btnAnalyze.click();
            }
        });

        // Preset Chips
        document.querySelectorAll('.preset-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const key = chip.getAttribute('data-sample');
                if (SAMPLE_EMAILS[key]) {
                    emailInput.value = SAMPLE_EMAILS[key];
                    updateTextStats();
                    analyzeEmail(SAMPLE_EMAILS[key]);
                }
            });
        });

        // Clear History
        btnClearHistory.addEventListener('click', () => {
            historyList = [];
            localStorage.removeItem('spam_detector_history');
            renderHistoryTable();
            showToast('History cleared');
        });
    }

    /**
     * Update word and character counts
     */
    function updateTextStats() {
        const text = emailInput.value;
        const chars = text.length;
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;

        statWords.textContent = words;
        statChars.textContent = chars;
    }

    /**
     * Perform email analysis via API
     */
    async function analyzeEmail(text) {
        showLoadingState();

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to analyze email');
            }

            renderResults(data);
            addToHistory(data);
        } catch (error) {
            showToast(error.message, 'error');
            showIdleState();
        }
    }

    /**
     * State Switchers
     */
    function showIdleState() {
        stateIdle.classList.remove('hidden');
        stateLoading.classList.add('hidden');
        stateActive.classList.add('hidden');
    }

    function showLoadingState() {
        stateIdle.classList.add('hidden');
        stateLoading.classList.remove('hidden');
        stateActive.classList.add('hidden');
    }

    function showActiveState() {
        stateIdle.classList.add('hidden');
        stateLoading.classList.add('hidden');
        stateActive.classList.remove('hidden');
    }

    /**
     * Render the Analysis Results
     */
    function renderResults(data) {
        showActiveState();

        const isSpam = data.is_spam;
        const confidence = data.confidence;
        const spamProb = data.spam_probability;
        const hamProb = data.ham_probability;

        // Update Verdict Banner
        verdictBanner.className = `verdict-banner ${isSpam ? 'is-spam' : 'is-ham'}`;
        
        if (isSpam) {
            verdictBadge.textContent = 'SPAM / THREAT DETECTED';
            verdictSummary.textContent = 'High probability of phishing, fraudulent promotion, or malicious intent.';
            verdictIcon.innerHTML = `
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                    <line x1="12" y1="9" x2="12" y2="13"/>
                    <line x1="12" y1="17" x2="12.01" y2="17"/>
                </svg>
            `;
        } else {
            verdictBadge.textContent = 'LEGITIMATE (HAM) EMAIL';
            verdictSummary.textContent = 'Standard authentic email patterns detected with low security risk.';
            verdictIcon.innerHTML = `
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                    <polyline points="9 12 11 14 15 10"/>
                </svg>
            `;
        }

        scoreConfidence.textContent = `${confidence.toFixed(1)}%`;

        // Update Probability Bars
        barSpam.style.width = `${spamProb}%`;
        barHam.style.width = `${hamProb}%`;
        valSpamProb.textContent = `${spamProb.toFixed(1)}%`;
        valHamProb.textContent = `${hamProb.toFixed(1)}%`;
        probDominantTag.textContent = `${isSpam ? spamProb.toFixed(1) + '% Spam Probability' : hamProb.toFixed(1) + '% Ham Probability'}`;

        // Render Trigger Keywords
        keywordsContainer.innerHTML = '';
        if (data.keywords && data.keywords.length > 0) {
            data.keywords.forEach(kw => {
                const tag = document.createElement('span');
                tag.className = `keyword-tag risk-${kw.risk || 'neutral'}`;
                tag.innerHTML = `
                    <span>${escapeHtml(kw.word)}</span>
                    <span class="keyword-weight">(${kw.weight})</span>
                `;
                keywordsContainer.appendChild(tag);
            });
        } else {
            keywordsContainer.innerHTML = '<span class="text-dim" style="font-size: 12px;">No distinctive keyword signals detected.</span>';
        }

        // Diagnostics
        if (data.stats) {
            diagUppercase.textContent = `${data.stats.uppercase_ratio}%`;
            diagSymbols.textContent = data.stats.has_suspicious_symbols ? '⚠️ Detected' : 'Clean';
            diagSymbols.style.color = data.stats.has_suspicious_symbols ? '#fb7185' : '#34d399';
            diagWords.textContent = data.stats.word_count;
        }
    }

    /**
     * Session History Management
     */
    function addToHistory(data) {
        const item = {
            id: Date.now(),
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
            text: data.text,
            label: data.label,
            is_spam: data.is_spam,
            confidence: data.confidence
        };

        historyList.unshift(item);
        if (historyList.length > 10) historyList.pop();

        localStorage.setItem('spam_detector_history', JSON.stringify(historyList));
        renderHistoryTable();
    }

    function loadHistoryFromStorage() {
        try {
            const saved = localStorage.getItem('spam_detector_history');
            if (saved) {
                historyList = JSON.parse(saved);
                renderHistoryTable();
            }
        } catch (e) {
            historyList = [];
        }
    }

    function renderHistoryTable() {
        if (!historyList || historyList.length === 0) {
            historyTbody.innerHTML = `
                <tr class="history-empty">
                    <td colspan="5">No emails analyzed yet in this session.</td>
                </tr>
            `;
            return;
        }

        historyTbody.innerHTML = historyList.map(item => `
            <tr>
                <td class="history-time">${item.timestamp}</td>
                <td>
                    <div class="history-snippet" title="${escapeHtml(item.text)}">${escapeHtml(item.text)}</div>
                </td>
                <td>
                    <span class="history-badge ${item.is_spam ? 'badge-spam' : 'badge-ham'}">
                        ${item.label}
                    </span>
                </td>
                <td style="font-family: var(--font-mono);">${item.confidence.toFixed(1)}%</td>
                <td>
                    <button type="button" class="btn-ghost btn-xs" onclick="window.reloadHistoryItem(${item.id})">Reload</button>
                </td>
            </tr>
        `).join('');
    }

    // Expose reload function to window
    window.reloadHistoryItem = function(id) {
        const item = historyList.find(h => h.id === id);
        if (item) {
            emailInput.value = item.text;
            updateTextStats();
            analyzeEmail(item.text);
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
    };

    /**
     * Toast notification helper
     */
    function showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.textContent = message;
        container.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(10px)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 2500);
    }

    function escapeHtml(str) {
        if (!str) return '';
        return str.replace(/[&<>'"]/g, 
            tag => ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                "'": '&#39;',
                '"': '&quot;'
            }[tag] || tag)
        );
    }
});
