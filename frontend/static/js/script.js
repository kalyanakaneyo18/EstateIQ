/* ============================================================
   EstateIQ — script.js
   Handles form submission, prediction display, activity log.
   Backend API contract unchanged: POST /predict → { predicted_price }
   ============================================================ */

document.addEventListener('DOMContentLoaded', function () {

    // ── Init timestamp ──────────────────────────────────────────
    const initTimeEl = document.getElementById('init-time');
    if (initTimeEl) initTimeEl.textContent = currentTime();

    // ── Element refs ────────────────────────────────────────────
    const form          = document.getElementById('prediction-form');
    const predictBtn    = document.getElementById('predict-btn');
    const resultSection = document.getElementById('result-section');
    const errorSection  = document.getElementById('error-section');
    const errorMessage  = document.getElementById('error-message');
    const priceValue    = document.getElementById('price-value');
    const activityLog   = document.getElementById('activity-log');

    // ── Form submit ─────────────────────────────────────────────
    form.addEventListener('submit', async function (e) {
        e.preventDefault();

        // Hide previous feedback
        hideElement(errorSection);
        hideElement(resultSection);

        // Read values
        const area         = parseFloat(document.getElementById('area').value);
        const bedrooms     = parseInt(document.getElementById('bedrooms').value);
        const bathrooms    = parseInt(document.getElementById('bathrooms').value);
        const age          = parseInt(document.getElementById('age').value);
        const location     = document.getElementById('location').value;
        const propertyType = document.getElementById('property_type').value;

        // Basic validation
        if (!location || !propertyType) {
            showError('Please select both Location and Property Type.');
            return;
        }

        if (isNaN(area) || area <= 0) {
            showError('Please enter a valid area in sq. ft.');
            return;
        }

        // Log user request
        addLog('usr', `Predict: ${area} sq.ft, ${bedrooms}BHK, ${location}, ${propertyType}`);

        // Set loading state
        setLoading(true);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    area,
                    bedrooms,
                    bathrooms,
                    age,
                    location,
                    property_type: propertyType
                })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Prediction failed. Please try again.');
            }

            const data  = await response.json();
            const price = data.predicted_price;

            // Show result
            priceValue.textContent = '₹ ' + price.toLocaleString('en-IN');
            showElement(resultSection);

            addLog('res', `Estimated price: ₹ ${price.toLocaleString('en-IN')}`);

            // Smooth scroll to result
            resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        } catch (err) {
            showError(err.message || 'Something went wrong. Please try again.');
            addLog('err', err.message || 'Request failed');
        } finally {
            setLoading(false);
        }
    });

    // ── Helpers ─────────────────────────────────────────────────

    function setLoading(state) {
        predictBtn.disabled = state;
        if (state) {
            predictBtn.classList.add('loading');
        } else {
            predictBtn.classList.remove('loading');
        }
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        showElement(errorSection);
    }

    function showElement(el) { el.style.display = 'block'; }
    function hideElement(el) { el.style.display = 'none'; }

    function currentTime() {
        const now = new Date();
        return now.toTimeString().slice(0, 8);
    }

    function addLog(type, message) {
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.innerHTML = `
            <span class="log-tag ${type}">${type.toUpperCase()}</span>
            <span class="log-time">${currentTime()}</span>
            <span class="log-msg">${message}</span>
        `;
        activityLog.appendChild(entry);
        activityLog.scrollTop = activityLog.scrollHeight;
    }

    // Expose addLog globally for resetForm
    window._addLog = addLog;
});

// ── Global helpers (called via inline onclick) ───────────────

function resetForm() {
    document.getElementById('prediction-form').reset();
    document.getElementById('result-section').style.display  = 'none';
    document.getElementById('error-section').style.display   = 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });

    if (window._addLog) {
        window._addLog('sys', 'Form reset. Ready for new prediction.');
    }
}

function clearLog() {
    const log = document.getElementById('activity-log');
    log.innerHTML = '';
    if (window._addLog) {
        window._addLog('sys', 'Activity log cleared.');
    }
}