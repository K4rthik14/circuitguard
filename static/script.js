// static/script.js

let lastAnalysisData = null; // Stores the most recent API response

// --- Get all DOM elements ---
const form = document.getElementById('upload-form');
const templateInput = document.getElementById('template_image');
const testInput = document.getElementById('test_image');
const templatePreview = document.getElementById('template-preview');
const testPreview = document.getElementById('test-preview');
const resultsSection = document.getElementById('results-section');
const resultImage = document.getElementById('result-image');
const spinner = document.getElementById('spinner');
const errorMessage = document.getElementById('error-message');
const successMessage = document.getElementById('success-message');
const outputDisplay = document.getElementById('output-display');
const defectCount = document.getElementById('defect-count');
const summaryBody = document.getElementById('summary-body');
const noDefectsMessage = document.getElementById('no-defects-message');
const downloadButtonContainer = document.getElementById('download-button-container');
const detectButton = document.getElementById('detect-button');
const diffImage = document.getElementById('diff-image');
const maskImage = document.getElementById('mask-image');

// --- Slider Value Synchronization ---
// Connects sliders to their number inputs and text displays
['diffThreshold', 'minArea', 'morphIter'].forEach(id => {
    const slider = document.getElementById(id);
    if (!slider) return;

    const displaySpan = slider.parentElement.parentElement.querySelector('label > span');
    const numInput = document.getElementById(id + 'Num');

    // Set initial values from slider
    if (displaySpan) displaySpan.textContent = slider.value;
    if (numInput) numInput.value = slider.value;

    // Update span/number input when slider moves
    slider.addEventListener('input', () => {
        if (displaySpan) displaySpan.textContent = slider.value;
        if (numInput) numInput.value = slider.value;
    });

    // Update span/slider when number input changes
    if (numInput) {
        numInput.addEventListener('input', () => {
            const val = parseInt(numInput.value || '0', 10);
            const min = parseInt(numInput.min, 10);
            const max = parseInt(numInput.max, 10);
            // Ensure value stays within min/max range
            const clampedVal = Math.max(min, Math.min(max, val));

            slider.value = clampedVal;
            numInput.value = clampedVal;
            if (displaySpan) displaySpan.textContent = String(clampedVal);
        });
    }
});

// --- Image Preview ---
// Shows a preview of the selected image
function setupPreview(input, preview) {
    if (!input || !preview) return;
    input.addEventListener('change', () => {
        const file = input.files[0];
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = e => { preview.src = e.target.result; preview.style.display = 'block'; };
            reader.readAsDataURL(file);
        } else { preview.src = '#'; preview.style.display = 'none'; }
    });
}
setupPreview(templateInput, templatePreview);
setupPreview(testInput, testPreview);

// --- Main Form Submission Handler ---
form.addEventListener('submit', async (e) => {
    e.preventDefault(); // Stop default page refresh
    lastAnalysisData = null; // Clear old data

    const templateFile = templateInput.files[0];
    const testFile = testInput.files[0];
    if (!templateFile || !testFile) {
        showError('⚠️ Please upload both Template and Test images!');
        return;
    }

    // 1. Prepare form data to send
    const formData = new FormData();
    formData.append('template_image', templateFile);
    formData.append('test_image', testFile);
    formData.append('diffThreshold', document.getElementById('diffThreshold').value);
    formData.append('minArea', document.getElementById('minArea').value);
    formData.append('morphIter', document.getElementById('morphIter').value);

    // 2. Set UI to "loading" state
    detectButton.disabled = true;
    detectButton.textContent = 'Processing...';
    spinner.style.display = 'block';
    resultsSection.style.display = 'block'; // Show the whole section
    outputDisplay.style.display = 'none'; // Hide the results part
    errorMessage.style.display = 'none';
    successMessage.style.display = 'none';
    noDefectsMessage.style.display = 'none';
    downloadButtonContainer.innerHTML = ''; // Clear old buttons
    summaryBody.innerHTML = '<tr><td colspan="6"><em>Processing...</em></td></tr>';

    // (Chart.js .destroy() calls are removed)

    try {
        // 3. Send API request
        const res = await fetch('/api/detect', { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data?.error || res.statusText);
        if (!data || !data.annotated_image_url || !Array.isArray(data.defects)) {
            throw new Error('Invalid response from server');
        }

        // 4. Success! Store data and update UI
        lastAnalysisData = data;
        successMessage.textContent = ' Analysis Complete!';
        successMessage.style.display = 'block';
        outputDisplay.style.display = 'block'; // Show results

        // Set image sources from API response
        resultImage.src = data.annotated_image_url;
        diffImage.src = data.diff_image_url;
        maskImage.src = data.mask_image_url;

        const defects = data.defects;
        const total = defects.length;
        defectCount.textContent = total;
        summaryBody.innerHTML = ''; // Clear "Processing..."

        if (total === 0) {
            // Case: No defects found
            noDefectsMessage.style.display = 'block';
            summaryBody.innerHTML = '<tr><td colspan="6"> No defects found!</td></tr>';
            // Hide chart containers
            document.getElementById('chart-container-bar').style.display = 'none';
            document.getElementById('chart-container-pie').style.display = 'none';
            document.getElementById("chart-container-scatter").style.display = 'none';
        } else {
            // Case: Defects found
            // Show chart containers
            document.getElementById('chart-container-bar').style.display = 'block';
            document.getElementById('chart-container-pie').style.display = 'block';
            document.getElementById("chart-container-scatter").style.display = 'block';

            // Set chart image sources from API response
            document.getElementById('bar-chart-img').src = data.bar_chart_url;
            document.getElementById('pie-chart-img').src = data.pie_chart_url;
            document.getElementById('scatter-chart-img').src = data.scatter_chart_url;

            // Populate summary table
            defects.forEach(d => {
                const row = summaryBody.insertRow();
                row.innerHTML = `<td>${d.id}</td><td>${d.label}</td><td>${(d.confidence*100).toFixed(2)}%</td><td>(${d.x}, ${d.y})</td><td>(${d.w}, ${d.h})</td><td>${d.area}</td>`;
            });

            // (Chart.js render() calls are removed)
        }

        // 5. Create Download Buttons
        // Download Annotated Image
        const downloadImgLink = document.createElement('a');
        downloadImgLink.href = data.annotated_image_url;
        const safeFilename = testFile.name.replace(/[^a-zA-Z0-9.\-_]/g, '_');
        downloadImgLink.download = `annotated_${safeFilename}.png`;
        downloadImgLink.textContent = '⬇️ Download Annotated Image';
        downloadImgLink.className = 'btn-download';
        downloadButtonContainer.appendChild(downloadImgLink);

        // Download PDF Report
        const pdfButton = document.createElement('button');
        pdfButton.id = 'download-pdf-button';
        pdfButton.className = 'btn-download pdf-button';
        pdfButton.textContent = '⬇️ Download PDF Report';
        // Calls generatePDF (from report.js) and passes the data
        pdfButton.onclick = () => generatePDF(lastAnalysisData);
        downloadButtonContainer.appendChild(pdfButton);

        // Download CSV Log
        const csvButton = document.createElement('button');
        csvButton.id = 'download-csv-button';
        csvButton.className = 'btn-download csv-button';
        csvButton.textContent = '⬇️ Download CSV Log';
        // Calls downloadCSV (from report.js) and passes the data
        csvButton.onclick = () => downloadCSV(lastAnalysisData);
        downloadButtonContainer.appendChild(csvButton);

    } catch (err) {
        // 6. Handle Errors
        showError(err.message || String(err));
        console.error(err);
    } finally {
        // 7. Reset UI from "loading" state
        spinner.style.display = 'none';
        detectButton.disabled = false;
        detectButton.textContent = 'Detect Defects';
    }
});

// --- Helper function to show errors ---
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    successMessage.style.display = 'none';
    outputDisplay.style.display = 'none'; // Hide results on error
}