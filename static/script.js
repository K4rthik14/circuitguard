// static/script.js

// --- DOM Element References ---
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
const defectCount = document.getElementById('defect-count'); // Span for total count
const summaryBody = document.getElementById('summary-body'); // tbody for summary table
const noDefectsMessage = document.getElementById('no-defects-message');
const downloadButtonContainer = document.getElementById('download-button-container');
const detectButton = document.getElementById('detect-button');

// ▼▼▼ ADD THIS SECTION ▼▼▼
const chartContainer = document.getElementById('chart-container');
const chartCanvas = document.getElementById('defect-chart');
// This variable will hold our chart instance so we can destroy it later
window.myDefectChart = null; 

// Update live values beside sliders
['diffThreshold', 'minArea', 'morphIter'].forEach(id => {
    const slider = document.getElementById(id);
    const display = document.getElementById(id === 'diffThreshold' ? 'diffVal' :
                    id === 'minArea' ? 'areaVal' : 'morphVal');
    slider.addEventListener('input', () => display.textContent = slider.value);
});


// --- Image Preview Logic ---
function setupPreview(inputId, previewId) {
    // ... (your existing setupPreview function - no changes needed)
    const input = document.getElementById(inputId);
    const preview = document.getElementById(previewId);

    input.addEventListener("change", () => {
        const file = input.files[0];
        if (file && file.type.startsWith("image/")) {
            const reader = new FileReader();
            reader.onload = e => {
                preview.src = e.target.result;
                preview.style.display = "block";
            };
            reader.onerror = () => { // Handle read error
                console.error("Error reading file for preview.");
                preview.style.display = "none";
                alert("Error reading file for preview.");
            };
            reader.readAsDataURL(file);
        } else {
            preview.src = "#"; // Clear src
            preview.style.display = "none";
            if(file) { // If file selected but not image
                 errorMessage.textContent = '⚠️ Please select a valid image file (PNG or JPG).';
                 errorMessage.style.display = 'block'; // Show error
                 resultsSection.style.display = 'block'; // Ensure section is visible for error
                 outputDisplay.style.display = 'none';
                 input.value = ""; // Reset file input
            }
        }
    });
}
setupPreview("template_image", "template-preview");
setupPreview("test_image", "test-preview");

// ▼▼▼ ADD THESE NEW HELPER FUNCTIONS ▼▼▼

/**
 * Counts the occurrences of each defect label.
 * @param {Array} defects - The list of defect objects from the API.
 * @returns {Object} - An object like { open: 2, short: 1, ... }
 */
function summarizeDefects(defects) {
    const summaryCounts = {};
    defects.forEach(defect => {
        summaryCounts[defect.label] = (summaryCounts[defect.label] || 0) + 1;
    });
    return summaryCounts;
}

/**
 * Creates or updates the bar chart.
 * @param {Object} summaryCounts - The object from summarizeDefects.
 */
function createDefectCharts(summaryCounts) {
    const ctx1 = document.getElementById('histogramCanvas').getContext('2d');
    const ctx2 = document.getElementById('pieCanvas').getContext('2d');
    const labels = Object.keys(summaryCounts);
    const data = Object.values(summaryCounts);

    if (window.histogramChart) window.histogramChart.destroy();
    if (window.pieChart) window.pieChart.destroy();

    window.histogramChart = new Chart(ctx1, {
        type: 'bar',
        data: { labels, datasets: [{ label: 'Defect Count', data }] },
        options: { responsive: true, plugins: { title: { display: true, text: 'Defect Histogram' } } }
    });

    window.pieChart = new Chart(ctx2, {
        type: 'pie',
        data: { labels, datasets: [{ data }] },
        options: { responsive: true, plugins: { title: { display: true, text: 'Defect Distribution' } } }
    });
}

// ▲▲▲ END OF ADDED FUNCTIONS ▲▲▲


form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const templateFile = templateInput.files[0];
    const testFile = testInput.files[0];
    if (!templateFile || !testFile) {
        errorMessage.textContent = "⚠️ Please upload both Template and Test images!";
        errorMessage.style.display = 'block';
        return;
    }

    // ✅ Get slider values
    const diffThreshold = document.getElementById('diffThreshold').value;
    const minArea = document.getElementById('minArea').value;
    const morphIter = document.getElementById('morphIter').value;

    // Prepare FormData
    const formData = new FormData();
    formData.append("template_image", templateFile);
    formData.append("test_image", testFile);
    // Add slider values to the request
    formData.append("diff_threshold", document.getElementById("diffThreshold").value);
    formData.append("min_area", document.getElementById("minArea").value);
    formData.append("morph_iterations", document.getElementById("morphIter").value);


    spinner.style.display = "block";
    detectButton.disabled = true;
    detectButton.textContent = "Processing...";

    try {
        const response = await fetch("/api/detect", { method: "POST", body: formData });
        const data = await response.json();

        spinner.style.display = "none";
        detectButton.disabled = false;
        detectButton.textContent = "Detect Defects";

        if (!response.ok || !data.defects) throw new Error(data.error || "Detection failed.");

        const defects = data.defects;
        const imageUrl = data.annotated_image_url;
        const total = defects.length;

        resultsSection.style.display = "block";
        outputDisplay.style.display = "block";
        resultImage.src = imageUrl;
        defectCount.textContent = total;

        // ✅ Summarize & show chart
        const summaryCounts = summarizeDefects(defects);
        summaryBody.innerHTML = "";
        for (const [label, count] of Object.entries(summaryCounts)) {
            summaryBody.innerHTML += `<tr><td>${label}</td><td>${count}</td></tr>`;
        }

        createDefectCharts(summaryCounts); // fixed function name

        // ✅ Show download button
        const downloadLink = document.createElement('a');
        downloadLink.href = imageUrl;
        downloadLink.download = 'annotated_result.png';
        downloadLink.textContent = '⬇️ Download Annotated Image';
        downloadLink.className = 'btn-download';
        downloadButtonContainer.innerHTML = '';
        downloadButtonContainer.appendChild(downloadLink);

    } catch (err) {
        spinner.style.display = "none";
        detectButton.disabled = false;
        detectButton.textContent = "Detect Defects";
        errorMessage.textContent = err.message;
        errorMessage.style.display = 'block';
        console.error(err);
    }
});
