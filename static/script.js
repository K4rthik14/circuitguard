

let lastAnalysisData = null; // Stores the most recent API response

//Get all DOM elements
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

//Slider Value Synchronization
['diffThreshold', 'minArea', 'morphIter'].forEach(id => {
    const slider = document.getElementById(id);
    if (!slider) return;
    const displaySpan = slider.parentElement.parentElement.querySelector('label > span');
    const numInput = document.getElementById(id + 'Num');
    if (displaySpan) displaySpan.textContent = slider.value;
    if (numInput) numInput.value = slider.value;
    slider.addEventListener('input', () => {
        if (displaySpan) displaySpan.textContent = slider.value;
        if (numInput) numInput.value = slider.value;
    });
    if (numInput) {
        numInput.addEventListener('input', () => {
            const val = parseInt(numInput.value || '0', 10);
            const min = parseInt(numInput.min, 10);
            const max = parseInt(numInput.max, 10);
            const clampedVal = Math.max(min, Math.min(max, val));
            slider.value = clampedVal;
            numInput.value = clampedVal;
            if (displaySpan) displaySpan.textContent = String(clampedVal);
        });
    }
});

// --- Image Preview ---
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

//Main Form Submission Handler
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    lastAnalysisData = null;

    const templateFile = templateInput.files[0];
    const testFile = testInput.files[0];
    if (!templateFile || !testFile) {
        showError('⚠️ Please upload both Template and Test images!');
        return;
    }

    const formData = new FormData();
    formData.append('template_image', templateFile);
    formData.append('test_image', testFile);
    formData.append('diffThreshold', document.getElementById('diffThreshold').value);
    formData.append('minArea', document.getElementById('minArea').value);
    formData.append('morphIter', document.getElementById('morphIter').value);

    detectButton.disabled = true;
    detectButton.textContent = 'Processing...';
    spinner.style.display = 'block';
    resultsSection.style.display = 'block';
    outputDisplay.style.display = 'none';
    errorMessage.style.display = 'none';
    successMessage.style.display = 'none';
    noDefectsMessage.style.display = 'none';
    downloadButtonContainer.innerHTML = '';
    summaryBody.innerHTML = '<tr><td colspan="6"><em>Processing...</em></td></tr>';

    try {
        const res = await fetch('/api/detect', { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data?.error || res.statusText);
        if (!data || !data.annotated_image_url || !Array.isArray(data.defects)) {
            throw new Error('Invalid response from server');
        }

        lastAnalysisData = data;
        successMessage.textContent = ' Analysis Complete!';
        successMessage.style.display = 'block';
        outputDisplay.style.display = 'block';

        resultImage.src = data.annotated_image_url;
        diffImage.src = data.diff_image_url;
        maskImage.src = data.mask_image_url;

        const defects = data.defects;
        const total = defects.length;
        defectCount.textContent = total;
        summaryBody.innerHTML = '';

        if (total === 0) {
            noDefectsMessage.style.display = 'block';
            summaryBody.innerHTML = '<tr><td colspan="6"> No defects found!</td></tr>';
            document.getElementById('chart-container-bar').style.display = 'none';
            document.getElementById('chart-container-pie').style.display = 'none';
            document.getElementById("chart-container-scatter").style.display = 'none';
        } else {
            document.getElementById('chart-container-bar').style.display = 'block';
            document.getElementById('chart-container-pie').style.display = 'block';
            document.getElementById("chart-container-scatter").style.display = 'block';

            document.getElementById('bar-chart-img').src = data.bar_chart_url;
            document.getElementById('pie-chart-img').src = data.pie_chart_url;
            document.getElementById('scatter-chart-img').src = data.scatter_chart_url;

            defects.forEach(d => {
                const row = summaryBody.insertRow();
                row.innerHTML = `<td>${d.id}</td><td>${d.label}</td><td>${(d.confidence*100).toFixed(2)}%</td><td>(${d.x}, ${d.y})</td><td>(${d.w}, ${d.h})</td><td>${d.area}</td>`;
            });
        }

        // To create Download Button

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

        pdfButton.onclick = async () => {
            pdfButton.disabled = true;
            pdfButton.textContent = '⏳ Generating PDF...';

            try {
                // Re-create form data for the PDF endpoint
                const pdfFormData = new FormData();
                pdfFormData.append('template_image', templateInput.files[0]);
                pdfFormData.append('test_image', testInput.files[0]);
                pdfFormData.append('diffThreshold', document.getElementById('diffThreshold').value);
                pdfFormData.append('minArea', document.getElementById('minArea').value);
                pdfFormData.append('morphIter', document.getElementById('morphIter').value);

                const res = await fetch('/api/download_report', {
                    method: 'POST',
                    body: pdfFormData
                });

                if (!res.ok) {
                    const errData = await res.json().catch(() => ({error: 'PDF generation failed on server.'}));
                    throw new Error(errData.error || 'PDF generation failed');
                }

                const blob = await res.blob();
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                const safeFilename = testInput.files[0].name.replace(/[^a-zA-Z0-9.\-_]/g, '_');
                link.download = `CircuitGuard_Report_${safeFilename}.pdf`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                URL.revokeObjectURL(url);

            } catch (err) {
                alert('Error generating PDF: ' + err.message);
                console.error(err);
            } finally {
                pdfButton.disabled = false;
                pdfButton.textContent = '⬇️ Download PDF Report';
            }
        };
        downloadButtonContainer.appendChild(pdfButton);

        // Download CSV Log
        const csvButton = document.createElement('button');
        csvButton.id = 'download-csv-button';
        csvButton.className = 'btn-download csv-button';
        csvButton.textContent = '⬇️ Download CSV Log';
        // This function is now defined below
        csvButton.onclick = () => downloadCSV(lastAnalysisData);
        downloadButtonContainer.appendChild(csvButton);

    } catch (err) {
        showError(err.message || String(err));
        console.error(err);
    } finally {
        spinner.style.display = 'none';
        detectButton.disabled = false;
        detectButton.textContent = 'Detect Defects';
    }
});

//Helper function
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    successMessage.style.display = 'none';
    outputDisplay.style.display = 'none';
}

//CSV Download Function
function downloadCSV(analysisData) {
    if (!analysisData || !analysisData.defects) {
        alert('Run an analysis first or no defects found.');
        return;
    }

    const defects = analysisData.defects;

    //Get Metadata from the DOM ---
    const templateName = document.getElementById('template_image').files[0]?.name || 'N/A';
    const testName = document.getElementById('test_image').files[0]?.name || 'report';
    const safeName = (testName || 'report').replace(/[^a-zA-Z0-9.\-_]/g,'_');
    const filename = `CircuitGuard_Log_${safeName}.csv`;

    // Read parameters from the sliders
    const diffThreshold = document.getElementById('diffThreshold').value;
    const minArea = document.getElementById('minArea').value;
    const morphIter = document.getElementById('morphIter').value;

    //Build Metadata Header

    let csvContent = `"# CircuitGuard Analysis Log"\n`;
    csvContent += `"# Date: ${new Date().toISOString()}"\n`;
    csvContent += `"# Template Image: ${templateName}"\n`;
    csvContent += `"# Test Image: ${testName}"\n`;
    csvContent += `"#"\n`;
    csvContent += `"# Parameters Used:"\n`;
    csvContent += `"# Difference Threshold: ${diffThreshold}"\n`;
    csvContent += `"# Minimum Area: ${minArea}"\n`;
    csvContent += `"# Noise Filter Strength: ${morphIter}"\n`;
    csvContent += `"#"\n`;
    csvContent += `"# Total Defects Found: ${defects.length}"\n`;
    csvContent += `"#"\n`;

    //Build Data Table
    const headers = ['defect_id', 'label', 'confidence_percent', 'x', 'y', 'w', 'h', 'area_pixels'];
    csvContent += headers.join(',') + '\n'; // Add the data headers

    defects.forEach(d => {
        const confidencePercent = (d.confidence * 100).toFixed(2);
        const row = [d.id, d.label, confidencePercent, d.x, d.y, d.w, d.h, d.area];
        csvContent += row.join(',') + '\n';
    });

    //Create Blob and Download
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    if (link.download !== undefined) {
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', filename);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }
}