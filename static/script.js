// static/script.js

// --- Global Chart Instances ---
let myBarChart = null;
let myScatterChart = null;
// Store last data for PDF generation
let lastAnalysisData = null; 

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
const defectCount = document.getElementById('defect-count');
const summaryBody = document.getElementById('summary-body');
const noDefectsMessage = document.getElementById('no-defects-message');
const downloadButtonContainer = document.getElementById('download-button-container');
const detectButton = document.getElementById('detect-button');

// Intermediate images
const diffImage = document.getElementById('diff-image');
const maskImage = document.getElementById('mask-image');

// --- Setup Slider Listeners ---
['diffThreshold', 'minArea', 'morphIter'].forEach(id => {
    const slider = document.getElementById(id);
    if (slider) {
        // Find the <span> inside the <label> that's the slider's *previous sibling*
        const display = slider.parentElement.querySelector('label > span');
        if(display) {
            // Set initial value
            display.textContent = slider.value;
            // Update on input
            slider.addEventListener('input', () => {
                display.textContent = slider.value;
            });
        } else {
            console.warn(`Could not find display span for slider ${id}`);
        }
    }
});


// --- Image Preview Logic ---
function setupPreview(inputId, previewId) {
    const input = document.getElementById(inputId);
    const preview = document.getElementById(previewId);
    if (!input || !preview) return; // Guard clause

    input.addEventListener("change", () => {
        const file = input.files[0];
        if (file && file.type.startsWith("image/")) {
            const reader = new FileReader();
            reader.onload = e => {
                preview.src = e.target.result;
                preview.style.display = "block";
            };
            reader.readAsDataURL(file);
        } else {
            preview.src = "#";
            preview.style.display = "none";
        }
    });
}
setupPreview("template_image", "template-preview");
setupPreview("test_image", "test-preview");

// --- Form Submission ---
form.addEventListener("submit", async (event) => {
    event.preventDefault();
    lastAnalysisData = null; // Clear last analysis data

    const templateFile = templateInput.files[0];
    const testFile = testInput.files[0];

    if (!templateFile || !testFile) {
        showError("⚠️ Please upload both Template and Test images!");
        return;
    }

    // --- Prepare FormData ---
    const formData = new FormData();
    formData.append("template_image", templateFile);
    formData.append("test_image", testFile);
    // Add slider values
    formData.append("diffThreshold", document.getElementById("diffThreshold").value);
    formData.append("minArea", document.getElementById("minArea").value);
    formData.append("morphIter", document.getElementById("morphIter").value);

    // --- UI Update: Start Processing ---
    detectButton.disabled = true;
    detectButton.textContent = 'Processing...';
    spinner.style.display = 'block';
    resultsSection.style.display = 'block';
    outputDisplay.style.display = 'none';
    errorMessage.style.display = 'none';
    successMessage.style.display = 'none';
    noDefectsMessage.style.display = 'none';
    downloadButtonContainer.innerHTML = ''; // Clear buttons
    summaryBody.innerHTML = '<tr><td colspan="6"><em>Processing...</em></td></tr>'; // 6 columns now
    
    // Destroy old charts
    if (myBarChart) myBarChart.destroy();
    if (myScatterChart) myScatterChart.destroy();

    try {
        // --- API Call ---
        const response = await fetch("/api/detect", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            let errorMsg = `Error ${response.status}: ${response.statusText}`;
            try {
                const errorData = await response.json();
                if (errorData && errorData.error) errorMsg = `Server Error: ${errorData.error}`;
            } catch (e) { /* no json error */ }
            throw new Error(errorMsg);
        }

        const data = await response.json(); // Expecting JSON

        if (!data || !data.annotated_image_url || !Array.isArray(data.defects) || !data.diff_image_url || !data.mask_image_url) {
             throw new Error("Invalid response format from server.");
        }
        
        lastAnalysisData = data; // Save data for PDF export
        
        // --- Update UI with Actual Data ---
        successMessage.textContent = '✅ Analysis Complete!';
        successMessage.style.display = 'block';
        outputDisplay.style.display = "block";

        // Display images
        resultImage.src = data.annotated_image_url;
        diffImage.src = data.diff_image_url;
        maskImage.src = data.mask_image_url;

        // Update defect summary
        const defects = data.defects;
        const totalDefects = defects.length;
        defectCount.textContent = totalDefects;
        summaryBody.innerHTML = ""; // Clear "Processing..."

        if (totalDefects === 0) {
            noDefectsMessage.style.display = 'block';
            summaryBody.innerHTML = `<tr><td colspan="6">✅ No defects found!</td></tr>`;
            // Hide chart canvases if no defects
            document.getElementById('chart-container-bar').style.display = 'none';
            document.getElementById('chart-container-scatter').style.display = 'none';
        } else {
            // Show chart canvases
            document.getElementById('chart-container-bar').style.display = 'block';
            document.getElementById('chart-container-scatter').style.display = 'block';

            const summaryCounts = {};
            defects.forEach(defect => {
                // Populate summary table
                const row = summaryBody.insertRow();
                row.innerHTML = `
                    <td>${defect.id}</td>
                    <td>${defect.label}</td>
                    <td>${(defect.confidence * 100).toFixed(2)}%</td>
                    <td>(${defect.x}, ${defect.y})</td>
                    <td>(${defect.w}, ${defect.h})</td>
                    <td>${defect.area}</td>
                `;
                // Aggregate counts for chart
                summaryCounts[defect.label] = (summaryCounts[defect.label] || 0) + 1;
            });

            // Render charts
            renderDefectChart(summaryCounts);
            renderScatterPlot(defects);
        }

        // --- Add Download Buttons ---
        // 1. Annotated Image
        const downloadImgLink = document.createElement('a');
        downloadImgLink.href = data.annotated_image_url;
        const safeFilename = testFile.name.replace(/[^a-zA-Z0-9.\-_]/g, '_');
        downloadImgLink.download = `annotated_${safeFilename}.png`;
        downloadImgLink.textContent = '⬇️ Download Annotated Image';
        downloadImgLink.className = 'btn-download';
        downloadButtonContainer.appendChild(downloadImgLink);

        // 2. PDF Report Button
        const pdfButton = document.createElement('button');
        pdfButton.id = 'download-pdf-button';
        pdfButton.className = 'btn-download pdf-button';
        pdfButton.textContent = '⬇️ Download PDF Report';
        pdfButton.onclick = generatePDF; // Assign click handler
        downloadButtonContainer.appendChild(pdfButton);


    } catch (err) {
        showError(err.message);
        console.error("Detection Error:", err);
    } finally {
         spinner.style.display = "none";
         detectButton.disabled = false;
         detectButton.textContent = '🚀 Detect Defects';
    }
});

/**
 * Renders the defect count bar chart.
 * @param {Object} summaryData - e.g., {short: 2, open: 1}
 */
function renderDefectChart(summaryData) {
    const ctx = document.getElementById('defectCountChart').getContext('2d');
    const labels = Object.keys(summaryData);
    const data = Object.values(summaryData);
    
    if (myBarChart) myBarChart.destroy();
    myBarChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Defect Count',
                data: data,
                backgroundColor: 'rgba(54, 162, 235, 0.6)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { y: { beginAtZero: true, ticks: { precision: 0 } } },
            plugins: { legend: { display: false } }
        }
    });
}

/**
 * Renders the defect location scatter plot.
 * @param {Array} defects - The full list of defect objects.
 */
function renderScatterPlot(defects) {
    const ctx = document.getElementById('defectScatterPlot').getContext('2d');
    if (myScatterChart) myScatterChart.destroy();

    // Map labels to colors
    const colors = {
        'copper': 'rgba(255, 159, 64, 0.7)',
        'mousebite': 'rgba(75, 192, 192, 0.7)',
        'open': 'rgba(54, 162, 235, 0.7)',
        'pin-hole': 'rgba(255, 206, 86, 0.7)',
        'short': 'rgba(255, 99, 132, 0.7)',
        'spur': 'rgba(153, 102, 255, 0.7)',
        'unknown': 'rgba(201, 203, 207, 0.7)'
    };
    
    // Group defects by label for the scatter plot
    const datasets = {};
    defects.forEach(defect => {
        if (!datasets[defect.label]) {
            datasets[defect.label] = {
                label: defect.label,
                data: [],
                backgroundColor: colors[defect.label] || colors['unknown'],
                pointRadius: 6
            };
        }
        datasets[defect.label].data.push({ x: defect.x, y: defect.y });
    });

    myScatterChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: Object.values(datasets)
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'X Position (px)' } },
                y: { title: { display: true, text: 'Y Position (px)' }, reverse: true } // Y=0 is top
            },
            plugins: {
                legend: { position: 'bottom' },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) label += ': ';
                            label += `(x: ${context.parsed.x}, y: ${context.parsed.y})`;
                            return label;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Displays an error message in the UI.
 * @param {string} message - The error message to display.
 */
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    successMessage.style.display = 'none';
    outputDisplay.style.display = 'none';
}

/**
 * Helper function to summarize defects (used by PDF and charts).
 * @param {Array} defects - The full list of defect objects.
 * @returns {Object} - e.g., {short: 2, open: 1}
 */
function summarizeDefects(defects) {
    const summaryCounts = {};
    defects.forEach(defect => {
        summaryCounts[defect.label] = (summaryCounts[defect.label] || 0) + 1;
    });
    return summaryCounts;
}


/**
 * Generates and downloads a PDF report.
 */
async function generatePDF() {
    if (!lastAnalysisData) {
        alert("Please run an analysis first.");
        return;
    }

    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF('p', 'mm', 'a4'); // A4 Portrait
    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = pdf.internal.pageSize.getHeight();
    const margin = 15;
    const contentWidth = pdfWidth - (margin * 2);
    let yPos = 20; // Start Y position

    // Get file names
    const templateName = document.getElementById('template_image').files[0]?.name || 'N/A';
    const testName = document.getElementById('test_image').files[0]?.name || 'N/A';

    console.log("Generating PDF...");
    const pdfButton = document.getElementById('download-pdf-button');
    pdfButton.disabled = true;
    pdfButton.textContent = '⏳ Generating PDF...';

    // --- PDF CONTENT ---
    
    // 1. Title
    pdf.setFontSize(20);
    pdf.text("CircuitGuard - Defect Analysis Report", pdfWidth / 2, yPos, { align: 'center' });
    yPos += 15;

    // 2. File Info
    pdf.setFontSize(12);
    pdf.text("Analysis Details", margin, yPos);
    yPos += 7;
    pdf.setFontSize(10);
    pdf.text(`Template Image: ${templateName}`, margin, yPos);
    yPos += 5;
    pdf.text(`Test Image: ${testName}`, margin, yPos);
    yPos += 10;

    // 3. Summary
    pdf.setFontSize(12);
    pdf.text("Defect Summary", margin, yPos);
    yPos += 7;
    pdf.setFontSize(10);
    const totalDefects = lastAnalysisData.defects.length;
    pdf.text(`Total Defects Found: ${totalDefects}`, margin, yPos);
    yPos += 7;

    // 4. Add Summary Table
    if (totalDefects > 0) {
        try {
            // Use autoTable for better table formatting (if it was imported)
            // Since we don't have it, we'll do a simple text table
            const summaryCounts = summarizeDefects(lastAnalysisData.defects);
            pdf.setFont("helvetica", "bold");
            pdf.text("Defect Type", margin, yPos);
            pdf.text("Count", margin + 50, yPos);
            yPos += 5;
            pdf.setFont("helvetica", "normal");
            for (const [label, count] of Object.entries(summaryCounts)) {
                pdf.text(label, margin, yPos);
                pdf.text(count.toString(), margin + 50, yPos);
                yPos += 5;
            }
        } catch(e) { console.error("Error drawing table:", e); }
        yPos += 5; // Extra space after table
    }

    // 5. Add Charts (as images)
    try {
        const barCanvas = document.getElementById('defectCountChart');
        const scatterCanvas = document.getElementById('defectScatterPlot');
        
        if (yPos + 80 > pdfHeight) { // Check if space for at least one chart
            pdf.addPage();
            yPos = 20;
        }

        if (totalDefects > 0 && barCanvas) {
            pdf.setFontSize(12);
            pdf.text("Defect Count per Class", margin, yPos);
            yPos += 7;
            const barImgData = barCanvas.toDataURL('image/png');
            const chartWidth = (pdfWidth - margin * 2) / 2 - 5; // Half width minus small gap
            const barHeight = (barCanvas.height * chartWidth) / barCanvas.width;
            
            pdf.addImage(barImgData, 'PNG', margin, yPos, chartWidth, barHeight);
            
            if (scatterCanvas) {
                pdf.text("Defect Scatter Plot", margin + chartWidth + 10, yPos - 7);
                const scatterImgData = scatterCanvas.toDataURL('image/png');
                const scatterHeight = (scatterCanvas.height * chartWidth) / scatterCanvas.width;
                pdf.addImage(scatterImgData, 'PNG', margin + chartWidth + 10, yPos, chartWidth, scatterHeight);
            }
            yPos += Math.max(barHeight, scatterHeight || 0) + 10;
        }
    } catch(e) { console.error("Error adding charts to PDF:", e); }


    // 6. Add Annotated Image (on a new page if needed)
    if (yPos + 80 > pdfHeight) { // Check if space remaining
        pdf.addPage();
        yPos = 20;
    }
    
    pdf.setFontSize(12);
    pdf.text("Annotated Image", margin, yPos);
    yPos += 7;
    
    try {
        const imgData = lastAnalysisData.annotated_image_url;
        const imgProps = pdf.getImageProperties(imgData);
        const imgWidth = pdfWidth - margin * 2;
        const imgHeight = (imgProps.height * imgWidth) / imgProps.width;
        
        if (yPos + imgHeight > pdfHeight) { // Check again
             pdf.addPage();
             yPos = 20;
             pdf.text("Annotated Image (Continued)", margin, yPos);
             yPos += 7;
        }

        pdf.addImage(imgData, 'PNG', margin, yPos, imgWidth, imgHeight);
    } catch(e) { console.error("Error adding annotated image to PDF:", e); }
    
    // --- PDF SAVING ---
    pdf.save(`CircuitGuard_Report_${testFile.name}.pdf`);
    
    // Restore button
    pdfButton.disabled = false;
    pdfButton.textContent = '⬇️ Download PDF Report';
}
// <-- The extra '}' at the end of your original file is removed HERE