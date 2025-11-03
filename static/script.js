// static/script.js (rebuilt)

let myBarChart = null;
let myPieChart = null;
let lastAnalysisData = null;

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

// Slider display spans
['diffThreshold', 'minArea', 'morphIter'].forEach(id => {
    const slider = document.getElementById(id);
    if (!slider) return;
    // Find the <span> and the <input type="number">
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

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    lastAnalysisData = null;

    const templateFile = templateInput.files[0];
    const testFile = testInput.files[0];
    if (!templateFile || !testFile) { showError('⚠️ Please upload both Template and Test images!'); return; }

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

    if (myBarChart) myBarChart.destroy();
    if (myPieChart) myPieChart.destroy();

    try {
        const res = await fetch('/api/detect', { method: 'POST', body: formData });
        const data = await res.json(); // Wait for the JSON promise to resolve
        if (!res.ok) throw new Error(data?.error || res.statusText);
        if (!data || !data.annotated_image_url || !Array.isArray(data.defects)) throw new Error('Invalid response');

        lastAnalysisData = data;
        successMessage.textContent = '✅ Analysis Complete!';
        successMessage.style.display = 'block';
        outputDisplay.style.display = 'block';

        resultImage.src = data.annotated_image_url;
        diffImage.src = data.diff_image_url;
        maskImage.src = data.mask_image_url;

        const defects = data.defects;
        const total = defects.length;
        defectCount.textContent = total;
        summaryBody.innerHTML = '';

        const summaryCounts = summarizeDefects(defects);

        if (total === 0) {
            noDefectsMessage.style.display = 'block';
            summaryBody.innerHTML = '<tr><td colspan="6">✅ No defects found!</td></tr>';
            // Hide chart containers
            document.getElementById('chart-container-bar').style.display = 'none';
            document.getElementById('chart-container-pie').style.display = 'none';
        } else {
              // Show chart containers
            document.getElementById('chart-container-bar').style.display = 'block';
            document.getElementById('chart-container-pie').style.display = 'block';

            // Populate table
            defects.forEach(d => {
                const row = summaryBody.insertRow();
                row.innerHTML = `<td>${d.id}</td><td>${d.label}</td><td>${(d.confidence*100).toFixed(2)}%</td><td>(${d.x}, ${d.y})</td><td>(${d.w}, ${d.h})</td><td>${d.area}</td>`;
            });

            // Render charts
            renderDefectChart(summaryCounts);
             renderDefectPie(summaryCounts);
        }

        // Add Download buttons
        const downloadImgLink = document.createElement('a');
        downloadImgLink.href = data.annotated_image_url;
        const safeFilename = testFile.name.replace(/[^a-zA-Z0-9.\-_]/g, '_');
        downloadImgLink.download = `annotated_${safeFilename}.png`;
        downloadImgLink.textContent = '⬇️ Download Annotated Image';
        downloadImgLink.className = 'btn-download';
        downloadButtonContainer.appendChild(downloadImgLink);

        const pdfButton = document.createElement('button');
        pdfButton.id = 'download-pdf-button';
        pdfButton.className = 'btn-download pdf-button';
        pdfButton.textContent = '⬇️ Download PDF Report';
        pdfButton.onclick = generatePDF;
        downloadButtonContainer.appendChild(pdfButton);

    } catch (err) {
        showError(err.message || String(err));
        console.error(err);
    } finally {
        spinner.style.display = 'none';
        detectButton.disabled = false;
        detectButton.textContent = '🚀 Detect Defects';
    }
});

function renderDefectChart(summaryData) {
    const ctx = document.getElementById('defectCountChart').getContext('2d');
    const labels = Object.keys(summaryData);
    const data = Object.values(summaryData);
    if (myBarChart) myBarChart.destroy();
    myBarChart = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets: [{ label: 'Defect Count', data, backgroundColor: 'rgba(54,162,235,0.6)', borderColor: 'rgba(54,162,235,1)', borderWidth: 1 }] },
        options: { 
            responsive: true, 
            maintainAspectRatio: false, // <-- THIS IS A KEY FIX
            scales: { y: { beginAtZero: true, ticks: { precision: 0 } } }, 
            plugins: { legend: { display: false } } 
        }
    });
}

function renderDefectPie(summaryData) {
    const ctx = document.getElementById('defectPieChart').getContext('2d');
    const labels = Object.keys(summaryData);
    const data = Object.values(summaryData);
    const colorMap = { 'copper':'rgba(255,159,64,0.7)','mousebite':'rgba(75,192,192,0.7)','open':'rgba(54,162,235,0.7)','pin-hole':'rgba(255,206,86,0.7)','short':'rgba(255,99,132,0.7)','spur':'rgba(153,102,255,0.7)','unknown':'rgba(201,203,207,0.7)' };
    const backgroundColor = labels.map(l => colorMap[l] || colorMap['unknown']);
    if (myPieChart) myPieChart.destroy();
    myPieChart = new Chart(ctx, { 
        type: 'pie', 
        data: { labels, datasets: [{ data, backgroundColor, borderColor: '#fff', borderWidth: 1 }] }, 
        options: { 
            responsive: true, 
            maintainAspectRatio: false, // <-- THIS IS A KEY FIX
            plugins: { legend: { position: 'bottom' } } 
        } 
    });
}

function renderScatterPlot(defects) {
    const ctx = document.getElementById('defectScatterPlot').getContext('2d');
    if (myScatterChart) myScatterChart.destroy();
    const colors = { 'copper':'rgba(255,159,64,0.7)','mousebite':'rgba(75,192,192,0.7)','open':'rgba(54,162,235,0.7)','pin-hole':'rgba(255,206,86,0.7)','short':'rgba(255,99,132,0.7)','spur':'rgba(153,102,255,0.7)','unknown':'rgba(201,203,207,0.7)' };
    const datasets = {};
    (defects || []).forEach(d => { if (!datasets[d.label]) datasets[d.label] = { label: d.label, data: [], backgroundColor: colors[d.label] || colors['unknown'], pointRadius: 5 }; datasets[d.label].data.push({ x: d.x, y: d.y }); });
    const finalDatasets = Object.values(datasets);
    if (finalDatasets.length === 0) finalDatasets.push({ label: 'No Data', data: [], backgroundColor: 'rgba(201,203,207,0.4)', pointRadius: 0 });
    
    myScatterChart = new Chart(ctx, { 
        type: 'scatter', 
        data: { datasets: finalDatasets }, 
        options: { 
            responsive: true, 
            maintainAspectRatio: false, // <-- THIS IS A KEY FIX
            scales: { 
                x: { title: { display: true, text: 'X (px)' } }, 
                y: { title: { display: true, text: 'Y (px)' }, reverse: true } 
            }, 
            plugins: { legend: { position: 'bottom' } } 
        } 
    });
}

function showError(message) { 
    errorMessage.textContent = message; 
    errorMessage.style.display = 'block'; 
    successMessage.style.display = 'none'; 
    outputDisplay.style.display = 'none'; 
}

function summarizeDefects(defects) { const c = {}; defects.forEach(d => { c[d.label] = (c[d.label] || 0) + 1; }); return c; }

async function generatePDF() {
    if (!lastAnalysisData) { alert('Run an analysis first.'); return; }
    
    // Check if jsPDF is loaded
    if (typeof window.jspdf === 'undefined') {
        console.error("jsPDF library is not loaded!");
        alert("Error: PDF generation library (jsPDF) is missing.");
        return;
    }
    // Check if html2canvas is loaded
    if (typeof html2canvas === 'undefined') {
        console.error("html2canvas library is not loaded!");
        alert("Error: PDF generation library (html2canvas) is missing.");
        return;
    }

    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF('p','mm','a4'); // A4 Portrait
    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = pdf.internal.pageSize.getHeight();
    const margin = 15;
    const contentWidth = pdfWidth - (margin * 2);
    let yPos = 20;
    
    const templateName = templateInput.files[0]?.name || 'N/A';
    const testName = testInput.files[0]?.name || 'N/A';

    const pdfButton = document.getElementById('download-pdf-button');
    pdfButton.disabled = true;
    pdfButton.textContent = '⏳ Generating PDF...';

    try {
        pdf.setFontSize(20); pdf.text('CircuitGuard - Defect Analysis Report', pdfWidth/2, yPos, { align:'center' }); yPos+=15;
        pdf.setFontSize(12); pdf.text('Analysis Details', margin, yPos); yPos+=7; pdf.setFontSize(10);
        pdf.text(`Template Image: ${templateName}`, margin, yPos); yPos+=5; pdf.text(`Test Image: ${testName}`, margin, yPos); yPos+=10;
        
        const total = lastAnalysisData.defects.length;
        pdf.setFontSize(12); pdf.text('Defect Summary', margin, yPos); yPos+=7; pdf.setFontSize(10);
        pdf.text(`Total Defects Found: ${total}`, margin, yPos); yPos+=7;

        if (total > 0) {
            // Add Summary Table
            try {
                // Check if jsPDF-AutoTable plugin is available
                if (typeof pdf.autoTable === 'function') {
                    const tableBody = lastAnalysisData.defects.map(d => [d.id, d.label, `${(d.confidence*100).toFixed(2)}%`, `(${d.x}, ${d.y})`, `(${d.w}, ${d.h})`, d.area]);
                    pdf.autoTable({
                        head: [['#', 'Class', 'Confidence', 'Position (x, y)', 'Size (w, h)', 'Area (px)']],
                        body: tableBody,
                        startY: yPos,
                        styles: { fontSize: 8 },
                        headStyles: { fillColor: [13, 110, 253] } // Blue header
                    });
                    yPos = pdf.lastAutoTable.finalY + 10; // Get Y pos after table
                } else {
                    // Fallback to simple text table
                    const summaryCounts = summarizeDefects(lastAnalysisData.defects); 
                    pdf.setFont('helvetica','bold'); pdf.text('Defect Type', margin, yPos); pdf.text('Count', margin+50, yPos); yPos+=5; pdf.setFont('helvetica','normal'); 
                    for (const [l,c] of Object.entries(summaryCounts)) { pdf.text(l, margin, yPos); pdf.text(String(c), margin+50, yPos); yPos+=5; } yPos+=5;
                }
            } catch (e) { console.error("Error drawing table:", e); }
        }

        // Add Charts
        if (total > 0) {
            if (yPos + 80 > pdfHeight) { pdf.addPage(); yPos = 20; }
            pdf.setFontSize(12); pdf.text("Visualizations", margin, yPos); yPos += 7;
            
            const barCanvas = document.getElementById('defectCountChart');
            const pieCanvas = document.getElementById('defectPieChart');
            const chartWidth = (pdfWidth - margin*2 - 10) / 2; // Half width minus gap
            
            try {
                if (barCanvas) {
                    const barImg = barCanvas.toDataURL('image/png', 1.0);
                    const barHeight = (barCanvas.height * chartWidth) / barCanvas.width;
                    pdf.addImage(barImg,'PNG', margin, yPos, chartWidth, barHeight);
                }
                if (pieCanvas) {
                    const pieImg = pieCanvas.toDataURL('image/png', 1.0);
                    const pieHeight = (pieCanvas.height * chartWidth) / pieCanvas.width;
                    pdf.addImage(pieImg,'PNG', margin + chartWidth + 10, yPos, chartWidth, pieHeight);
                    yPos += Math.max(pieHeight, barHeight || 0) + 10;
                }
                // Scatter removed
            } catch(e) { console.error("Error adding charts to PDF:", e); }
        }

        // Add Annotated Image
        if (yPos + 80 > pdfHeight) { pdf.addPage(); yPos = 20; }
        pdf.setFontSize(12); pdf.text('Annotated Image', margin, yPos); yPos+=7;
        try { 
            const imgData = lastAnalysisData.annotated_image_url; 
            const props = pdf.getImageProperties(imgData); 
            const imgWidth = pdfWidth - margin*2; 
            const imgHeight = (props.height * imgWidth)/props.width; 
            if (yPos + imgHeight > pdfHeight) { pdf.addPage(); yPos=20; pdf.text('Annotated Image (Continued)', margin, yPos); yPos+=7; } 
            pdf.addImage(imgData,'PNG', margin, yPos, imgWidth, imgHeight);
        } catch(e) { console.error("Error adding annotated image to PDF:", e); }

        // --- PDF SAVING ---
        const safeName = (testName || 'report').replace(/[^a-zA-Z0-9.\-_]/g,'_'); 
        pdf.save(`CircuitGuard_Report_${safeName}.pdf`);
    
    } catch(e) {
        console.error("Error generating PDF:", e);
        alert("An error occurred while generating the PDF. Check console for details.");
    } finally {
        pdfButton.disabled = false; 
        pdfButton.textContent = '⬇️ Download PDF Report';
    }
}
// <-- NO EXTRA BRACE HERE