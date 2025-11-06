// static/report.js

// Make sure libraries are loaded (optional, but good practice)
var html2canvas = html2canvas || {};
var jspdf = jspdf || {};


async function generatePDF(analysisData) {
    if (!analysisData) { alert('Run an analysis first.'); return; }

    if (typeof window.jspdf === 'undefined' || typeof html2canvas === 'undefined' || typeof jspdf.plugin === 'undefined' || typeof jspdf.plugin.autotable === 'undefined') {
        console.error("jsPDF, html2canvas, or jspdf-autotable library is not loaded!");
        alert("Error: PDF generation libraries are missing. Check console.");
        return;
    }

    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF('p','mm','a4'); // A4 Portrait
    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = pdf.internal.pageSize.getHeight();
    const margin = 15;
    const contentWidth = pdfWidth - (margin * 2);
    let yPos = 20;

    // Get filenames from the input elements
    const templateName = document.getElementById('template_image').files[0]?.name || 'N/A';
    const testName = document.getElementById('test_image').files[0]?.name || 'N/A';

    const pdfButton = document.getElementById('download-pdf-button');
    pdfButton.disabled = true;
    pdfButton.textContent = '⏳ Generating PDF...';

    try {
        // --- 1. Title and Details ---
        pdf.setFontSize(20); pdf.text('CircuitGuard - Defect Analysis Report', pdfWidth/2, yPos, { align:'center' }); yPos+=15;
        pdf.setFontSize(12); pdf.text('Analysis Details', margin, yPos); yPos+=7; pdf.setFontSize(10);
        pdf.text(`Template Image: ${templateName}`, margin, yPos); yPos+=5; pdf.text(`Test Image: ${testName}`, margin, yPos); yPos+=10;

        const total = analysisData.defects.length;
        pdf.setFontSize(12); pdf.text('Defect Summary', margin, yPos); yPos+=7; pdf.setFontSize(10);
        pdf.text(`Total Defects Found: ${total}`, margin, yPos); yPos+=7;

        // --- 2. Summary Table ---
        if (total > 0) {
            try {
                const tableBody = analysisData.defects.map(d => [d.id, d.label, `${(d.confidence*100).toFixed(2)}%`, `(${d.x}, ${d.y})`, `(${d.w}, ${d.h})`, d.area]);
                pdf.autoTable({
                    head: [['#', 'Class', 'Confidence', 'Position (x, y)', 'Size (w, h)', 'Area (px)']],
                    body: tableBody,
                    startY: yPos,
                    styles: { fontSize: 8 },
                    headStyles: { fillColor: [13, 110, 253] }
                });
                yPos = pdf.lastAutoTable.finalY + 10;
            } catch (e) { console.error("Error drawing table:", e); }
        }

        // --- 3. Input Images (Template and Test) ---
        if (yPos + 80 > pdfHeight) { pdf.addPage(); yPos = 20; }

        pdf.setFontSize(12);
        pdf.text('Input Images', margin, yPos);
        yPos += 7;

        try {
            const templateImgData = document.getElementById('template-preview').src;
            const testImgData = document.getElementById('test-preview').src;

            if (!templateImgData || !templateImgData.startsWith('data:image/')) { throw new Error('Template preview image is not loaded.'); }
            if (!testImgData || !testImgData.startsWith('data:image/')) { throw new Error('Test preview image is not loaded.'); }

            const imgWidth = (contentWidth - 10) / 2;
            const templateProps = pdf.getImageProperties(templateImgData);
            const templateHeight = (templateProps.height * imgWidth) / templateProps.width;
            const testProps = pdf.getImageProperties(testImgData);
            const testHeight = (testProps.height * imgWidth) / testProps.width;
            const maxHeight = Math.max(templateHeight, testHeight);

            if (yPos + maxHeight + 10 > pdfHeight) { pdf.addPage(); yPos = 20; }

            pdf.setFontSize(10);
            pdf.text('Template Image', margin, yPos);
            pdf.addImage(templateImgData, 'PNG', margin, yPos + 3, imgWidth, templateHeight);
            pdf.text('Test Image', margin + imgWidth + 10, yPos);
            pdf.addImage(testImgData, 'PNG', margin + imgWidth + 10, yPos + 3, imgWidth, testHeight);

            yPos += maxHeight + 10;

        } catch (e) {
            console.error("Error adding input images to PDF:", e);
            yPos += 5;
        }

        // --- 4. Visualizations (Charts) ---
        if (total > 0) {
            if (yPos + 80 > pdfHeight) { pdf.addPage(); yPos = 20; }
            pdf.setFontSize(12); pdf.text("Visualizations", margin, yPos); yPos += 7;

            const barImgEl = document.getElementById('bar-chart-img');
            const pieImgEl = document.getElementById('pie-chart-img');
            const chartWidth = (contentWidth - 10) / 2;

            try {
                let barHeight = 0, pieHeight = 0;

                if (barImgEl && barImgEl.src.startsWith('data:image')) {
                    const barImg = barImgEl.src;
                    const props = pdf.getImageProperties(barImg);
                    barHeight = (props.height * chartWidth) / props.width;
                    pdf.addImage(barImg,'PNG', margin, yPos, chartWidth, barHeight);
                }
                if (pieImgEl && pieImgEl.src.startsWith('data:image')) {
                    const pieImg = pieImgEl.src;
                    const props = pdf.getImageProperties(pieImg);
                    pieHeight = (props.height * chartWidth) / props.width;
                    pdf.addImage(pieImg,'PNG', margin + chartWidth + 10, yPos, chartWidth, pieHeight);
                }
                yPos += Math.max(pieHeight, barHeight) + 10;

            } catch(e) { console.error("Error adding charts to PDF:", e); }
        }

        // --- 5. Annotated Image ---
        if (yPos + 80 > pdfHeight) { pdf.addPage(); yPos = 20; }
        pdf.setFontSize(12); pdf.text('Annotated Image', margin, yPos); yPos+=7;
        try {
            const imgData = analysisData.annotated_image_url;
            const props = pdf.getImageProperties(imgData);
            const imgWidth = contentWidth;
            const imgHeight = (props.height * imgWidth)/props.width;
            if (yPos + imgHeight > pdfHeight) { pdf.addPage(); yPos=20; pdf.text('Annotated Image (Continued)', margin, yPos); yPos+=7; }
            pdf.addImage(imgData,'PNG', margin, yPos, imgWidth, imgHeight);
        } catch(e) { console.error("Error adding annotated image to PDF:", e); }

        // --- 6. Save PDF ---
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

// Fuction to generate the CSV FILE
function downloadCSV(analysisData) {
    if (!analysisData || !analysisData.defects) {
        alert('Run an analysis first or no defects found.');
        return;
    }

    const defects = analysisData.defects;
    const testName = document.getElementById('test_image').files[0]?.name || 'report';
    const safeName = (testName || 'report').replace(/[^a-zA-Z0-9.\-_]/g,'_');
    const filename = `CircuitGuard_Log_${safeName}.csv`;

    const headers = ['id', 'label', 'confidence', 'x', 'y', 'w', 'h', 'area'];
    let csvContent = headers.join(',') + '\n';

    defects.forEach(d => {
        const confidencePercent = (d.confidence * 100).toFixed(2);
        const row = [d.id, d.label, confidencePercent, d.x, d.y, d.w, d.h, d.area];
        csvContent += row.join(',') + '\n';
    });

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