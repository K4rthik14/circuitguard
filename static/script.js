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
// ▲▲▲ END OF ADDED SECTION ▲▲▲


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
function createDefectChart(summaryCounts) {
    const ctx = chartCanvas.getContext('2d');
    const labels = Object.keys(summaryCounts);
    const data = Object.values(summaryCounts);

    // If a chart already exists, destroy it
    if (window.myDefectChart) {
        window.myDefectChart.destroy();
    }

    // Create the new chart
    window.myDefectChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Defect Count',
                data: data,
                backgroundColor: [
                    'rgba(255, 99, 132, 0.2)',
                    'rgba(54, 162, 235, 0.2)',
                    'rgba(255, 206, 86, 0.2)',
                    'rgba(75, 192, 192, 0.2)',
                    'rgba(153, 102, 255, 0.2)',
                    'rgba(255, 159, 64, 0.2)'
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(75, 192, 192, 1)',
                    'rgba(153, 102, 255, 1)',
                    'rgba(255, 159, 64, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        // Ensure only whole numbers are used for ticks
                        stepSize: 1
                    }
                }
            },
            responsive: true,
            plugins: {
                legend: {
                    display: false // Hide legend since it's a simple chart
                }
            }
        }
    });

    chartContainer.style.display = 'block'; // Show the chart
}

// ▲▲▲ END OF ADDED FUNCTIONS ▲▲▲


// --- Form Submission ---
form.addEventListener("submit", async (event) => {
    event.preventDefault(); // Prevent default form submission & page reload

    // ... (your existing validation and FormData logic) ...
    const templateFile = templateInput.files[0];
    const testFile = testInput.files[0];

    // --- Validation ---
    if (!templateFile || !testFile) {
        errorMessage.textContent = "⚠️ Please upload both Template and Test images!";
        successMessage.textContent = '';
        errorMessage.style.display = 'block'; // Show error
        resultsSection.style.display = "block";
        outputDisplay.style.display = 'none';
        spinner.style.display = 'none';
        noDefectsMessage.style.display = 'none';
        return;
    }

    // --- Prepare FormData ---
    const formData = new FormData();
    formData.append("template_image", templateFile);
    formData.append("test_image", testFile);

    // --- UI Update: Start Processing ---
    spinner.style.display = "block"; // Show spinner next to button
    detectButton.disabled = true;
    detectButton.textContent = 'Processing...';
    errorMessage.textContent = ""; // Clear errors
    errorMessage.style.display = 'none'; // Hide error area
    successMessage.textContent = "";
    successMessage.style.display = 'none'; // Hide success area
    resultsSection.style.display = "block"; // Show results section container
    outputDisplay.style.display = "none"; // Hide specific output area initially
    noDefectsMessage.style.display = 'none';
    downloadButtonContainer.innerHTML = '';
    summaryBody.innerHTML = '<tr><td colspan="2"><em>Processing...</em></td></tr>'; // Update table state
    
    // ▼▼▼ HIDE CHART CONTAINER ON NEW SUBMISSION ▼▼▼
    chartContainer.style.display = 'none';
    if (window.myDefectChart) {
        window.myDefectChart.destroy(); // Clear old chart
    }

    try {
        // --- API Call ---
        console.log("Sending request to /api/detect");
        const response = await fetch("/api/detect", {
            method: "POST",
            body: formData,
        });
        console.log(`Received response with status: ${response.status}`);

        // --- Handle API Response ---
        spinner.style.display = "none"; // Hide spinner first

        if (!response.ok) {
            // ... (your existing error handling - no changes needed)
            let errorMsg = `Error ${response.status}: ${response.statusText}`;
            try {
                 const errorText = await response.text();
                 if (errorText.trim().startsWith('{')) {
                     const errorData = JSON.parse(errorText);
                     if (errorData && errorData.error) {
                         errorMsg = `Server Error: ${errorData.error}`;
                     }
                 } else if (errorText) {
                     errorMsg = `Server Error: ${errorText}`;
                 }
            } catch (e) { console.warn("Could not parse error response."); }
            throw new Error(errorMsg); // Propagate error
        }

        // --- Process Successful JSON Response ---
        const data = await response.json(); // Expecting JSON
        console.log("Received data:", data);

        // ... (your existing data validation - no changes needed)
        if (!data || typeof data.annotated_image_url !== 'string' || !Array.isArray(data.defects)) {
             console.error("Invalid JSON response format:", data);
             throw new Error("Invalid response format from server.");
        }

        const defects = data.defects;
        const imageUrl = data.annotated_image_url; // Base64 Data URL

        // --- Update UI with Actual Data ---
        successMessage.textContent = '✅ Analysis Complete!';
        successMessage.style.display = 'block';
        outputDisplay.style.display = "block"; // Show the output area

        // Update defect summary table
        const totalDefects = defects.length;
        defectCount.textContent = totalDefects; // Update total count span

        summaryBody.innerHTML = ""; // Clear "Processing..." message
        if (totalDefects === 0) {
            noDefectsMessage.style.display = 'block'; // Show "No defects" message
            summaryBody.innerHTML = `<tr><td colspan="2">✅ No defects found!</td></tr>`; // Indicate in table too
            resultImage.style.display = 'none'; // Hide image container if no defects?
            downloadButtonContainer.innerHTML = ''; // No download button if no defects
            
            // ▼▼▼ HIDE CHART CONTAINER IF NO DEFECTS ▼▼▼
            chartContainer.style.display = 'none';

        } else {
            noDefectsMessage.style.display = 'none'; // Hide "No defects" message
            
            // ▼▼▼ GET SUMMARY COUNTS (REPLACES OLD LOGIC) ▼▼▼
            const summaryCounts = summarizeDefects(defects);

            // Populate table rows
            for (const [type, count] of Object.entries(summaryCounts)) {
                const row = summaryBody.insertRow();
                const cellType = row.insertCell();
                const cellCount = row.insertCell();
                cellType.textContent = type;
                cellCount.textContent = count;
            }
            
            // ▼▼▼ CALL CHART FUNCTION ▼▼▼
            createDefectChart(summaryCounts);


            // Display annotated image using the data URL
            resultImage.src = imageUrl;
            resultImage.style.display = "block";

            // Set up download link for the annotated image (using the data URL)
            // ... (your existing download link logic - no changes)
            const downloadLink = document.createElement('a');
            downloadLink.href = imageUrl;
            const safeFilename = testFile.name.replace(/[^a-zA-Z0-9.\-_]/g, '_');
            downloadLink.download = `annotated_${safeFilename}.png`; // Ensure .png
            downloadLink.textContent = '⬇️ Download Annotated Image';
            downloadLink.className = 'btn-download'; // Use class for styling
            downloadButtonContainer.appendChild(downloadLink);
        }

    } catch (err) {
        // --- Handle Errors ---
        // ... (your existing catch block - no changes needed)
        spinner.style.display = "none";
        errorMessage.textContent = "Error: " + err.message;
        errorMessage.style.display = 'block'; // Make error visible
        successMessage.style.display = 'none'; // Hide success message
        outputDisplay.style.display = "none"; // Hide output area on error
        console.error("Detection Error:", err);
    } finally {
         detectButton.disabled = false; // Re-enable button
         detectButton.textContent = '🚀 Detect Defects'; // Restore original text
    }
});