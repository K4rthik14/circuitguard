// static/script.js
const detectBtn = document.getElementById("detect-button");
const spinner = document.getElementById("spinner");
const resultSection = document.getElementById("results-section");
const resultImg = document.getElementById("result-image");
const downloadLink = document.getElementById("download-link");
const defectCount = document.getElementById("defect-count");
const summaryBody = document.getElementById("summary-body"); // Target the tbody

// --- Image Preview (keep as is) ---
function setupPreview(inputId, previewId) {
    // ... (preview code remains the same) ...
}
setupPreview("template_image", "template-preview");
setupPreview("test_image", "test-preview");

// --- Detection Button ---
detectBtn.addEventListener("click", async () => {
    const templateFile = document.getElementById("template_image").files[0];
    const testFile = document.getElementById("test_image").files[0];
    if (!templateFile || !testFile) {
        alert("⚠️ Please upload both Template and Test images!");
        return;
    }

    const formData = new FormData();
    formData.append("template_image", templateFile);
    formData.append("test_image", testFile);

    spinner.style.display = "block";
    detectBtn.disabled = true;
    resultSection.style.display = "none"; // Hide previous results
    resultImg.style.display = "none";
    downloadLink.style.display = "none";


    try {
        const response = await fetch("/api/detect", {
            method: "POST",
            body: formData,
        });

        spinner.style.display = "none"; // Hide spinner once response starts

        if (!response.ok) {
             // Try to get error message from API response if possible
            let errorMsg = `Error: ${response.status} ${response.statusText}`;
            try {
                const errorData = await response.json();
                if (errorData && errorData.error) {
                    errorMsg = `API Error: ${errorData.error}`;
                }
            } catch (e) { /* Response wasn't JSON */ }
            throw new Error(errorMsg);
        }

        // --- Handle JSON Response ---
        const data = await response.json(); // Expecting JSON now

        if (!data || !data.annotated_image_url || !data.defects) {
             throw new Error("Invalid response format from server.");
        }

        const defects = data.defects; // Get the actual defect list
        const imageUrl = data.annotated_image_url; // Get the image data URL

        // --- Update UI with Actual Data ---
        resultSection.style.display = "block"; // Show results section
        detectBtn.disabled = false;

        // Update defect summary table
        const totalDefects = defects.length;
        defectCount.textContent = totalDefects; // Update total count

        summaryBody.innerHTML = ""; // Clear previous/mock data
        if (totalDefects === 0) {
            summaryBody.innerHTML = `<tr><td colspan="2">✅ No defects found!</td></tr>`;
        } else {
            // Calculate counts per defect type
            const summaryCounts = {};
            defects.forEach(defect => {
                summaryCounts[defect.label] = (summaryCounts[defect.label] || 0) + 1;
            });

            // Populate table rows
            for (const [type, count] of Object.entries(summaryCounts)) {
                const row = summaryBody.insertRow();
                const cellType = row.insertCell();
                const cellCount = row.insertCell();
                cellType.textContent = type;
                cellCount.textContent = count;
            }
        }

        // Display annotated image using the data URL
        resultImg.src = imageUrl;
        resultImg.style.display = "block";

        // Set up download link for the annotated image (using the data URL)
        downloadLink.href = imageUrl;
        // Suggest a filename based on the original test file name
        const safeFilename = testFile.name.replace(/[^a-zA-Z0-9.\-_]/g, '_');
        downloadLink.download = `annotated_${safeFilename}.png`; // Ensure .png
        downloadLink.style.display = "inline-block"; // Make download link visible


    } catch (err) {
        spinner.style.display = "none";
        detectBtn.disabled = false;
        resultSection.style.display = "block"; // Show section to display error
        // Display error message in the UI (add an element with id="error-message" in HTML)
        const errorElement = document.getElementById("error-message") || resultSection; // Fallback
        errorElement.textContent = "Error: " + err.message;
        errorElement.style.color = 'red'; // Make error visible
        console.error("Detection Error:", err);
    }
});