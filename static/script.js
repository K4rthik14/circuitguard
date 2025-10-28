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
const outputDisplay = document.getElementById('output-display');
const defectDetailsDiv = document.getElementById('defect-details');
const noDefectsMessage = document.getElementById('no-defects-message');
const downloadButtonContainer = document.getElementById('download-button-container');
const detectButton = document.getElementById('detect-button');
const detailsPlaceholder = document.getElementById('defect-details-placeholder'); // Get placeholder element


// --- Image Preview Logic ---
function setupImagePreview(fileInput, previewElement) {
    fileInput.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file && file.type.startsWith('image/')) { // Basic type check
            const reader = new FileReader();
            reader.onload = function(e) {
                previewElement.src = e.target.result;
                previewElement.style.display = 'block';
            }
            reader.onerror = function() {
                console.error("Error reading file for preview.");
                previewElement.src = '#';
                previewElement.style.display = 'none';
            }
            reader.readAsDataURL(file);
        } else {
            previewElement.src = '#';
            previewElement.style.display = 'none';
            if (file) { // If a file was selected but wasn't an image
                alert("Please select a valid image file (PNG or JPG).");
                fileInput.value = ""; // Reset input
            }
        }
    });
}
setupImagePreview(templateInput, templatePreview);
setupImagePreview(testInput, testPreview);

// --- Form Submission Logic ---
form.addEventListener('submit', async (event) => {
    event.preventDefault(); // Prevent default page reload

    const formData = new FormData(form);
    const templateFile = templateInput.files[0];
    const testFile = testInput.files[0];

    // --- Frontend Validation ---
    if (!templateFile || !testFile) {
        errorMessage.textContent = '⚠️ Please select both template and test images.';
        resultsSection.style.display = 'block';
        outputDisplay.style.display = 'none';
        spinner.style.display = 'none';
        return;
    }

    // --- UI Update: Show Processing State ---
    detectButton.disabled = true; // Disable button during processing
    detectButton.textContent = 'Processing...'; // Change button text
    spinner.style.display = 'block';
    outputDisplay.style.display = 'none'; // Hide previous results
    errorMessage.textContent = ''; // Clear previous errors
    resultsSection.style.display = 'block'; // Ensure results area is visible
    defectDetailsDiv.innerHTML = ''; // Clear previous defect details
    detailsPlaceholder.style.display = 'none'; // Hide placeholder
    noDefectsMessage.style.display = 'none';
    downloadButtonContainer.innerHTML = ''; // Clear previous download button

    try {
        // --- API Call ---
        // Sends files to the Flask API endpoint '/api/detect'
        console.log("Sending request to /api/detect");
        const response = await fetch('/api/detect', { // Path matches Blueprint route
            method: 'POST',
            body: formData, // FormData handles multipart/form-data encoding
        });
        console.log(`Received response with status: ${response.status}`);

        // --- Handle API Response ---
        spinner.style.display = 'none'; // Hide spinner

        if (!response.ok) {
            // Attempt to parse error JSON from backend, otherwise use status text
            let errorMsg = `Error ${response.status}: ${response.statusText}`;
            try {
                const errorData = await response.json();
                if (errorData && errorData.error) {
                    errorMsg = `Server Error: ${errorData.error}`;
                }
            } catch (e) { console.warn("Could not parse error response as JSON."); }
             // Display specific error from backend or generic one
            errorMessage.textContent = errorMsg;
            console.error('API request failed:', errorMsg);
            throw new Error(errorMsg); // Stop execution
        }

        // --- Process Successful Response (Image Blob) ---
        const imageBlob = await response.blob();
        if (imageBlob.size === 0) {
            console.error("Received empty image blob from server.");
            throw new Error("Received empty image response from server.");
        }
        console.log(`Received image blob, size: ${imageBlob.size} bytes`);
        const imageUrl = URL.createObjectURL(imageBlob);

        // Display the resulting annotated image
        resultImage.src = imageUrl;
        outputDisplay.style.display = 'block'; // Show results area
        resultImage.style.display = 'block';

        // --- Add Download Button ---
        const downloadLink = document.createElement('a');
        downloadLink.href = imageUrl;
        // Suggest a filename, removing spaces and adding prefix
        const safeFilename = testFile.name.replace(/[^a-zA-Z0-9.\-_]/g, '_');
        downloadLink.download = `annotated_${safeFilename}`;
        downloadLink.textContent = '⬇️ Download Annotated Image';
        downloadLink.className = 'download-button'; // Add class for styling
        downloadButtonContainer.appendChild(downloadLink);

        // --- Fetch and Display Defect Details (Placeholder) ---
        // Current API only returns the image. Display placeholder message.
        // If the API were updated to return JSON (e.g., {'imageUrl': '...', 'defects': [...]}),
        // you would parse `response.json()` here instead of `response.blob()`.
        detailsPlaceholder.style.display = 'block'; // Show the placeholder message
        // You would hide this placeholder and populate defectDetailsDiv if the API returned details.
        // Example (if API returned defects):
        // const data = await response.json();
        // displayDefectDetails(data.defects);
        // resultImage.src = data.imageUrl; // If image URL was returned


    } catch (error) {
        // --- Handle Network or Other Errors ---
        spinner.style.display = 'none'; // Ensure spinner is hidden
        errorMessage.textContent = `An error occurred: ${error.message}`;
        console.error('An error occurred during fetch or processing:', error);
        outputDisplay.style.display = 'none'; // Hide results area on error
    } finally {
         detectButton.disabled = false; // Re-enable button
         detectButton.textContent = '🚀 Run Defect Detection & Classification'; // Restore button text
    }
});

// Optional: Function to display defect details if API returns them
/*
function displayDefectDetails(defects) {
    if (!defects || defects.length === 0) {
        noDefectsMessage.style.display = 'block';
        defectDetailsDiv.innerHTML = ''; // Clear any previous details
        detailsPlaceholder.style.display = 'none';
        return;
    }

    noDefectsMessage.style.display = 'none';
    detailsPlaceholder.style.display = 'none';
    defectDetailsDiv.innerHTML = ''; // Clear previous entries

    defects.forEach(defect => {
        const detailElement = document.createElement('div');
        detailElement.className = 'defect-item'; // For styling
        detailElement.innerHTML = `
            <strong>Defect #${defect.id}: ${defect.label}</strong><br>
            <small>Coords (x,y,w,h): (${defect.x}, ${defect.y}, ${defect.w}, ${defect.h})</small>
        `;
        defectDetailsDiv.appendChild(detailElement);
    });
}
*/