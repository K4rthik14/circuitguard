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
// const defectDetailsDiv = document.getElementById('defect-details'); // No longer needed
const noDefectsMessage = document.getElementById('no-defects-message');
const downloadButtonContainer = document.getElementById('download-button-container');
const detectButton = document.getElementById('detect-button');
// const detailsPlaceholder = document.getElementById('defect-details-placeholder'); // No longer needed

// --- Image Preview Logic ---
function setupImagePreview(fileInput, previewElement) {
    fileInput.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewElement.src = e.target.result;
                previewElement.style.display = 'block'; // Show preview
            }
            reader.onerror = function() {
                console.error("Error reading file for preview.");
                previewElement.src = '#';
                previewElement.style.display = 'none';
            }
            reader.readAsDataURL(file);
        } else {
            previewElement.src = '#';
            previewElement.style.display = 'none'; // Hide preview
            if (file) {
                errorMessage.textContent = 'Please select a valid image file (PNG or JPG).';
                resultsSection.style.display = 'block';
                outputDisplay.style.display = 'none';
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

    const formData = new FormData(); // Only need images now
    const templateFile = templateInput.files[0];
    const testFile = testInput.files[0];

    // --- Frontend Validation ---
    if (!templateFile || !testFile) {
        errorMessage.textContent = '⚠️ Please select both template and test images.';
        successMessage.textContent = '';
        resultsSection.style.display = 'block';
        outputDisplay.style.display = 'none';
        spinner.style.display = 'none';
        noDefectsMessage.style.display = 'none';
        return;
    }
    // Add files to FormData
    formData.append("template_image", templateFile);
    formData.append("test_image", testFile);
    // REMOVED: formData appends for sliders

    // --- UI Update: Show Processing State ---
    detectButton.disabled = true;
    detectButton.textContent = 'Processing...';
    spinner.style.display = 'block';
    outputDisplay.style.display = 'none';
    errorMessage.textContent = '';
    successMessage.textContent = '';
    resultsSection.style.display = 'block';
    // defectDetailsDiv.innerHTML = ''; // No longer needed
    // detailsPlaceholder.style.display = 'none'; // No longer needed
    noDefectsMessage.style.display = 'none'; // Hide initially
    downloadButtonContainer.innerHTML = '';

    try {
        // --- API Call ---
        console.log("Sending request to /api/detect");
        const response = await fetch('/api/detect', {
            method: 'POST',
            body: formData,
        });
        console.log(`Received response with status: ${response.status}`);

        // --- Handle API Response ---
        spinner.style.display = 'none';

        if (!response.ok) {
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
            throw new Error(errorMsg);
        }

        // --- Process Successful Response (Image Blob) ---
        const imageBlob = await response.blob();
        // --- Check for potential "no defects" case ---
        // This is tricky because the backend currently *always* returns an image.
        // A better backend would return JSON indicating if defects were found.
        // For now, we assume defects *were* found if the blob isn't tiny.
        const MIN_IMAGE_SIZE_BYTES = 1000; // Heuristic: assume tiny response means no defects drawn
        if (imageBlob.size < MIN_IMAGE_SIZE_BYTES) {
             console.log("Received small image blob, assuming no defects found.");
             successMessage.textContent = '✅ Analysis Complete!';
             noDefectsMessage.style.display = 'block'; // Show "No defects" message
             outputDisplay.style.display = 'block'; // Show results area, but not image/download
             resultImage.style.display = 'none';
        } else {
             console.log(`Received image blob, size: ${imageBlob.size} bytes`);
             const imageUrl = URL.createObjectURL(imageBlob);

             // Display the resulting annotated image
             successMessage.textContent = '✅ Analysis Complete! Defects Found:'; // Updated message
             resultImage.src = imageUrl;
             outputDisplay.style.display = 'block'; // Show results area
             resultImage.style.display = 'block'; // Make image visible
             noDefectsMessage.style.display = 'none'; // Hide "No defects" message

             // --- Add Download Button ---
             const downloadLink = document.createElement('a');
             downloadLink.href = imageUrl;
             const safeFilename = testFile.name.replace(/[^a-zA-Z0-9.\-_]/g, '_');
             downloadLink.download = `annotated_${safeFilename}.png`;
             downloadLink.textContent = '⬇️ Download Annotated Image';
             downloadLink.className = 'download-button';
             downloadButtonContainer.appendChild(downloadLink);
         }


    } catch (error) {
        // --- Handle Errors ---
        spinner.style.display = 'none';
        errorMessage.textContent = `An error occurred: ${error.message}`;
        console.error('An error occurred during fetch or processing:', error);
        outputDisplay.style.display = 'none';
        successMessage.textContent = '';
    } finally {
         detectButton.disabled = false;
         detectButton.textContent = '🚀 Run Defect Detection & Classification';
    }
});