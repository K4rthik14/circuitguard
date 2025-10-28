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

// --- Image Preview Logic ---
function setupImagePreview(fileInput, previewElement) {
    fileInput.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewElement.src = e.target.result;
                previewElement.style.display = 'block';
            }
            reader.readAsDataURL(file);
        } else {
            previewElement.src = '#';
            previewElement.style.display = 'none';
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
    spinner.style.display = 'block';
    outputDisplay.style.display = 'none'; // Hide previous results
    errorMessage.textContent = ''; // Clear previous errors
    resultsSection.style.display = 'block'; // Ensure results area is visible
    defectDetailsDiv.innerHTML = ''; // Clear previous defect details
    noDefectsMessage.style.display = 'none';
    downloadButtonContainer.innerHTML = ''; // Clear previous download button

    try {
        // --- API Call ---
        const response = await fetch('/api/detect', {
            method: 'POST',
            body: formData,
        });

        // --- Handle API Response ---
        spinner.style.display = 'none'; // Hide spinner

        if (!response.ok) {
            // Attempt to parse error JSON
            let errorMsg = `Error ${response.status}: ${response.statusText}`;
            try {
                const errorData = await response.json();
                if (errorData && errorData.error) {
                    errorMsg = `Server Error: ${errorData.error}`;
                }
            } catch (e) { console.warn("Could not parse error response as JSON."); }
            throw new Error(errorMsg); // Throw error to be caught below
        }

        // --- Process Successful Response (Image Blob) ---
        const imageBlob = await response.blob();
        if (imageBlob.size === 0) {
            throw new Error("Received empty image response from server.");
        }
        const imageUrl = URL.createObjectURL(imageBlob);

        // Display the resulting annotated image
        resultImage.src = imageUrl;
        outputDisplay.style.display = 'block'; // Show results area
        resultImage.style.display = 'block';

        // --- Add Download Button ---
        const downloadLink = document.createElement('a');
        downloadLink.href = imageUrl;
        downloadLink.download = `annotated_${testFile.name}`; // Suggest filename
        downloadLink.textContent = '⬇️ Download Annotated Image';
        downloadLink.className = 'download-button'; // Add class for styling
        downloadButtonContainer.appendChild(downloadLink);

        // --- Fetch and Display Defect Details (Requires backend modification) ---
        // For now, we assume no defects if the image is small, or display a generic message.
        // A better approach: API returns JSON with image URL *and* defect list.
        // Let's simulate based on image size (placeholder logic)
        // A proper solution requires the backend API to return defect details (e.g., in headers or a JSON response)
        // Since the current API returns only the image, we can't display details accurately here.
        // We'll add a placeholder message.
         defectDetailsDiv.innerHTML = '<p><em>Defect details require API modification to return data alongside the image.</em></p>';
         // If you modify the API later to return JSON, you'd parse it here and build the defect list.


    } catch (error) {
        // --- Handle Errors ---
        spinner.style.display = 'none';
        errorMessage.textContent = `An error occurred: ${error.message}`;
        console.error('Error during fetch:', error);
        outputDisplay.style.display = 'none'; // Hide results area on error
    } finally {
         detectButton.disabled = false; // Re-enable button
    }
});