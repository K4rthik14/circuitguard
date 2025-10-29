const detectBtn = document.getElementById("detect-button");
const spinner = document.getElementById("spinner");
const resultSection = document.getElementById("results-section");
const resultImg = document.getElementById("result-image");
const downloadLink = document.getElementById("download-link");
const defectCount = document.getElementById("defect-count");
const summaryBody = document.getElementById("summary-body");

// --- Image Preview ---
function setupPreview(inputId, previewId) {
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
            reader.readAsDataURL(file);
        } else {
            preview.style.display = "none";
        }
    });
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

    try {
        const response = await fetch("/api/detect", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) throw new Error("Detection failed!");

        // Expect backend to return JSON with keys: summary, image
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);

        // --- Mock summary (replace when backend returns JSON) ---
        const mockSummary = {
            short: 2,
            pinhole: 3,
            mousebite: 1
        };

        resultSection.style.display = "block";
        spinner.style.display = "none";
        detectBtn.disabled = false;

        // Update defect summary
        const totalDefects = Object.values(mockSummary).reduce((a, b) => a + b, 0);
        defectCount.textContent = totalDefects;

        summaryBody.innerHTML = "";
        for (const [type, count] of Object.entries(mockSummary)) {
            summaryBody.innerHTML += `
                <tr>
                    <td>${type}</td>
                    <td>${count}</td>
                </tr>`;
        }

        // Display annotated image
        resultImg.src = imageUrl;
        resultImg.style.display = "block";
        downloadLink.href = imageUrl;
        downloadLink.style.display = "inline-block";

    } catch (err) {
        spinner.style.display = "none";
        detectBtn.disabled = false;
        alert("Error: " + err.message);
    }
});
