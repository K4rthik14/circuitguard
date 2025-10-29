const detectBtn = document.getElementById("detect-button");
const spinner = document.getElementById("spinner");
const resultSection = document.getElementById("results-section");
const resultImg = document.getElementById("result-image");
const downloadLink = document.getElementById("download-link");
const defectCount = document.getElementById("defect-count");
const defectTableBody = document.getElementById("defect-table-body");

const diffRange = document.getElementById("diff-threshold");
const roiRange = document.getElementById("roi-area");
const confRange = document.getElementById("conf-threshold");

document.getElementById("diff-value").textContent = diffRange.value;
document.getElementById("roi-value").textContent = roiRange.value;
document.getElementById("conf-value").textContent = confRange.value;

[diffRange, roiRange, confRange].forEach(slider => {
    slider.addEventListener("input", () => {
        document.getElementById(slider.id.replace("-", "_") + "value");
    });
});

detectBtn.addEventListener("click", async () => {
    const templateFile = document.getElementById("template_image").files[0];
    const testFile = document.getElementById("test_image").files[0];
    if (!templateFile || !testFile) {
        alert("Please upload both Template and Test images!");
        return;
    }

    const formData = new FormData();
    formData.append("template_image", templateFile);
    formData.append("test_image", testFile);
    formData.append("diff_threshold", diffRange.value);
    formData.append("roi_area", roiRange.value);
    formData.append("conf_threshold", confRange.value);

    spinner.style.display = "block";
    detectBtn.disabled = true;

    try {
        const response = await fetch("/api/detect", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) throw new Error("Detection failed!");

        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);

        spinner.style.display = "none";
        detectBtn.disabled = false;
        resultSection.style.display = "block";
        resultImg.src = imageUrl;
        resultImg.style.display = "block";
        downloadLink.href = imageUrl;
        downloadLink.style.display = "inline-block";

        // Mock defect summary (replace with backend data later)
        const randomDefects = Math.floor(Math.random() * 5) + 1;
        defectCount.textContent = `Detected Defects: ${randomDefects}`;

        defectTableBody.innerHTML = "";
        for (let i = 1; i <= randomDefects; i++) {
            const conf = (Math.random() * 0.3 + 0.7).toFixed(2);
            defectTableBody.innerHTML += `
                <tr>
                    <td>${i}</td>
                    <td>ROI_${100 + i}</td>
                    <td>${conf}</td>
                </tr>`;
        }
    } catch (err) {
        spinner.style.display = "none";
        detectBtn.disabled = false;
        alert("Error: " + err.message);
    }
});
