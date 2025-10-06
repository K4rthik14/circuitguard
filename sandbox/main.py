# main.py
import os
import cv2
import matplotlib.pyplot as plt
from preprocessing import process_pcb_pair, extract_rois

def run_and_visualize(base_name, dataset_path='PCB_DATASET', output_path='outputs'):
    """
    Runs the full pipeline for a given image base name and saves/shows the results.
    """
    # --- Setup Paths ---
    image_dir = os.path.join(dataset_path, 'images')
    template_path = os.path.join(image_dir, f"{base_name}_temp.jpg")
    test_path = os.path.join(image_dir, f"{base_name}_test.jpg")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    try:
        # --- Run Processing ---
        results = process_pcb_pair(template_path, test_path)

        # --- Extract ROIs and Draw Bounding Boxes ---
        final_image, cropped_rois = extract_rois(results["test"], results["contours"])

        # --- Visualization ---
        print(f"Found {len(cropped_rois)} potential defects for image {base_name}.")

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"PCB Defect Detection Pipeline for: {base_name}", fontsize=16)

        # Row 1: Core processing
        axes[0, 0].imshow(results["template"], cmap='gray')
        axes[0, 0].set_title("1. Template Image")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(results["test"], cmap='gray')
        axes[0, 1].set_title("2. Test Image")
        axes[0, 1].axis('off')

        axes[0, 2].imshow(results["difference"], cmap='gray')
        axes[0, 2].set_title("3. Image Subtraction")
        axes[0, 2].axis('off')

        # Row 2: Defect isolation
        axes[1, 0].imshow(results["threshold"], cmap='gray')
        axes[1, 0].set_title("4. Otsu's Thresholding")
        axes[1, 0].axis('off')

        axes[1, 1].imshow(results["cleaned_mask"], cmap='gray')
        axes[1, 1].set_title("5. Cleaned Defect Mask")
        axes[1, 1].axis('off')

        axes[1, 2].imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title("6. Final Defects Detected")
        axes[1, 2].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the final visualization [cite: 44]
        output_filename = os.path.join(output_path, f"{base_name}_pipeline_visualization.png")
        plt.savefig(output_filename)
        print(f"Saved pipeline visualization to {output_filename}")
        
        plt.show()

        # --- Save Cropped Defects (ROIs) [cite: 55] ---
        if cropped_rois:
            print("Saving cropped defect samples...")
            roi_dir = os.path.join(output_path, base_name + "_rois")
            if not os.path.exists(roi_dir):
                os.makedirs(roi_dir)
            for i, roi in enumerate(cropped_rois):
                roi_filename = os.path.join(roi_dir, f"defect_{i+1}.png")
                cv2.imwrite(roi_filename, roi)
            print(f"Saved {len(cropped_rois)} ROIs to {roi_dir}/")


    except FileNotFoundError:
        print(f"Error: Could not find images for base name '{base_name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    # You can change this name to test different PCB images from the dataset
    SAMPLE_IMAGE_BASENAME = '0001' 
    run_and_visualize(SAMPLE_IMAGE_BASENAME)