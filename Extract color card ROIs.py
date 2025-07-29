import os
import cv2
import rawpy
import numpy as np
import pandas as pd
import json

def load_metadata(file_path):
    """Load metadata from an Excel file and clean it."""
    try:
        metadata = pd.read_excel(file_path)
        metadata.columns = metadata.columns.str.strip()  # Normalize column names
        metadata = metadata.dropna()  # Remove rows with NaN values
        print(f"Columns in metadata: {metadata.columns.tolist()}\n")  # Debugging
        return metadata
    except Exception as e:
        print(f"Failed to load metadata: {e}")
        return None

def load_raw_image_as_linear_rgb(image_path):
    """Load a Sony ARW raw image using rawpy and return a linear RGB image."""
    try:
        with rawpy.imread(image_path) as raw:
            rgb = raw.postprocess(
                no_auto_bright=False,  # Apply auto brightness for better visualization
                gamma=(1, 1),  # Linear gamma
                use_camera_wb=True,
                output_bps=16
            )
        # Normalize to [0, 1] range
        rgb_linear = rgb.astype(np.float32) / 65535.0
        print(f"Loaded raw image: {image_path}")
        return rgb_linear
    except Exception as e:
        print(f"Failed to load raw image {image_path}: {e}")
        return None

def select_and_save_rois(image, output_path, window_name="Select ROIs"):
    """Display the image, allow ROI selection, and save the selected ROIs to a JSON file."""
    try:
        # Brighten the image for better display
        image_brightened = np.clip(image * 1.5, 0, 1)  # Increase brightness for display

        # Resize image for display
        h, w = image_brightened.shape[:2]
        max_dim = 800  # Scale to fit within 800 pixels
        scale_factor = min(max_dim / h, max_dim / w, 1.0)
        display_image = cv2.resize(image_brightened, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_AREA)

        # Convert RGB to BGR for OpenCV display
        display_image_bgr = cv2.cvtColor((display_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Select ROIs
        print("Select ROIs and press ENTER when done, or ESC to skip.")
        rois_resized = cv2.selectROIs(window_name, display_image_bgr)
        cv2.destroyAllWindows()

        # If no ROIs were selected, return
        if len(rois_resized) == 0:
            print("No ROIs selected. Skipping this image.")
            return None

        # Scale ROIs back to original image size
        rois = []
        for roi in rois_resized:
            x, y, rw, rh = roi
            rois.append([
                int(x / scale_factor),
                int(y / scale_factor),
                int(rw / scale_factor),
                int(rh / scale_factor)
            ])

        # Save ROIs to JSON
        roi_data = {"rois": rois}
        with open(output_path, 'w') as json_file:
            json.dump(roi_data, json_file, indent=4)
        print(f"ROIs saved to {output_path}")

        return len(rois)

    except Exception as e:
        print(f"Error during ROI selection and saving: {e}")
        return None

def check_consistent_rois(metadata):
    """Check if all images have the same number of ROIs saved."""
    try:
        roi_counts = []
        for index, row in metadata.iterrows():
            image_path = row['Image Path']
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            json_path = os.path.join(os.path.dirname(image_path), f"{base_name}_rois.json")

            if not os.path.exists(json_path):
                print(f"JSON file missing for image: {image_path}")
                continue

            with open(json_path, 'r') as json_file:
                roi_data = json.load(json_file)
                roi_counts.append(len(roi_data.get('rois', [])))

        if len(set(roi_counts)) == 1:
            print(f"All images have the same number of ROIs: {roi_counts[0]}.")
        else:
            print(f"Inconsistent ROI counts across images: {roi_counts}")
            redo_inconsistent_rois(metadata, roi_counts)

    except Exception as e:
        print(f"Error during consistency check: {e}")

def redo_inconsistent_rois(metadata, roi_counts):
    """Redo ROI selection for images with inconsistent ROI counts."""
    try:
        expected_count = max(set(roi_counts), key=roi_counts.count)
        for index, row in metadata.iterrows():
            image_path = row['Image Path']
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            json_path = os.path.join(os.path.dirname(image_path), f"{base_name}_rois.json")

            if not os.path.exists(json_path):
                print(f"Skipping {image_path}, no JSON file found.")
                continue

            with open(json_path, 'r') as json_file:
                roi_data = json.load(json_file)
                if len(roi_data.get('rois', [])) != expected_count:
                    print(f"Redoing ROI selection for {image_path}.")
                    image = load_raw_image_as_linear_rgb(image_path)
                    if image is None:
                        print(f"Could not reload image: {image_path}. Skipping.")
                        continue
                    select_and_save_rois(image, json_path)

    except Exception as e:
        print(f"Error during ROI redo: {e}")

def main():
    # File path to metadata Excel file
    metadata_file = r"Y:\1. Thomas_2024\10_code\Philippines\D4_Colorcard\D4_colorcard.xlsx"

    # Load metadata
    metadata = load_metadata(metadata_file)
    if metadata is None:
        print("No metadata loaded. Exiting.")
        return

    # Print all image paths in the spreadsheet
    print("Image paths found in the spreadsheet:")
    for index, row in metadata.iterrows():
        try:
            image_path = row['Image Path']
            print(image_path)
        except KeyError as e:
            print(f"Error accessing 'Image Path' column: {e}")
        except Exception as e:
            print(f"Unexpected error accessing row {index}: {e}")

    # Iterate through each image in the metadata
    for index, row in metadata.iterrows():
        try:
            image_path = row['Image Path']  # Adjusted to match column name
            print(f"Processing image: {image_path}")

            # Load raw image
            image = load_raw_image_as_linear_rgb(image_path)
            if image is None:
                print(f"Skipping image due to load failure: {image_path}")
                continue

            # Output path for JSON
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(os.path.dirname(image_path), f"{base_name}_rois.json")

            # Select ROIs and save
            select_and_save_rois(image, output_path)

        except KeyError as e:
            print(f"Missing column in metadata or data issue: {e}")
        except Exception as e:
            print(f"Unexpected error processing row {index}: {e}")

    # Check consistency of ROI counts
    check_consistent_rois(metadata)

    print("Finished processing all images.")

if __name__ == "__main__":
    main()
