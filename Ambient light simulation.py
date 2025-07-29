import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rawpy
from scipy.interpolate import Rbf, interp1d
import matplotlib.colors as mcolors  # For RGB <-> HSV conversions

# -----------------------------
# Utility Functions (same as before)
# -----------------------------

def load_raw_image(image_path):
    """Load and postprocess a raw image with a fixed brightness scaling."""
    with rawpy.imread(image_path) as raw:
        rgb = raw.postprocess(
            no_auto_bright=True,
            gamma=(1, 1),
            use_camera_wb=False,
            output_bps=16,
            bright=2.0  # Adjusted brightness
        )
    return rgb.astype(np.float32) / 65535.0

def extract_avg_colors(image, roi_file):
    """Extract average color values from each ROI defined in the ROI JSON file."""
    if not os.path.exists(roi_file):
        print(f"ROI file not found: {roi_file}")
        return None
    with open(roi_file, 'r') as f:
        rois = json.load(f).get("rois", [])
    if len(rois) < 3:
        print(f"Warning: {roi_file} has {len(rois)} ROIs. Expected at least 3.")
        return None
    avg_colors = []
    for roi in rois:
        x, y, w, h = roi
        roi_pixels = image[y:y + h, x:x + w, :]
        avg_color = roi_pixels.mean(axis=(0, 1))
        avg_colors.append(avg_color)
    return np.array(avg_colors)

def interpolate_roi_colors(roi_colors, target_depths):
    """
    Given a dictionary of ROI colors keyed by depth, return a new dictionary mapping each
    target depth to interpolated ROI colors.
    """
    known_depths = np.array(sorted(roi_colors.keys()))
    interpolated = {}
    for depth in target_depths:
        if depth in roi_colors:
            interpolated[depth] = roi_colors[depth]
        else:
            interp_list = []
            for i in range(len(roi_colors[known_depths[0]])):
                all_colors = np.array([roi_colors[d][i] for d in known_depths])
                interp_func = interp1d(known_depths, all_colors, axis=0, kind='linear', fill_value="extrapolate")
                interp_list.append(interp_func(depth))
            interpolated[depth] = np.array(interp_list)
    return interpolated

def fit_color_transform(source_colors, target_colors):
    """
    Fit an RBF transformation that maps from source ROI colors to target ROI colors.
    Uses the 'multiquadric' kernel with smoothing.
    """
    source_colors = np.array(source_colors)
    target_colors = np.array(target_colors)
    rbf_R = Rbf(source_colors[:, 0], source_colors[:, 1], source_colors[:, 2],
                target_colors[:, 0], function='multiquadric', smooth=0.1)
    rbf_G = Rbf(source_colors[:, 0], source_colors[:, 1], source_colors[:, 2],
                target_colors[:, 1], function='multiquadric', smooth=0.1)
    rbf_B = Rbf(source_colors[:, 0], source_colors[:, 1], source_colors[:, 2],
                target_colors[:, 2], function='multiquadric', smooth=0.1)
    return rbf_R, rbf_G, rbf_B

def apply_color_transform(image, rbf_R, rbf_G, rbf_B):
    """
    Apply the color transformation to an image. The raw RBF outputs are first computed,
    then channel-wise clipping is performed. For pixels whose raw outputs exceed the
    [0,1] range, a correction based on the original pixel (near black/white) is applied.
    """
    h, w, c = image.shape
    pixels = image.reshape(-1, 3).astype(np.float32)
    raw_R = rbf_R(pixels[:, 0], pixels[:, 1], pixels[:, 2])
    raw_G = rbf_G(pixels[:, 0], pixels[:, 1], pixels[:, 2])
    raw_B = rbf_B(pixels[:, 0], pixels[:, 1], pixels[:, 2])
    raw_transformed = np.stack([raw_R, raw_G, raw_B], axis=1)
    raw_transformed = np.nan_to_num(raw_transformed, nan=0.5, posinf=1, neginf=0)
    clipped = np.clip(raw_transformed, 0, 1)

    epsilon = 0.05
    orig_near_black = np.all(pixels < epsilon, axis=1)
    orig_near_white = np.all(pixels > 1 - epsilon, axis=1)
    result = clipped.copy()
    for i in range(raw_transformed.shape[0]):
        if orig_near_black[i] and np.any(raw_transformed[i] < 0):
            result[i] = [0, 0, 0]
        elif orig_near_white[i] and np.any(raw_transformed[i] > 1):
            result[i] = [1, 1, 1]
    return result.reshape(h, w, 3)

def adjust_brightness(image, reference_image):
    """
    Adjusts the brightness of the transformed image based on the ratio of mean brightness
    between the reference image and the transformed image, scaled by 0.5.
    """
    ref_mean = np.mean(reference_image)
    img_mean = np.mean(image)
    if img_mean > 0:
        scale_factor = (ref_mean / img_mean) * 0.5
        scaled_image = np.clip(image * scale_factor, 0, 1)
        return scaled_image
    else:
        return image

# -----------------------------
# Hue Correction Utility Functions
# -----------------------------

def compute_hue_shift(day4_color, day1_color):
    """
    Compute the hue difference between two RGB colors (in [0,1]) by converting them to HSV.
    Returns a hue shift (in the range [-0.5, 0.5]) that, when added to the day1 hue,
    yields the day4 hue.
    """
    hsv_day4 = mcolors.rgb_to_hsv(day4_color.reshape(1, 3))[0]
    hsv_day1 = mcolors.rgb_to_hsv(day1_color.reshape(1, 3))[0]
    hue_shift = hsv_day4[0] - hsv_day1[0]
    if hue_shift > 0.5:
        hue_shift -= 1.0
    elif hue_shift < -0.5:
        hue_shift += 1.0
    return hue_shift

def shift_hue(image, hue_shift):
    """
    Shift the hue of an RGB image (values in [0,1]) by hue_shift.
    """
    hsv = mcolors.rgb_to_hsv(image)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 1.0
    return mcolors.hsv_to_rgb(hsv)

# -----------------------------
# Batch Simulation Code with Hue Correction and Dual Output Formats
# -----------------------------

def simulate_batch():
    # -- Calibration Data (Day 1 Color Card Images) --
    calib_folder = r"Y:\1. Thomas_2024\10_code\Philippines\D1_Surface_images_versus_depth"
    calib_excel_file = os.path.join(calib_folder, "D1_Surface_images_versus_depth.xlsx")
    calib_df = pd.read_excel(calib_excel_file)

    # Build dictionary of ROI colors keyed by depth (from calibration images).
    calib_roi_colors = {}
    for _, row in calib_df.iterrows():
        depth = row["Depth"]
        filename = row["Image Path"]  # assuming column name matches previous format
        file_path = os.path.join(calib_folder, filename)
        roi_file = os.path.join(calib_folder, f"{os.path.splitext(filename)[0]}_rois.json")
        if depth not in calib_roi_colors:
            if os.path.exists(file_path) and os.path.exists(roi_file):
                image = load_raw_image(file_path)
                roi = extract_avg_colors(image, roi_file)
                if roi is not None:
                    calib_roi_colors[depth] = roi
            else:
                print(f"Calibration file or ROI missing for {filename}.")

    if not calib_roi_colors:
        raise ValueError("No ROI colors extracted from calibration images!")

    calib_depths = sorted(calib_roi_colors.keys())
    ref_depth = min(calib_depths)
    source_roi = calib_roi_colors[ref_depth]

    # --- Hue Correction: Compute Hue Shift Using Day 4 Color Card ---
    day4_colorcard_path = r"Y:\1. Thomas_2024\10_code\Philippines\D4_Colorcard\20241114_Proj_Cuttlefish_D4_DSC0880.ARW"
    day4_roi_json = r"Y:\1. Thomas_2024\10_code\Philippines\D4_Colorcard\20241114_Proj_Cuttlefish_D4_DSC0880_rois.json"
    day4_excel = r"Y:\1. Thomas_2024\10_code\Philippines\D4_Colorcard\D4_colorcard.xlsx"

    day4_image = load_raw_image(day4_colorcard_path)
    day4_avg_colors = extract_avg_colors(day4_image, day4_roi_json)
    if day4_avg_colors is None or len(day4_avg_colors) < 3:
        raise ValueError("Insufficient ROIs in day 4 color card.")
    day4_roi3 = day4_avg_colors[2]  # ROI #3 (zero-indexed)

    day4_df = pd.read_excel(day4_excel)
    day4_filename = os.path.basename(day4_colorcard_path)
    depth_row = day4_df[day4_df["Image Path"].str.contains(day4_filename, case=False)]
    if depth_row.empty:
        raise ValueError("No matching depth found for day 4 color card in the Excel file.")
    day4_depth = depth_row.iloc[0]["Depth"]

    # Use day 1 calibration data to predict ROI colors at day 4 depth.
    day1_predicted_rois = interpolate_roi_colors(calib_roi_colors, [day4_depth])
    day1_roi3 = day1_predicted_rois[day4_depth][2]

    hue_shift = compute_hue_shift(day4_roi3, day1_roi3)
    print(f"Computed hue shift (day 4 - day 1): {hue_shift:.3f}")

    # -- Simulation Data (Target Images) --
    sim_folder = r"Y:\1. Thomas_2024\10_code\Philippines\Depth sim batch"
    sim_excel_file = os.path.join(sim_folder, "Filename and depth.xlsx")
    output_folder = os.path.join(sim_folder, "Simulated_output")
    os.makedirs(output_folder, exist_ok=True)
    sim_df = pd.read_excel(sim_excel_file)
    sim_depths = sim_df["Depth"].unique()
    interp_roi = interpolate_roi_colors(calib_roi_colors, sim_depths)

    # Process each simulation target image.
    for _, row in sim_df.iterrows():
        depth = row["Depth"]
        filename = row["Filename"]
        file_path = os.path.join(sim_folder, filename)
        if not os.path.exists(file_path):
            print(f"Simulation file not found: {file_path}")
            continue
        print(f"Processing {filename} at depth {depth}m...")
        image = load_raw_image(file_path)
        if depth == ref_depth:
            simulated_image = image
        else:
            target_roi = interp_roi[depth]
            rbf_R, rbf_G, rbf_B = fit_color_transform(source_roi, target_roi)
            simulated_image = apply_color_transform(image, rbf_R, rbf_G, rbf_B)
            simulated_image = adjust_brightness(simulated_image, image)
        simulated_image = shift_hue(simulated_image, hue_shift)

        # --- Create Figure at Original Resolution and Add Text ---
        height, width, _ = simulated_image.shape
        dpi = 100  # you can adjust this if needed
        fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
        ax.imshow((simulated_image * 255).astype(np.uint8))
        ax.axis("off")
        # Compute font size (in points) so that text height is ~10% of the image height:
        font_size = (0.07 * height * 72) / dpi
        ax.text(0.98, 1.1, f"Depth: {depth}m",
                transform=ax.transAxes,
                color='black',
                fontsize=font_size,
                ha='right',
                va='top')

        # --- Save as TIFF ---
        out_filename_tiff = os.path.splitext(filename)[0] + "_simulated.tiff"
        out_path_tiff = os.path.join(output_folder, out_filename_tiff)
        plt.savefig(out_path_tiff, dpi=dpi, pad_inches=0)
        print(f"Saved simulated TIFF to {out_path_tiff}")

        # --- Save as SVG ---
        out_filename_svg = os.path.splitext(filename)[0] + "_simulated.svg"
        out_path_svg = os.path.join(output_folder, out_filename_svg)
        plt.savefig(out_path_svg, format='svg', dpi=dpi, pad_inches=0)
        print(f"Saved simulated SVG to {out_path_svg}")

        plt.close(fig)

if __name__ == "__main__":
    simulate_batch()
