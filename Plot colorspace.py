import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rawpy
import colorsys

from skimage.color import rgb2lab, lab2rgb
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.path import Path
import matplotlib.colors as mcolors

# -------------------- IMAGE PROCESSING FUNCTIONS -------------------- #

def load_raw_image_as_linear_rgb(image_path):
    with rawpy.imread(image_path) as raw:
        rgb = raw.postprocess(
            no_auto_bright=True, gamma=(1, 1), use_camera_wb=False, output_bps=16
        )
    return rgb.astype(np.float32) / 65535.0


def extract_avg_colors_from_rois(image, rois):
    avg_colors = []
    for roi in rois:
        x, y, w, h = roi
        roi_pixels = image[y:y + h, x:x + w, :]
        avg_color = roi_pixels.mean(axis=(0, 1))
        avg_colors.append(avg_color)
    return np.array(avg_colors)


def normalize_brightness(colors, reference_index=2, target_brightness=0.5):
    ref_brightness = np.mean(colors[reference_index])
    scale_factor = target_brightness / ref_brightness
    return np.clip(colors * scale_factor, 0, 1)


def rgb_to_lch(rgb):
    lab = rgb2lab(rgb.reshape(1, 1, 3)).reshape(-1, 3)
    L, a, b = lab[0]
    C = np.sqrt(a ** 2 + b ** 2)
    h = (np.degrees(np.arctan2(b, a)) + 360) % 360
    return L, C, h


# -------------------- PLOTTING HELPERS -------------------- #

def generate_background_colors(resolution=100, chroma_threshold=10):
    """
    Generate a background in hue–chroma space.
    For chroma values below `chroma_threshold`, the output is forced to white.
    Otherwise, a LAB value is computed with a lightness that linearly decreases from 100 (chroma=0)
    to 70 (chroma=100), then converted to RGB.
    """
    hues, chromas = np.meshgrid(np.linspace(0, 360, resolution),
                                np.linspace(0, 100, resolution))
    l_values = 100 - 0.3 * chromas  # Linear mapping: 0->100, 100->70.
    background_colors = []
    for h, c, l in zip(hues.flatten(), chromas.flatten(), l_values.flatten()):
        if c < chroma_threshold:
            rgb = np.array([1, 1, 1])
        else:
            a = c * np.cos(np.radians(h))
            b = c * np.sin(np.radians(h))
            lab = np.array([l, a, b]).reshape(1, 1, 3)
            try:
                rgb = lab2rgb(lab).flatten()
            except ValueError:
                rgb = np.array([1, 1, 1])
        background_colors.append(rgb)
    background_colors = np.clip(
        np.array(background_colors).reshape(resolution, resolution, 3), 0, 1
    )
    return hues, chromas, background_colors


def ray_polygon_intersection_chroma(polygon, hue_angle, tol=1e-6):
    """
    Given a closed polygon (N+1 x 2, where last point == first point) and a ray
    (from the origin at angle hue_angle in degrees), compute the intersection distance
    (i.e. chroma) along the ray.
    Returns the maximum distance (chroma) inside the polygon or None if no intersection.
    """
    theta = np.radians(hue_angle)
    r = np.array([np.cos(theta), np.sin(theta)])  # unit ray direction
    intersections = []

    for i in range(len(polygon) - 1):
        P1 = polygon[i]
        P2 = polygon[i + 1]
        d = P2 - P1  # edge direction

        cross_rd = r[0] * d[1] - r[1] * d[0]
        if np.abs(cross_rd) < tol:
            continue  # ray and segment nearly parallel

        cross_P1_d = P1[0] * d[1] - P1[1] * d[0]
        cross_P1_r = P1[0] * r[1] - P1[1] * r[0]
        t = cross_P1_d / cross_rd
        u = cross_P1_r / cross_rd

        if t >= 0 and 0 <= u <= 1:
            intersections.append(t)

    if not intersections:
        return None
    return max(intersections)


# -------------------- PLOTTING FUNCTIONS -------------------- #

def plot_hue_chroma_with_convex_hull(hue_chroma_list, depths):
    resolution = 100
    # Generate the background, forcing very low chroma (below 10) to appear white.
    hues, chromas, background_colors = generate_background_colors(resolution, chroma_threshold=10)

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # Create grid edges for pcolormesh in polar coordinates.
    theta_edges = np.linspace(0, 2 * np.pi, resolution + 1)
    r_edges = np.linspace(0, 100, resolution + 1)
    theta_edges_grid, r_edges_grid = np.meshgrid(theta_edges, r_edges)
    ax.pcolormesh(theta_edges_grid, r_edges_grid, background_colors, shading='auto')

    min_depth = min(depths)
    max_depth = max(depths)

    # Plot each convex hull for the hue–chroma data.
    for (hues_pts, chromas_pts), depth in zip(hue_chroma_list, depths):
        if len(hues_pts) < 3:
            continue
        # Convert measured (hue, chroma) data to Cartesian coordinates.
        pts = np.column_stack((chromas_pts * np.cos(np.radians(hues_pts)),
                                chromas_pts * np.sin(np.radians(hues_pts))))
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
        hull_pts = np.vstack((hull_pts, hull_pts[0]))  # Close the polygon

        # Convert hull points back to polar coordinates.
        hull_angles = (np.degrees(np.arctan2(hull_pts[:, 1], hull_pts[:, 0])) + 360) % 360
        hull_radii = np.sqrt(hull_pts[:, 0] ** 2 + hull_pts[:, 1] ** 2)

        # Compute a grey tone based on depth (shallow: white, deep: darker).
        grayscale = 1 - ((depth - min_depth) / (max_depth - min_depth))
        ax.plot(np.radians(hull_angles), hull_radii, color=(grayscale, grayscale, grayscale), linewidth=2)

    # Restore the original polar plot settings.
    ax.set_ylim(0, 100)
    ax.set_yticks(np.linspace(0, 100, 11))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    plt.title("Hue–Chroma Color Spectrum by Depth", fontsize=14)

    # Add a colorbar key mapping the grey tone to depth.
    norm = plt.Normalize(vmin=min_depth, vmax=max_depth)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap='gray_r')
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax, pad=0.1, orientation='vertical')
    cbar.set_label("Depth (m)")
    # Invert the colorbar's y-axis so that 0 depth appears at the top and deeper depths at the bottom.
    cbar.ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig("Y:/1. Thomas_2024/10_code/Philippines/Plots/Hue-Chroma_Color_Spectrum_with_Depth_Convex_Hulls.svg", format="svg")
    plt.show()


def plot_hue_chroma_3d(hue_chroma_list, depths):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    resolution = 100
    hues, chromas, background_colors = generate_background_colors(resolution)
    theta = np.radians(hues)
    r = chromas
    x_base = r * np.cos(theta)
    y_base = r * np.sin(theta)
    z_base = np.zeros_like(x_base)
    ax.plot_surface(x_base, y_base, z_base, facecolors=background_colors, rstride=1, cstride=1, shade=False)

    min_depth, max_depth = min(depths), max(depths)
    depth_levels = np.interp(depths, (min_depth, max_depth), (0, 1))

    for (hues_pts, chromas_pts), depth_norm in zip(hue_chroma_list, depth_levels):
        if len(hues_pts) > 2:
            cartesian_points = np.column_stack(
                (chromas_pts * np.cos(np.radians(hues_pts)),
                 chromas_pts * np.sin(np.radians(hues_pts)))
            )
            hull = ConvexHull(cartesian_points)
            hull_points = cartesian_points[hull.vertices]
            hull_points = np.vstack((hull_points, hull_points[0]))
            hull_points_3d = np.column_stack((hull_points, np.full(len(hull_points), depth_norm)))
            poly = Poly3DCollection([hull_points_3d], alpha=0.5)
            poly.set_facecolor((1 - depth_norm, 1 - depth_norm, 1 - depth_norm, 0.6))
            ax.add_collection3d(poly)

    ax.set_xlabel("Hue Component (X)")
    ax.set_ylabel("Chroma Component (Y)")
    ax.set_zlabel("Normalized Depth")
    ax.set_title("3D Hue-Chroma Space with Depth Separation")
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])
    ax.set_zlim([0, 1])
    plt.show()


def plot_chroma_vs_depth_by_hue(hue_chroma_list, depths, hue_angles=None):
    if hue_angles is None:
        hue_angles = np.linspace(0, 360, 60, endpoint=False)
    hue_angles = np.array(hue_angles)
    hue_data = {h: [] for h in hue_angles}

    for (hues_pts, chromas_pts), depth in zip(hue_chroma_list, depths):
        x = chromas_pts * np.cos(np.radians(hues_pts))
        y = chromas_pts * np.sin(np.radians(hues_pts))
        points = np.column_stack((x, y))
        if len(points) < 3:
            continue
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        hull_polygon = np.vstack([hull_points, hull_points[0]])
        for h in hue_angles:
            max_chroma = ray_polygon_intersection_chroma(hull_polygon, h)
            if max_chroma is None:
                max_chroma = 0
            hue_data[h].append((depth, max_chroma))

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_facecolor('white')

    for h in hue_angles:
        if not hue_data[h]:
            continue
        sorted_data = sorted(hue_data[h], key=lambda x: x[0])
        d_vals, c_vals = zip(*sorted_data)
        rgb_color = plt.cm.hsv(h / 360.0)
        plt.plot(d_vals, c_vals, color=rgb_color, label=f'{int(h)}°')

    plt.xlabel('Depth', color='black')
    plt.ylabel('Maximum Chroma', color='black')
    plt.title('Chroma vs Depth for Selected Hues', color='black')
    legend = plt.legend(title='Hue Angle', facecolor='white', edgecolor='white')
    plt.setp(legend.get_texts(), color='black')
    plt.setp(legend.get_title(), color='black')
    plt.tick_params(colors='black')
    plt.tight_layout()
    #plt.savefig("Y:/1. Thomas_2024/10_code/Philippines/Plots/Chroma vs Depth and Cuttlefish Sightings.svg", format="svg")
    plt.show()


def plot_chroma_and_cuttlefish_histogram(hue_chroma_list, depths, cuttlefish_excel_path, n_hue_samples=360, fixed_L=70):
    from skimage.color import lab2rgb
    from scipy.spatial import ConvexHull
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    # --- Compute the Chroma–Depth Image Data ---
    n_depth = len(hue_chroma_list)
    hue_samples = np.linspace(0, 360, n_hue_samples, endpoint=False)
    image_data = np.zeros((n_depth, n_hue_samples, 3))

    # Sort by depth (shallowest first)
    depths_arr = np.array(depths)
    sort_idx = np.argsort(depths_arr)
    sorted_depths = depths_arr[sort_idx]
    sorted_hue_chroma_list = [hue_chroma_list[i] for i in sort_idx]

    for i, (hue_data, d) in enumerate(zip(sorted_hue_chroma_list, sorted_depths)):
        hues_pts, chromas_pts = hue_data

        # Convert measured (hue, chroma) data into Cartesian coordinates.
        x = chromas_pts * np.cos(np.radians(hues_pts))
        y = chromas_pts * np.sin(np.radians(hues_pts))
        points = np.column_stack((x, y))
        # Determine local maximum from measured data for this card.
        local_max = np.max(chromas_pts) if len(chromas_pts) > 0 else 0

        if len(points) < 3:
            # If not enough points, fill the row with a neutral color.
            for j, h in enumerate(hue_samples):
                lab = np.array([fixed_L, 0, 0]).reshape(1, 1, 3)
                try:
                    rgb = lab2rgb(lab).flatten()
                except Exception:
                    rgb = np.array([1, 1, 1])
                image_data[i, j, :] = rgb
            continue

        # Compute convex hull.
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        hull_polygon = np.vstack([hull_points, hull_points[0]])

        for j, h in enumerate(hue_samples):
            measured_chroma = ray_polygon_intersection_chroma(hull_polygon, h)
            if measured_chroma is None:
                measured_chroma = 0
            # Cap by local maximum.
            measured_chroma = min(measured_chroma, local_max)
            # Convert directly to LAB and then to RGB.
            a = measured_chroma * np.cos(np.radians(h))
            b = measured_chroma * np.sin(np.radians(h))
            lab = np.array([fixed_L, a, b]).reshape(1, 1, 3)
            try:
                rgb = lab2rgb(lab).flatten()
            except Exception:
                rgb = np.array([1, 1, 1])
            image_data[i, j, :] = rgb

    # --- Load Cuttlefish Sightings Data ---
    df = pd.read_excel(cuttlefish_excel_path)
    cuttlefish_depths = df["Depth (m)"].dropna().values

    max_depth_val = max(np.max(cuttlefish_depths), np.max(depths_arr))
    bins = np.arange(0, max_depth_val + 1, 1)

    # --- Create Combined Plot ---
    fig, (ax_img, ax_hist) = plt.subplots(1, 2, sharey=True, figsize=(14, 8))

    # Left: Display the chroma–depth image.
    # The extent is set so that x runs from 0 to 360 (hue) and y from the shallowest to the deepest.
    ax_img.imshow(image_data, aspect='auto',
                  extent=[0, 360, sorted_depths[0], sorted_depths[-1]],
                  origin='upper')
    ax_img.set_xlabel("Hue (degrees)")
    ax_img.set_ylabel("Depth (m)")
    ax_img.set_title("Hue-Chroma (LAB) with Depth")
    # Invert the y-axis so that deeper depths appear at the top.
    ax_img.invert_yaxis()

    # Right: Plot histogram of cuttlefish sightings.
    ax_hist.hist(cuttlefish_depths, bins=bins, orientation='horizontal',
                 color='gray', edgecolor='black')
    ax_hist.set_xlabel("Cuttlefish Count")
    ax_hist.set_title("Cuttlefish Sightings")
    ax_hist.invert_yaxis()  # Invert histogram y-axis as well.

    plt.suptitle("Chroma vs Depth and Cuttlefish Sightings", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("Y:/1. Thomas_2024/10_code/Philippines/Plots/Chroma_vs_Depth_and_Cuttlefish_Sightings.svg",
                format="svg")
    plt.show()


def plot_hue_chroma_depth_image(hue_chroma_list, depths, n_hue_samples=360, fixed_L=70):
    """
    For each color card (depth), sample n_hue_samples hue values (0–360°). For each sampled hue,
    compute the maximum chroma via the convex hull, capping it by the local maximum measured on that card.
    Then, instead of using HLS, we compute LAB values (with fixed lightness fixed_L) as:
         a = measured_chroma * cos(hue)
         b = measured_chroma * sin(hue)
    and convert that LAB color to RGB. This ensures that the colors in the filled image match
    the absolute chroma values used in the convex-hull plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.color import lab2rgb

    n_depth = len(hue_chroma_list)
    hue_samples = np.linspace(0, 360, n_hue_samples, endpoint=False)
    image_data = np.zeros((n_depth, n_hue_samples, 3))

    # Sort by depth (shallowest first)
    depths = np.array(depths)
    sort_idx = np.argsort(depths)
    sorted_depths = depths[sort_idx]
    sorted_hue_chroma_list = [hue_chroma_list[i] for i in sort_idx]

    for i, (hue_data, d) in enumerate(zip(sorted_hue_chroma_list, sorted_depths)):
        hues_pts, chromas_pts = hue_data

        # Convert measured (hue, chroma) data into Cartesian coordinates.
        x = chromas_pts * np.cos(np.radians(hues_pts))
        y = chromas_pts * np.sin(np.radians(hues_pts))
        points = np.column_stack((x, y))

        # Determine local maximum from measured data for this card.
        local_max = np.max(chromas_pts) if len(chromas_pts) > 0 else 0

        if len(points) < 3:
            # If not enough points for a convex hull, fill the row with grey (chroma=0).
            for j, h in enumerate(hue_samples):
                lab = np.array([fixed_L, 0, 0]).reshape(1, 1, 3)
                try:
                    rgb = lab2rgb(lab).flatten()
                except Exception:
                    rgb = np.array([1, 1, 1])
                image_data[i, j, :] = rgb
            continue

        # Compute convex hull.
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        hull_polygon = np.vstack([hull_points, hull_points[0]])

        for j, h in enumerate(hue_samples):
            measured_chroma = ray_polygon_intersection_chroma(hull_polygon, h)
            if measured_chroma is None:
                measured_chroma = 0
            # Cap the computed chroma by the local maximum measured on this card.
            measured_chroma = min(measured_chroma, local_max)
            # Instead of mapping to HLS saturation, convert directly to LAB:
            # a = measured_chroma * cos(hue), b = measured_chroma * sin(hue)
            a = measured_chroma * np.cos(np.radians(h))
            b = measured_chroma * np.sin(np.radians(h))
            lab = np.array([fixed_L, a, b]).reshape(1, 1, 3)
            try:
                rgb = lab2rgb(lab).flatten()
            except Exception:
                rgb = np.array([1, 1, 1])
            image_data[i, j, :] = rgb

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(image_data, aspect='auto',
                   extent=[0, 360, sorted_depths[-1], sorted_depths[0]],
                   origin='upper')
    ax.set_xlabel("Hue (degrees)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Hue-Chroma (LAB) with Depth")
    ax.set_xticks(np.arange(0, 361, 60))
    plt.savefig("Y:/1. Thomas_2024/10_code/Philippines/Plots/Hue-Chroma (LAB) with Depth.svg", format="svg")
    plt.show()


def plot_chroma_and_cuttlefish_histogram(hue_chroma_list, depths, cuttlefish_excel_path, n_hue_samples=360, fixed_L=70):
    from skimage.color import lab2rgb
    from scipy.spatial import ConvexHull
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    # --- Compute the Chroma–Depth Image Data ---
    n_depth = len(hue_chroma_list)
    hue_samples = np.linspace(0, 360, n_hue_samples, endpoint=False)
    image_data = np.zeros((n_depth, n_hue_samples, 3))

    # Sort by depth (shallowest first)
    depths_arr = np.array(depths)
    sort_idx = np.argsort(depths_arr)
    sorted_depths = depths_arr[sort_idx]
    sorted_hue_chroma_list = [hue_chroma_list[i] for i in sort_idx]

    for i, (hue_data, d) in enumerate(zip(sorted_hue_chroma_list, sorted_depths)):
        hues_pts, chromas_pts = hue_data

        # Convert measured (hue, chroma) data into Cartesian coordinates.
        x = chromas_pts * np.cos(np.radians(hues_pts))
        y = chromas_pts * np.sin(np.radians(hues_pts))
        points = np.column_stack((x, y))
        # Determine local maximum from measured data.
        local_max = np.max(chromas_pts) if len(chromas_pts) > 0 else 0

        if len(points) < 3:
            # If not enough points, fill the row with a neutral color.
            for j, h in enumerate(hue_samples):
                lab = np.array([fixed_L, 0, 0]).reshape(1, 1, 3)
                try:
                    rgb = lab2rgb(lab).flatten()
                except Exception:
                    rgb = np.array([1, 1, 1])
                image_data[i, j, :] = rgb
            continue

        # Compute convex hull.
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        hull_polygon = np.vstack([hull_points, hull_points[0]])

        for j, h in enumerate(hue_samples):
            measured_chroma = ray_polygon_intersection_chroma(hull_polygon, h)
            if measured_chroma is None:
                measured_chroma = 0
            measured_chroma = min(measured_chroma, local_max)
            a = measured_chroma * np.cos(np.radians(h))
            b = measured_chroma * np.sin(np.radians(h))
            lab = np.array([fixed_L, a, b]).reshape(1, 1, 3)
            try:
                rgb = lab2rgb(lab).flatten()
            except Exception:
                rgb = np.array([1, 1, 1])
            image_data[i, j, :] = rgb

    # --- Load Cuttlefish Sightings Data ---
    df = pd.read_excel(cuttlefish_excel_path)
    cuttlefish_depths = df["Depth (m)"].dropna().values

    max_depth_val = max(np.max(cuttlefish_depths), np.max(depths_arr))
    bins = np.arange(0, max_depth_val + 1, 1)

    # --- Create Combined Plot ---
    fig, (ax_img, ax_hist) = plt.subplots(1, 2, sharey=True, figsize=(14, 8))

    # Use the same extent as in the standalone hue–chroma depth image:
    #   x from 0 to 360, y from deep (largest depth) to shallow (smallest depth)
    ax_img.imshow(image_data, aspect='auto',
                  extent=[0, 360, sorted_depths[-1], sorted_depths[0]],
                  origin='upper')
    ax_img.set_xlabel("Hue (degrees)")
    ax_img.set_ylabel("Depth (m)")
    ax_img.set_title("Hue-Chroma (LAB) with Depth")
    # Explicitly set y-limits so that the deepest (largest) depth is at the top.
    ax_img.set_ylim(sorted_depths[-1], sorted_depths[0])

    # Plot the cuttlefish sightings histogram.
    ax_hist.hist(cuttlefish_depths, bins=bins, orientation='horizontal',
                 color='gray', edgecolor='black')
    ax_hist.set_xlabel("Cuttlefish Count")
    ax_hist.set_title("Cuttlefish Sightings")
    ax_hist.set_ylim(sorted_depths[-1], sorted_depths[0])

    plt.suptitle("Chroma vs Depth and Cuttlefish Sightings", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("Y:/1. Thomas_2024/10_code/Philippines/Plots/Chroma_vs_Depth_and_Cuttlefish_Sightings.svg",
                format="svg")
    plt.show()


# -------------------- MAIN FUNCTION -------------------- #

def main(image_dir, output_dir, depth_file):
    os.makedirs(output_dir, exist_ok=True)

    depth_data = pd.read_excel(depth_file)
    depth_mapping = {
        os.path.basename(row["Image Path"]): row["Depth"] for _, row in depth_data.iterrows()
    }

    hue_chroma_list = []
    depths = []

    for image_name in os.listdir(image_dir):
        if not image_name.lower().endswith('.arw'):
            continue

        image_path = os.path.join(image_dir, image_name)
        roi_file_path = os.path.join(image_dir, f"{os.path.splitext(image_name)[0]}_rois.json")

        if not os.path.exists(roi_file_path):
            print(f"ROI file not found for {image_name}, skipping.")
            continue

        print(f"Processing {image_name}...")

        with open(roi_file_path, 'r') as f:
            rois = json.load(f)['rois']

        image = load_raw_image_as_linear_rgb(image_path)
        avg_colors = extract_avg_colors_from_rois(image, rois)
        avg_colors_normalized = normalize_brightness(avg_colors)
        lch_values = np.array([rgb_to_lch(color) for color in avg_colors_normalized])
        hues, chromas = lch_values[:, 2], lch_values[:, 1]
        hue_chroma_list.append((hues, chromas))
        depths.append(depth_mapping.get(image_name, 0))

        output_json_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_colors.json")
        with open(output_json_path, 'w') as f:
            json.dump({"colors": avg_colors_normalized.tolist()}, f)

    # Call the plotting functions.
    plot_hue_chroma_with_convex_hull(hue_chroma_list, depths)
    plot_hue_chroma_3d(hue_chroma_list, depths)
    plot_chroma_vs_depth_by_hue(hue_chroma_list, depths)

    cuttlefish_excel_path = r"Y:\1. Thomas_2024\10_code\Philippines\Cuttlefish_sighting_depths.xlsx"
    plot_chroma_and_cuttlefish_histogram(hue_chroma_list, depths, cuttlefish_excel_path)
    plot_hue_chroma_depth_image(hue_chroma_list, depths)
    plot_chroma_and_cuttlefish_histogram(hue_chroma_list,depths,cuttlefish_excel_path)


if __name__ == "__main__":
    image_directory = r"Y:\1. Thomas_2024\10_code\Philippines\D1_Surface_images_versus_depth"
    output_directory = r"Y:\1. Thomas_2024\10_code\Philippines\Processed"
    depth_file = r"Y:\1. Thomas_2024\10_code\Philippines\D1_Surface_images_versus_depth\D1_Surface_images_versus_depth.xlsx"
    main(image_directory, output_directory, depth_file)
