import SimpleITK
from glob import glob
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
from sklearn.utils import resample

def load_image_file(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def load_gt_label_and_spacing(input_file):
    # Use SimpleITK to read a file
    result = SimpleITK.ReadImage(input_file)
    sitk_spacing = result.GetSpacing()
    array_spacing = sitk_spacing[::-1]
    return array_spacing, SimpleITK.GetArrayFromImage(result)


def calculate_3d_iou(vol1, vol2):
    # Calculate intersection and union
    intersection = np.logical_and(vol1, vol2).sum()
    union = np.logical_or(vol1, vol2).sum()

    # Calculate IoU
    if union == 0:
        return 0  # Avoid division by zero
    return intersection / union


def calculate_3d_hd95_from_points(vol1_points, vol2_points):
    if not vol1_points.size or not vol2_points.size:
        return np.inf

    distances = cdist(vol1_points, vol2_points, metric='euclidean').astype(np.float32)
    d1 = np.percentile(np.min(distances, axis=1), 95)
    d2 = np.percentile(np.min(distances, axis=0), 95)
    return max(d1, d2)


def calculate_3d_assd_from_points(vol1_points, vol2_points):
    if not vol1_points.size or not vol2_points.size:
        return np.inf

    distances = cdist(vol1_points, vol2_points, metric='euclidean').astype(np.float32)
    assd1 = np.mean(np.min(distances, axis=1))
    assd2 = np.mean(np.min(distances, axis=0))
    return (assd1 + assd2) / 2


def match_labels_single_bone(gt_volume, pred_volume, label_range):
    # Initialize IoU dictionary
    matches = {}

    # Loop through each label in ground truth within the specified range
    for label in label_range:
        gt_mask = gt_volume == label
        if gt_mask.any():
            # Calculate IoU with each label in prediction
            iou_scores = {pred_label: calculate_3d_iou(gt_mask, pred_volume == pred_label)
                          for pred_label in label_range if (pred_volume == pred_label).any()}

            # Find the prediction label with the highest IoU
            if iou_scores:
                best_match = max(iou_scores, key=iou_scores.get)
                matches[label] = (best_match, iou_scores[best_match])

    return matches


def match_labels_whole_pelvis(gt_volume, pred_volume):
    SA_matches = match_labels_single_bone(gt_volume, pred_volume, range(1, 11))
    LI_matches = match_labels_single_bone(gt_volume, pred_volume, range(11, 21))
    RI_matches = match_labels_single_bone(gt_volume, pred_volume, range(21, 31))
    matches = SA_matches | LI_matches | RI_matches
    return matches


def extract_surface_points(volume, label, pixel_spacing, sample_size=10000):
    # Create a binary mask for the given label
    mask = volume == label

    # Use morphological operations to find the surface (contour) of the mask
    struct = generate_binary_structure(3, 1)  # 3D connectivity
    eroded = binary_erosion(mask, structure=struct)
    surface_mask = binary_dilation(mask, structure=struct) & ~eroded

    # Extract coordinates of the surface points
    surface_points = np.argwhere(surface_mask)

    # Downsample if there are too many points
    if surface_points.shape[0] > sample_size:
        surface_points = resample(surface_points, n_samples=sample_size, random_state=2024)

    # Apply pixel spacing to convert coordinates to real-world measurements
    adjusted_points = surface_points * pixel_spacing  # Apply pixel spacing
    return adjusted_points


def calculate_sphere_radius(volume, label):
    points = np.argwhere(volume == label)
    if points.size == 0:
        return np.inf, np.inf  # Return inf if no points exist
    center = np.mean(points, axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    radius = np.max(radii)
    return radius


def evaluate_fracture_segmentation(matches, gt_volume, pred_volume, spacing):
    results = {}
    for label in matches:
        if matches[label][1] > 0:
            pred_label, _ = matches[label]
            gt_points = extract_surface_points(gt_volume, label, spacing)
            pred_points = extract_surface_points(pred_volume, pred_label, spacing)
            hd95 = calculate_3d_hd95_from_points(gt_points, pred_points)
            assd = calculate_3d_assd_from_points(gt_points, pred_points)
        else:
            print("Label", label, "using maximum value.")
            radius = calculate_sphere_radius(gt_volume, label)
            hd95 = 2 * radius
            assd = radius

        results[label] = (matches[label][1], hd95, assd)
    return results


def evaluate_anatomical_segmentation(gt_volume, pred_volume, spacing):
    results = {}
    anatomical_ranges = {
        'SA': range(1, 11),
        'LI': range(11, 21),
        'RI': range(21, 31)
    }
    for bone, label_range in anatomical_ranges.items():
        gt_mask = np.isin(gt_volume, label_range)
        pred_mask = np.isin(pred_volume, label_range)
        gt_points = extract_surface_points(gt_mask, 1, spacing)
        pred_points = extract_surface_points(pred_mask, 1, spacing)
        iou = calculate_3d_iou(gt_mask, pred_mask)
        hd95 = calculate_3d_hd95_from_points(gt_points, pred_points)
        assd = calculate_3d_assd_from_points(gt_points, pred_points)

        results[bone] = (iou, hd95, assd)

    return results


def evaluate_3d_single_case(gt_volume, pred_volume, spacing, verbose = False):
    if verbose:
        print("Spacing =", spacing)
        print("Size =", gt_volume.shape)

    # Extract and match sacrum fragments
    matches = match_labels_whole_pelvis(gt_volume, pred_volume)
    if verbose:
        print("Matches and IoU scores:", matches)

    # Evaluate fracture segmentation results
    if verbose:
        print("Evaluate fracture segmentation results")
    # Initialize sums and counter
    fracture_iou, fracture_hd95, fracture_assd = 0, 0, 0
    count = 0
    # Loop through results to process metrics and calculate totals
    fracture_results = evaluate_fracture_segmentation(matches, gt_volume, pred_volume, spacing)
    for label, (iou, hd95, assd) in fracture_results.items():
        if verbose:
            print(f"Label {label}: IoU = {iou}, HD95 = {hd95}, ASSD = {assd}")
        fracture_iou += iou
        fracture_hd95 += hd95
        fracture_assd += assd
        count += 1

    # Calculate averages if there are any entries
    fracture_iou = fracture_iou / count
    fracture_hd95 = fracture_hd95 / count
    fracture_assd = fracture_assd / count
    if verbose:
        print(f"Fracture Average IoU = {fracture_iou:.2f}, "
              f"Average HD95 = {fracture_hd95:.2f}, "
              f"Average ASSD = {fracture_assd:.2f}")

    # Evaluate anatomical segmentation results
    if verbose:
        print("Evaluate anatomical segmentation results")
    # Initialize sums and counter
    anatomical_iou, anatomical_hd95, anatomical_assd = 0, 0, 0
    count = 0
    # Loop through results to process metrics and calculate totals
    anatomical_results = evaluate_anatomical_segmentation(gt_volume, pred_volume, spacing)
    for label, (iou, hd95, assd) in anatomical_results.items():
        if verbose:
            print(f"Label {label}: IoU = {iou}, HD95 = {hd95}, ASSD = {assd}")
        anatomical_iou += iou
        anatomical_hd95 += hd95
        anatomical_assd += assd

    # Calculate averages if there are any entries
    anatomical_iou = anatomical_iou / 3
    anatomical_hd95 = anatomical_hd95 / 3
    anatomical_assd = anatomical_assd / 3
    if verbose:
        print(f"Anatomical Average IoU = {anatomical_iou:.2f}, "
              f"Average HD95 = {anatomical_hd95:.2f}, "
              f"Average ASSD = {anatomical_assd:.2f}")

    metrics_single_case = {"fracture_iou": fracture_iou,
                           "fracture_hd95": fracture_hd95,
                           "fracture_assd": fracture_assd,
                           "anatomical_iou": anatomical_iou,
                           "anatomical_hd95": anatomical_hd95,
                           "anatomical_assd": anatomical_assd}
    return metrics_single_case


if __name__ == "__main__":
    pred_volume = load_image_file(location=Path("/home/yudi/PycharmProjects/PENGWIN_dataset/prediction"))
    spacing, gt_volume = load_gt_label_and_spacing(Path("/home/yudi/PycharmProjects/PENGWIN_dataset/ground_truth/107.mha"))
    metrics_single_case = evaluate_3d_single_case(gt_volume, pred_volume, spacing, verbose = True)
