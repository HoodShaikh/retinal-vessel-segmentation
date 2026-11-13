"""
Sharpening-first retinal vessel segmentation (classical image processing only).

Steps: green channel -> unsharp-mask sharpening -> CLAHE -> background removal
-> line detector + Hessian ridge detector -> fusion -> hysteresis threshold
-> small-object removal.

All sizes below are for a DRIVE-sized retina (FOV about 540 px wide) and are
scaled automatically for larger images. The values were chosen on the 20 DRIVE
training images only and then kept fixed for every other image.
"""

import cv2
import numpy as np
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import apply_hysteresis_threshold

from .utils import estimate_fov_mask

PARAMS = {
    "sharpen_sigma": 2.0,
    "sharpen_amount": 1.0,
    "clahe_clip": 2.0,
    "clahe_grid": 8,
    "background_size": 25,
    "line_window": 15,
    "line_lengths": [3, 5, 7, 9, 11, 13, 15],
    "line_angles": 12,
    "hessian_sigmas": [1.0, 1.5, 2.0, 3.0],
    "weight_darkness": 0.4,
    "weight_line": 0.2,
    "weight_hessian": 0.4,
    "seed_percent": 10.0,
    "grow_percent": 13.0,
    "min_area": 30,
}

REFERENCE_FOV_WIDTH = 540.0
MAX_WORKING_FOV_WIDTH = 1100.0


def odd(value):
    return max(3, int(round(value)) | 1)


def zscore(image, fov):
    values = image[fov]
    return (image - values.mean()) / (values.std() + 1e-6)


def fov_width(fov):
    columns = np.where(fov.any(axis=0))[0]
    return float(columns[-1] - columns[0])


def pad_outside_fov(gray, fov, iterations=50):
    """Fills the black corners with neighbouring retina values so filters do not see a fake edge."""
    image = gray.astype(np.float32).copy()
    known = fov.astype(np.float32).copy()
    kernel = np.ones((3, 3), np.float32)
    for _ in range(iterations):
        total = cv2.filter2D(image * known, -1, kernel, borderType=cv2.BORDER_REFLECT)
        count = cv2.filter2D(known, -1, kernel, borderType=cv2.BORDER_REFLECT)
        new_pixels = (known == 0) & (count > 0)
        if not new_pixels.any():
            break
        image[new_pixels] = total[new_pixels] / count[new_pixels]
        known[new_pixels] = 1
    image[known == 0] = image[known > 0].mean()
    return image


def sharpen(image, sigma, amount):
    """Unsharp masking: image + amount * (image - blurred image)."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return np.clip(image + amount * (image - blurred), 0, 255)


def line_kernel(length, angle):
    kernel = np.zeros((length, length), np.float32)
    centre = (length - 1) / 2
    theta = np.deg2rad(angle)
    for r in np.linspace(-centre, centre, 4 * length):
        x = int(round(centre + r * np.cos(theta)))
        y = int(round(centre - r * np.sin(theta)))
        kernel[y, x] = 1
    return kernel / kernel.sum()


def line_detector(vessel_map, fov, window, lengths, angles):
    """
    Multiscale line detector: best average along a line through each pixel
    (over all angles) minus the average of the surrounding square window.
    A vessel has one direction where the whole line is dark; noise does not.
    """
    window_mean = cv2.blur(vessel_map, (window, window))
    combined = zscore(vessel_map, fov)
    for length in lengths:
        best_line = np.full(vessel_map.shape, -np.inf, np.float32)
        for i in range(angles):
            kernel = line_kernel(length, i * 180.0 / angles)
            best_line = np.maximum(best_line, cv2.filter2D(vessel_map, -1, kernel))
        combined += zscore(best_line - window_mean, fov)
    return combined / (len(lengths) + 1)


def hessian_ridge(vessel_map, sigmas):
    """
    Second-derivative ridge detector. Across a bright vessel the intensity curves
    down sharply (large negative eigenvalue), along it the profile stays flat.
    Small sigma responds to thin capillaries, larger sigma to wide vessels.
    """
    response = np.zeros(vessel_map.shape, np.float32)
    for sigma in sigmas:
        hessian = hessian_matrix(vessel_map, sigma=sigma, order="rc", use_gaussian_derivatives=False)
        _, smaller_eigenvalue = hessian_matrix_eigvals(hessian)
        ridge = np.clip(-smaller_eigenvalue, 0, None) * sigma ** 2
        response = np.maximum(response, ridge.astype(np.float32))
    return response


def remove_small_objects(mask, min_area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def segment_vessels(image, fov=None, prefilter="sharpen", params=PARAMS):
    """
    image: BGR fundus image. fov: boolean retina mask (estimated if None).
    prefilter: "sharpen" (final method), "gaussian" or "none" (used for the ablation study).
    Returns a dict with the binary mask (0/255), the FOV and every intermediate stage.
    """
    if fov is None:
        fov = estimate_fov_mask(image)
    original_size = image.shape[:2]
    original_fov = fov

    # very large photos (e.g. downloaded from the web) are processed at a reduced size for speed
    width = fov_width(fov)
    resize_factor = min(1.0, MAX_WORKING_FOV_WIDTH / width)
    if resize_factor < 1.0:
        image = cv2.resize(image, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
        fov = cv2.resize(fov.astype(np.uint8), (image.shape[1], image.shape[0]),
                         interpolation=cv2.INTER_NEAREST) > 0
        width = fov_width(fov)
    scale = width / REFERENCE_FOV_WIDTH

    green = image[:, :, 1].astype(np.float32)
    padded = pad_outside_fov(green, fov)

    if prefilter == "sharpen":
        prefiltered = sharpen(padded, params["sharpen_sigma"] * scale, params["sharpen_amount"])
    elif prefilter == "gaussian":
        prefiltered = cv2.GaussianBlur(padded, (5, 5), 0)
    else:
        prefiltered = padded

    clahe = cv2.createCLAHE(clipLimit=params["clahe_clip"],
                            tileGridSize=(params["clahe_grid"], params["clahe_grid"]))
    enhanced = clahe.apply(prefiltered.astype(np.uint8)).astype(np.float32)

    # background estimate: a median window much wider than any vessel ignores the vessels
    background = cv2.medianBlur(enhanced.astype(np.uint8), odd(params["background_size"] * scale))
    vessel_map = np.clip(background.astype(np.float32) - enhanced, 0, None)

    lengths = [odd(length * scale) for length in params["line_lengths"]]
    line_response = line_detector(vessel_map, fov, odd(params["line_window"] * scale),
                                  lengths, params["line_angles"])
    hessian_response = hessian_ridge(vessel_map, [s * scale for s in params["hessian_sigmas"]])
    darkness = cv2.GaussianBlur(vessel_map, (0, 0), scale)

    fused = (params["weight_darkness"] * zscore(darkness, fov)
             + params["weight_line"] * zscore(line_response, fov)
             + params["weight_hessian"] * zscore(hessian_response, fov))
    fused[~fov] = fused[fov].min()

    # hysteresis: strong pixels are sure vessels, weak pixels survive only if connected to them
    high = np.percentile(fused[fov], 100 - params["seed_percent"])
    low = np.percentile(fused[fov], 100 - params["grow_percent"])
    seeds = (fused >= high) & fov
    thresholded = apply_hysteresis_threshold(fused, low, high) & fov
    cleaned = remove_small_objects(thresholded, params["min_area"] * scale ** 2)

    final_mask = cleaned.astype(np.uint8) * 255
    if resize_factor < 1.0:
        final_mask = cv2.resize(final_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
        final_mask = np.where(final_mask > 127, 255, 0).astype(np.uint8)

    return {
        "mask": final_mask,
        "fov": original_fov,
        "stage_fov": fov,
        "resize_factor": resize_factor,
        "stages": {
            "01_green_channel": green,
            "02_sharpened": prefiltered,
            "03_clahe": enhanced,
            "04_background_removed": vessel_map,
            "05_line_detector": line_response,
            "06_hessian_ridge": hessian_response,
            "07_fused_response": fused,
            "08_strong_seeds": seeds.astype(np.uint8) * 255,
            "09_hysteresis_threshold": thresholded.astype(np.uint8) * 255,
            "10_final_mask": cleaned.astype(np.uint8) * 255,
        },
    }
