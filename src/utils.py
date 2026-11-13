import os

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".ppm", ".gif", ".webp")


def read_color(path):
    """Returns a BGR image, or None if the file is not a readable image."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        # Pillow handles formats some OpenCV builds cannot (GIF, WEBP, odd TIFFs)
        try:
            image = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
        except (OSError, ValueError):
            return None
    return image


def read_gray(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        # DRIVE masks and annotations are GIFs; OpenCV < 4.7 cannot read them
        image = np.array(Image.open(path).convert("L"))
    return image


def read_binary(path, shape=None):
    mask = read_gray(path) > 127
    if shape is not None and mask.shape != shape:
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]),
                          interpolation=cv2.INTER_NEAREST) > 0
    return mask


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, name) for name in os.listdir(folder)
                  if name.lower().endswith(IMAGE_EXTENSIONS) and not name.startswith("."))


def file_stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def make_folder(path):
    os.makedirs(path, exist_ok=True)
    return path


def estimate_fov_mask(image):
    """Finds the bright circular retina region when no FOV mask file is available."""
    red = cv2.GaussianBlur(image[:, :, 2].astype(np.float32), (0, 0), 2)
    fov = (red > 0.10 * np.percentile(red, 99)).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(fov, connectivity=8)
    if count > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        fov = (labels == largest).astype(np.uint8)
    fov = ndi.binary_fill_holes(fov).astype(np.uint8)
    if fov.mean() < 0.2:
        # no clear circular retina (e.g. a cropped photo): use the whole image
        fov = np.ones_like(fov)

    # shrink slightly so the camera rim is not mistaken for a vessel
    size = max(5, int(round(fov.shape[1] / 100)) | 1)
    fov = cv2.erode(fov, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    return fov.astype(bool)


def to_uint8(image, fov=None):
    """Stretches any response map to 0-255 for saving or display."""
    image = image.astype(np.float32)
    values = image[fov] if fov is not None else image.ravel()
    low, high = np.percentile(values, 1), np.percentile(values, 99.5)
    scaled = np.clip((image - low) / (high - low + 1e-6), 0, 1) * 255
    if fov is not None:
        scaled[~fov] = 0
    return scaled.astype(np.uint8)


def vessel_overlay(image, mask, color=(0, 255, 255)):
    """Original image with detected vessels painted in a bright colour (BGR)."""
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(image, 0.35, overlay, 0.65, 0)


def error_overlay(prediction, groundtruth, fov):
    """White = correct vessel, red = false vessel, blue = missed vessel (BGR)."""
    pred = prediction > 0
    result = np.zeros(groundtruth.shape + (3,), np.uint8)
    result[pred & groundtruth] = (255, 255, 255)
    result[pred & ~groundtruth & fov] = (40, 40, 255)
    result[~pred & groundtruth & fov] = (255, 160, 0)
    return result


def add_title(image, title):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    scale = max(0.6, image.shape[1] / 900)
    thickness = max(1, int(round(scale * 2)))
    bar_height = int(45 * scale)
    bar = np.full((bar_height, image.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (int(12 * scale), int(bar_height * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (20, 20, 20), thickness, cv2.LINE_AA)
    return np.vstack([bar, image])


def side_by_side(panels, gap=12):
    """Joins titled images horizontally: [(image, title), ...]."""
    titled = [add_title(img, title) for img, title in panels]
    height = max(p.shape[0] for p in titled)
    spacer = np.full((height, gap, 3), 255, np.uint8)
    row = []
    for panel in titled:
        pad = height - panel.shape[0]
        row.append(cv2.copyMakeBorder(panel, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        row.append(spacer)
    return np.hstack(row[:-1])


def save_image(path, image, jpeg_quality=92):
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if path.lower().endswith(".jpg") else []
    cv2.imwrite(path, image, params)
