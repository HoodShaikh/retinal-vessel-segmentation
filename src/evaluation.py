"""Dataset loading, metrics, CSV files and graphs for DRIVE and CHASE_DB1."""

import csv
import glob
import os
import re

import numpy as np
from scipy import ndimage as ndi

from .utils import list_images, file_stem

METRICS = ["Accuracy", "Precision", "Recall", "Specificity", "Dice", "IoU", "ThinVesselRecall"]
CSV_COLUMNS = ["Dataset", "Split", "Image", "Evaluated", "FOV_Source"] + METRICS + ["TP", "TN", "FP", "FN"]


# ------------------------------------------------------------------
# Dataset discovery
# ------------------------------------------------------------------

def first_existing(paths):
    return next((p for p in paths if os.path.exists(p)), None)


def find_drive_samples(root):
    """
    Official DRIVE layout:
        DRIVE/training/images/21_training.tif   1st_manual/21_manual1.gif   mask/21_training_mask.gif
        DRIVE/test/images/01_test.tif           mask/01_test_mask.gif       (1st_manual only if you have it)
    """
    samples = []
    for split in ("training", "test"):
        split_dir = os.path.join(root, split)
        for image_path in list_images(os.path.join(split_dir, "images")):
            stem = file_stem(image_path)
            number = stem.split("_")[0]
            groundtruth = first_existing(
                [os.path.join(split_dir, "1st_manual", f"{number}_manual1{ext}") for ext in (".gif", ".png", ".tif")])
            fov_mask = first_existing(
                [os.path.join(split_dir, "mask", f"{stem}_mask{ext}") for ext in (".gif", ".png", ".tif")])
            samples.append({"dataset": "DRIVE", "split": split, "name": stem, "image": image_path,
                            "groundtruth": groundtruth, "fov": fov_mask})
    return samples


def find_chase_samples(root):
    """
    Official CHASEDB1.zip is flat: Image_01L.jpg, Image_01L_1stHO.png, Image_01L_2ndHO.png ...
    The first human observer (1stHO) is used as ground truth. No FOV masks are provided.
    """
    samples = []
    pattern = re.compile(r"^Image_\d{2}[LR]\.(jpg|jpeg|png|tif)$", re.IGNORECASE)
    image_paths = sorted(p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                         if pattern.match(os.path.basename(p)))
    for image_path in image_paths:
        stem = file_stem(image_path)
        folder = os.path.dirname(image_path)
        groundtruth = first_existing(
            [os.path.join(folder, f"{stem}_1stHO{ext}") for ext in (".png", ".jpg", ".tif")]
            + glob.glob(os.path.join(root, "**", f"{stem}_1stHO.png"), recursive=True))
        samples.append({"dataset": "CHASE_DB1", "split": "all", "name": stem, "image": image_path,
                        "groundtruth": groundtruth, "fov": None})
    return samples


def find_samples(dataset_name, datasets_root):
    root = os.path.join(datasets_root, dataset_name)
    if dataset_name == "DRIVE":
        return find_drive_samples(root)
    if dataset_name == "CHASE_DB1":
        return find_chase_samples(root)
    raise ValueError(f"Unknown dataset: {dataset_name}")


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def thin_vessel_pixels(groundtruth):
    """Ground-truth pixels on vessels about 3 px wide or less."""
    half_width = ndi.maximum_filter(ndi.distance_transform_edt(groundtruth), size=5)
    return groundtruth & (half_width <= 1.5)


def calculate_metrics(prediction, groundtruth, fov):
    """Pixel-wise metrics computed only inside the field of view."""
    pred = (prediction > 0)[fov]
    truth = groundtruth[fov]

    tp = int(np.sum(pred & truth))
    tn = int(np.sum(~pred & ~truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    eps = 1e-9

    thin = thin_vessel_pixels(groundtruth) & fov
    return {
        "Accuracy": (tp + tn) / (tp + tn + fp + fn),
        "Precision": tp / (tp + fp + eps),
        "Recall": tp / (tp + fn + eps),
        "Specificity": tn / (tn + fp + eps),
        "Dice": 2 * tp / (2 * tp + fp + fn + eps),
        "IoU": tp / (tp + fp + fn + eps),
        "ThinVesselRecall": np.sum((prediction > 0) & thin) / (np.sum(thin) + eps),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


# ------------------------------------------------------------------
# CSV files and summaries
# ------------------------------------------------------------------

def metrics_row(sample, fov_source, metrics):
    row = {"Dataset": sample["dataset"], "Split": sample["split"], "Image": sample["name"],
           "Evaluated": "Yes" if metrics else "No - no ground truth", "FOV_Source": fov_source}
    for key in METRICS + ["TP", "TN", "FP", "FN"]:
        value = metrics.get(key, "") if metrics else ""
        row[key] = round(float(value), 4) if isinstance(value, float) else value
    return row


def write_csv(path, rows, columns):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def summarise(rows):
    """Mean and standard deviation per dataset and per split (only evaluated images)."""
    groups = {}
    for row in rows:
        groups.setdefault((row["Dataset"], "all splits"), []).append(row)
        if row["Split"] != "all":
            groups.setdefault((row["Dataset"], row["Split"]), []).append(row)

    summary = []
    for (dataset, split), group in groups.items():
        evaluated = [r for r in group if r["Evaluated"] == "Yes"]
        entry = {"Dataset": dataset, "Split": split,
                 "Images_Segmented": len(group), "Images_Evaluated": len(evaluated)}
        for metric in METRICS:
            values = [float(r[metric]) for r in evaluated]
            entry[f"{metric}_Mean"] = round(float(np.mean(values)), 4) if values else ""
            entry[f"{metric}_Std"] = round(float(np.std(values)), 4) if values else ""
        summary.append(entry)
    return summary


def summary_columns():
    columns = ["Dataset", "Split", "Images_Segmented", "Images_Evaluated"]
    for metric in METRICS:
        columns += [f"{metric}_Mean", f"{metric}_Std"]
    return columns


def summary_markdown(summary):
    lines = ["| Dataset | Split | Segmented | Evaluated | " + " | ".join(METRICS) + " |",
             "|" + "---|" * (4 + len(METRICS))]
    for s in summary:
        if s["Images_Evaluated"]:
            values = [f"{s[m + '_Mean']:.4f} ± {s[m + '_Std']:.4f}" for m in METRICS]
        else:
            values = ["not evaluated"] * len(METRICS)
        lines.append(f"| {s['Dataset']} | {s['Split']} | {s['Images_Segmented']} | {s['Images_Evaluated']} | "
                     + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def print_summary(summary):
    print("\n" + "=" * 112)
    print("OVERALL PERFORMANCE  (mean over evaluated images, inside the field of view)")
    print("=" * 112)
    print(f"{'Dataset':<11}{'Split':<12}{'Seg.':>5}{'Eval.':>6}" + "".join(f"{m[:11]:>12}" for m in METRICS))
    for s in summary:
        values = "".join(f"{s[m + '_Mean']:>12.4f}" for m in METRICS) if s["Images_Evaluated"] \
            else "   no ground truth - segmentation only"
        print(f"{s['Dataset']:<11}{s['Split']:<12}{s['Images_Segmented']:>5}{s['Images_Evaluated']:>6}{values}")


# ------------------------------------------------------------------
# Graphs
# ------------------------------------------------------------------

def plot_summary(summary, path):
    import matplotlib.pyplot as plt

    groups = []
    for s in summary:
        if not s["Images_Evaluated"]:
            continue
        overall = next(o for o in summary if o["Dataset"] == s["Dataset"] and o["Split"] == "all splits")
        # show a split separately only when it differs from the whole dataset
        if s["Split"] == "all splits" or s["Images_Evaluated"] != overall["Images_Evaluated"]:
            groups.append(s)
    if not groups:
        return
    x = np.arange(len(METRICS))
    width = 0.8 / len(groups)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, s in enumerate(groups):
        means = [s[f"{m}_Mean"] for m in METRICS]
        stds = [s[f"{m}_Std"] for m in METRICS]
        label = s["Dataset"] if s["Split"] == "all splits" else f"{s['Dataset']} {s['Split']}"
        bars = ax.bar(x + (i - (len(groups) - 1) / 2) * width, means, width, yerr=stds, capsize=3,
                      label=f"{label} (n={s['Images_Evaluated']})")
        for bar, value in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, 0.02, f"{value:.3f}", ha="center",
                    va="bottom", rotation=90, fontsize=8, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (inside field of view)")
    ax.set_title("Vessel segmentation performance by dataset")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", ncol=len(groups), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_per_image(rows, dataset, path):
    import matplotlib.pyplot as plt

    evaluated = [r for r in rows if r["Dataset"] == dataset and r["Evaluated"] == "Yes"]
    if not evaluated:
        return
    names = [r["Image"] for r in evaluated]
    fig, ax = plt.subplots(figsize=(max(8, 0.32 * len(names) + 2), 4.5))
    x = np.arange(len(names))
    for metric, offset in (("Accuracy", -0.2), ("Dice", 0.0), ("IoU", 0.2)):
        ax.bar(x + offset, [float(r[metric]) for r in evaluated], 0.2, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{dataset}: image-wise Accuracy, Dice and IoU")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_ablation(rows, path):
    import matplotlib.pyplot as plt

    datasets = sorted({r["Dataset"] for r in rows})
    variants = ["gaussian", "none", "sharpen"]
    labels = {"gaussian": "Gaussian blur first", "none": "No pre-filter", "sharpen": "Sharpening first (final)"}
    shown = ["Accuracy", "Recall", "Dice", "ThinVesselRecall"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.5), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        x = np.arange(len(shown))
        for i, variant in enumerate(variants):
            row = next(r for r in rows if r["Dataset"] == dataset and r["Prefilter"] == variant)
            ax.bar(x + (i - 1) * 0.27, [row[m] for m in shown], 0.27, label=labels[variant])
        ax.set_xticks(x)
        ax.set_xticklabels(shown)
        ax.set_ylim(0, 1.15)
        ax.set_title(f"{dataset}: effect of the first filtering step")
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].legend(loc="upper right", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
