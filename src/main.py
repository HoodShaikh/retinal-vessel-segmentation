"""
Command-line entry point.

    python run.py                                 custom images in input/
    python run.py --mode dataset --dataset DRIVE  evaluate one dataset
    python run.py --mode dataset                  evaluate DRIVE and CHASE_DB1
    python run.py --mode all                      everything
"""

import argparse
import os
import sys
import time

import matplotlib
import numpy as np

from .pipeline import segment_vessels
from .utils import (read_color, read_binary, list_images, file_stem, make_folder, estimate_fov_mask,
                    to_uint8, vessel_overlay, error_overlay, side_by_side, save_image)
from . import evaluation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_NAMES = ["DRIVE", "CHASE_DB1"]


# ------------------------------------------------------------------
# Saving outputs
# ------------------------------------------------------------------

def save_detailed_outputs(folder, image, result, groundtruth=None, metrics=None, title=""):
    """Intermediate stages + one overview figure (outputs/ folder)."""
    import matplotlib.pyplot as plt

    make_folder(folder)
    fov = result["stage_fov"]
    for name, stage in result["stages"].items():
        stage_image = stage if stage.dtype == np.uint8 else to_uint8(stage, fov)
        save_image(os.path.join(folder, f"{name}.png"), stage_image)

    panels = [(image[:, :, ::-1], "Original", None)]
    shown_stages = ["02_sharpened", "03_clahe", "04_background_removed", "05_line_detector",
                    "06_hessian_ridge", "07_fused_response", "09_hysteresis_threshold"]
    for name in shown_stages:
        stage = result["stages"][name]
        panels.append((stage if stage.dtype == np.uint8 else to_uint8(stage, fov),
                       name[3:].replace("_", " ").capitalize(), "gray"))
    panels.append((result["mask"], "Final vessel mask", "gray"))

    if groundtruth is not None:
        errors = error_overlay(result["mask"], groundtruth, result["fov"])
        save_image(os.path.join(folder, "11_error_map.png"), errors)
        panels.append((groundtruth.astype(np.uint8) * 255, "Ground truth (1st observer)", "gray"))
        panels.append((errors[:, :, ::-1], "White=correct  Red=false  Blue=missed", None))
    else:
        panels.append((vessel_overlay(image, result["mask"])[:, :, ::-1], "Detected vessels", None))

    fig, axes = plt.subplots(3, 4, figsize=(18, 13.5))
    for ax, (picture, label, cmap) in zip(axes.flat, panels):
        ax.imshow(picture, cmap=cmap)
        ax.set_title(label, fontsize=12)
    last = axes.flat[-1]
    if metrics:
        text = "\n".join(f"{m:<17}: {metrics[m]:.4f}" for m in evaluation.METRICS)
        last.text(0.02, 0.5, text + "\n\n(inside field of view)", fontsize=12, family="monospace", va="center")
    elif groundtruth is None:
        last.text(0.02, 0.5, "No ground truth available.\nAccuracy is not computed\nfor this image.",
                  fontsize=13, va="center")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(title, fontsize=17)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "pipeline_overview.jpg"), dpi=90)
    plt.close(fig)


def save_final_outputs(folder, name, image, mask):
    """FINAL_RESULTS: only the clean binary mask and the original / mask / highlighted result."""
    make_folder(folder)
    save_image(os.path.join(folder, f"{name}_vessel_mask.png"), mask)
    result = side_by_side([(image, "Original"), (mask, "Vessel mask"),
                           (vessel_overlay(image, mask), "Detected vessels")])
    save_image(os.path.join(folder, f"{name}_result.jpg"), result)
    return os.path.join(folder, f"{name}_result.jpg")


# ------------------------------------------------------------------
# Mode A: custom images
# ------------------------------------------------------------------

def run_custom(input_folder, save_details=True, show=False):
    images = list_images(input_folder)
    print("\n" + "=" * 70)
    print(f"CUSTOM IMAGE MODE  -  {len(images)} image(s) in {input_folder}")
    print("=" * 70)
    if not images:
        print("No images found. Copy any retinal fundus photo (.jpg/.png/.tif) into the input folder and run again.")
        return []

    result_files = []
    for index, path in enumerate(images, 1):
        start = time.time()
        name = file_stem(path)
        image = read_color(path)
        if image is None:
            print(f"[{index}/{len(images)}] {os.path.basename(path):<32} skipped - not a readable image file")
            continue
        result = segment_vessels(image)
        coverage = 100 * np.mean(result["mask"][result["fov"]] > 0)

        if save_details:
            save_detailed_outputs(os.path.join(PROJECT_ROOT, "outputs", "custom_images", name),
                                  image, result, title=f"Custom image - {os.path.basename(path)}")
        final_path = save_final_outputs(os.path.join(PROJECT_ROOT, "FINAL_RESULTS", "custom_images"),
                                        name, image, result["mask"])
        result_files.append(final_path)
        print(f"[{index}/{len(images)}] {os.path.basename(path):<32} {image.shape[1]}x{image.shape[0]}  "
              f"vessels cover {coverage:4.1f}% of retina  ({time.time() - start:.1f} s)")

    print("\nNo ground truth exists for these images, so no accuracy is reported.")
    print(f"Final results : {os.path.join(PROJECT_ROOT, 'FINAL_RESULTS', 'custom_images')}")
    if save_details:
        print(f"Step-by-step  : {os.path.join(PROJECT_ROOT, 'outputs', 'custom_images')}")

    if show:
        show_images(result_files)
    return result_files


def show_images(paths):
    import matplotlib.pyplot as plt
    import cv2
    for path in paths:
        picture = cv2.imread(path)[:, :, ::-1]
        plt.figure(figsize=(16, 6))
        plt.imshow(picture)
        plt.axis("off")
        plt.title(os.path.basename(path))
        plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# Mode B: dataset evaluation
# ------------------------------------------------------------------

def run_dataset(dataset_name, datasets_root, save_details=True, ablation=False):
    samples = evaluation.find_samples(dataset_name, datasets_root)
    print("\n" + "=" * 70)
    print(f"DATASET MODE  -  {dataset_name}")
    print("=" * 70)
    if not samples:
        print(f"No images found in {os.path.join(datasets_root, dataset_name)}.")
        print("See README.md -> 'Downloading the datasets' for the expected folder layout.")
        return None

    with_gt = sum(s["groundtruth"] is not None for s in samples)
    print(f"Images found: {len(samples)}   with ground truth: {with_gt}   "
          f"segmentation only: {len(samples) - with_gt}")

    rows, ablation_scores = [], {"gaussian": [], "none": [], "sharpen": []}
    start_all = time.time()
    for index, sample in enumerate(samples, 1):
        image = read_color(sample["image"])
        if image is None:
            print(f"[{index:2d}/{len(samples)}] {sample['name']:<20} skipped - could not read image")
            continue
        if sample["fov"]:
            fov, fov_source = read_binary(sample["fov"], image.shape[:2]), "provided"
        else:
            fov, fov_source = estimate_fov_mask(image), "estimated"

        result = segment_vessels(image, fov)
        groundtruth = read_binary(sample["groundtruth"], image.shape[:2]) if sample["groundtruth"] else None
        metrics = evaluation.calculate_metrics(result["mask"], groundtruth, fov) if groundtruth is not None else None

        if ablation and metrics:
            ablation_scores["sharpen"].append(metrics)
            for variant in ("gaussian", "none"):
                variant_mask = segment_vessels(image, fov, prefilter=variant)["mask"]
                ablation_scores[variant].append(evaluation.calculate_metrics(variant_mask, groundtruth, fov))

        label = f"{sample['split']}/{sample['name']}" if sample["split"] != "all" else sample["name"]
        if save_details:
            title = f"{dataset_name} - {label}"
            save_detailed_outputs(os.path.join(PROJECT_ROOT, "outputs", dataset_name, sample["name"]),
                                  image, result, groundtruth, metrics, title)
        save_final_outputs(os.path.join(PROJECT_ROOT, "FINAL_RESULTS", dataset_name),
                           sample["name"], image, result["mask"])
        rows.append(evaluation.metrics_row(sample, fov_source, metrics))

        if metrics:
            print(f"[{index:2d}/{len(samples)}] {label:<20} Acc {metrics['Accuracy']:.4f}  "
                  f"Pre {metrics['Precision']:.4f}  Rec {metrics['Recall']:.4f}  "
                  f"Spe {metrics['Specificity']:.4f}  Dice {metrics['Dice']:.4f}  IoU {metrics['IoU']:.4f}")
        else:
            print(f"[{index:2d}/{len(samples)}] {label:<20} segmented - no ground truth, not evaluated")

    results_folder = make_folder(os.path.join(PROJECT_ROOT, "results"))
    evaluation.write_csv(os.path.join(results_folder, f"{dataset_name}_metrics.csv"), rows, evaluation.CSV_COLUMNS)
    evaluation.plot_per_image(rows, dataset_name, os.path.join(results_folder, f"{dataset_name}_per_image.png"))
    print(f"Finished {dataset_name} in {time.time() - start_all:.0f} s")

    if ablation and ablation_scores["sharpen"]:
        return {variant: {m: float(np.mean([s[m] for s in scores])) for m in evaluation.METRICS}
                for variant, scores in ablation_scores.items()}
    return {}


def update_combined_results():
    """Rebuilds metrics.csv, summary.csv/.md and the summary graph from every evaluated dataset."""
    results_folder = os.path.join(PROJECT_ROOT, "results")
    rows = []
    for name in DATASET_NAMES:
        path = os.path.join(results_folder, f"{name}_metrics.csv")
        if os.path.exists(path):
            rows += evaluation.read_csv(path)
    if not rows:
        return
    summary = evaluation.summarise(rows)
    evaluation.write_csv(os.path.join(results_folder, "metrics.csv"), rows, evaluation.CSV_COLUMNS)
    evaluation.write_csv(os.path.join(results_folder, "summary.csv"), summary, evaluation.summary_columns())
    with open(os.path.join(results_folder, "summary.md"), "w", encoding="utf-8") as file:
        file.write(evaluation.summary_markdown(summary))
    evaluation.plot_summary(summary, os.path.join(results_folder, "performance_summary.png"))
    evaluation.print_summary(summary)
    print(f"\nCSV files and graphs saved in: {results_folder}")


def save_ablation(ablation_results):
    results_folder = os.path.join(PROJECT_ROOT, "results")
    rows = []
    for dataset, variants in ablation_results.items():
        for variant, scores in variants.items():
            rows.append({"Dataset": dataset, "Prefilter": variant, **{m: round(scores[m], 4) for m in scores}})
    evaluation.write_csv(os.path.join(results_folder, "ablation.csv"), rows, ["Dataset", "Prefilter"] + evaluation.METRICS)
    evaluation.plot_ablation(rows, os.path.join(results_folder, "ablation.png"))
    print("\nABLATION - what the first filtering step changes")
    print(f"{'Dataset':<11}{'Prefilter':<11}" + "".join(f"{m[:11]:>12}" for m in evaluation.METRICS))
    for row in rows:
        print(f"{row['Dataset']:<11}{row['Prefilter']:<11}" + "".join(f"{row[m]:>12.4f}" for m in evaluation.METRICS))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Retinal blood vessel segmentation (classical image processing)")
    parser.add_argument("--mode", choices=["custom", "dataset", "all"], default="custom",
                        help="custom: images in input/, dataset: DRIVE / CHASE_DB1 evaluation, all: both")
    parser.add_argument("--dataset", choices=DATASET_NAMES + ["all"], default="all")
    parser.add_argument("--input", default=os.path.join(PROJECT_ROOT, "input"), help="folder with custom images")
    parser.add_argument("--datasets-root", default=os.path.join(PROJECT_ROOT, "datasets"))
    parser.add_argument("--no-details", action="store_true", help="skip intermediate images in outputs/")
    parser.add_argument("--ablation", action="store_true",
                        help="also compare sharpening vs Gaussian blur vs no pre-filter")
    parser.add_argument("--show", action="store_true", help="open the custom-image results in a window")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if not args.show:
        matplotlib.use("Agg")
    save_details = not args.no_details

    if args.mode in ("custom", "all"):
        run_custom(args.input, save_details, args.show)

    if args.mode in ("dataset", "all"):
        names = DATASET_NAMES if args.dataset == "all" else [args.dataset]
        ablation_results = {}
        for name in names:
            outcome = run_dataset(name, args.datasets_root, save_details, args.ablation)
            if outcome:
                ablation_results[name] = outcome
        update_combined_results()
        if ablation_results:
            save_ablation(ablation_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
