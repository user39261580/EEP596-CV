"""Utility script to regenerate HW7 task outputs (4-9) and
assemble an HTML report for quick PDF conversion."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

# Use a non-interactive backend so the script can run headless.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import Assignment7 as hw7  # noqa: E402  # import after matplotlib backend selection

ROOT = Path(__file__).parent
REPORT_DIR = ROOT / "report" / "task_outputs"
ASSET_DIR = REPORT_DIR / "assets"


def reset_assets_dir() -> None:
    """Clear and recreate the assets directory to avoid stale figures."""
    if ASSET_DIR.exists():
        shutil.rmtree(ASSET_DIR)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)


def save_line_plot(values: List[float], title: str, asset_name: str) -> str:
    asset_path = ASSET_DIR / asset_name
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(values)), values, marker="o")
    plt.xlabel("Disparity (pixels)")
    plt.ylabel("Cost")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(asset_path, dpi=200)
    plt.close()
    return f"assets/{asset_name}"


def save_heatmap(array: np.ndarray, title: str, asset_name: str, cmap: str = "viridis") -> str:
    asset_path = ASSET_DIR / asset_name
    plt.figure(figsize=(5, 5))
    plt.imshow(array, cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(asset_path, dpi=200)
    plt.close()
    return f"assets/{asset_name}"


def describe(values: Sequence[float] | np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def list_to_pretty_string(values: List[float]) -> str:
    arr = np.array(values)
    return np.array2string(arr, separator=", ", formatter={"float_kind": lambda x: f"{x:.4f}"})


def count_ply_vertices(ply_path: Path) -> int:
    vertex_count = 0
    with ply_path.open("r", encoding="utf-8") as f:
        in_header = True
        for line in f:
            stripped = line.strip()
            if in_header:
                if stripped == "end_header":
                    in_header = False
                continue
            vertex_count += 1
    return vertex_count


def build_html(report_payload: Dict[str, Dict]) -> Path:
    html_path = REPORT_DIR / "hw7_tasks4-9_report.html"
    sections = []
    for key in ["task4", "task5", "task6", "task7", "task8", "task9"]:
        data = report_payload[key]
        section = [f"<section id='{key}'>", f"  <h2>{data['title']}</h2>"]
        if "image" in data:
            section.append(f"  <img src='{data['image']}' alt='{data['title']}' style='max-width: 720px; width: 100%;'>")
        if "values" in data:
            section.append("  <pre>" + data["values"] + "</pre>")
        if "stats" in data:
            stats = data["stats"]
            section.append(
                "  <p>min: {min:.4f} | max: {max:.4f} | mean: {mean:.4f} | std: {std:.4f}</p>".format(**stats)
            )
        if "details" in data:
            section.append(f"  <p>{data['details']}</p>")
        section.append("</section>")
        sections.append("\n".join(section))

    html = f"""<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <title>EEP596 HW7 Tasks 4-9 Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; }}
    h1 {{ text-align: center; }}
    section {{ margin-bottom: 48px; }}
    img {{ border: 1px solid #ccc; padding: 8px; background: #fafafa; }}
    pre {{ background: #111; color: #f1f1f1; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>EEP596 HW7 Tasks 4-9 Output</h1>
  {''.join(sections)}
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return html_path


def main() -> None:
    reset_assets_dir()
    tb_left = hw7.load_image_in_grayscale("tsukuba_left.png")
    tb_right = hw7.load_image_in_grayscale("tsukuba_right.png")
    if tb_left is None or tb_right is None:
        raise FileNotFoundError("Tsukuba stereo pair is missing from the hw7 directory.")

    payload: Dict[str, Dict] = {}

    # Task 4: Smoothed auto-correlation values and plot.
    smoothed_auto = hw7.smoothing(tb_right)
    payload["task4"] = {
        "title": "Task 4 – Smoothed Auto-correlation",
        "image": save_line_plot(smoothed_auto, "Task 4 Smoothed Auto-correlation", "task4_smoothed_auto.png"),
        "values": list_to_pretty_string(smoothed_auto),
        "stats": describe(smoothed_auto),
    }

    # Task 5: Cross-correlation values and plot.
    cross_corr = hw7.cross_correlation(tb_left, tb_right)
    payload["task5"] = {
        "title": "Task 5 – Cross-correlation",
        "image": save_line_plot(cross_corr, "Task 5 Cross-correlation", "task5_cross_correlation.png"),
        "values": list_to_pretty_string(cross_corr),
        "stats": describe(cross_corr),
    }

    # Task 6: Left-right disparity map visualization.
    disp_lr = hw7.disparity_map(tb_left, tb_right)
    payload["task6"] = {
        "title": "Task 6 – Left/Right Disparity Map",
        "image": save_heatmap(disp_lr, "Task 6 Disparity", "task6_disparity.png"),
        "stats": describe(disp_lr),
    }

    # Task 7: Right-left disparity map visualization.
    disp_rl = hw7.right_left_disparity(tb_left, tb_right)
    payload["task7"] = {
        "title": "Task 7 – Right/Left Disparity Map",
        "image": save_heatmap(disp_rl, "Task 7 Disparity", "task7_disparity.png"),
        "stats": describe(disp_rl),
    }

    # Task 8: Cleaned disparity check visualization.
    cleaned = hw7.disparity_check(tb_left, tb_right)
    payload["task8"] = {
        "title": "Task 8 – Left/Right Consistency Check",
        "image": save_heatmap(cleaned, "Task 8 Cleaned Disparity", "task8_cleaned_disparity.png"),
        "stats": describe(cleaned),
    }

    # Task 9: 3D reconstruction summary.
    ply_path = Path(hw7.reconstruction(tb_left, tb_right))
    if not ply_path.is_absolute():
        ply_path = ROOT / ply_path
    payload["task9"] = {
        "title": "Task 9 – 3D Reconstruction",
        "details": f"PLY saved to {ply_path} with {count_ply_vertices(ply_path)} vertices.",
    }

    html_path = build_html(payload)
    print(f"Report generated at {html_path}")


if __name__ == "__main__":
    main()
