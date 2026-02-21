"""
visualize.py
------------
Generate visualisasi dari hasil evaluasi.

Output:
    reports/charts/faithfulness_clean.png
    reports/charts/faithfulness_distractor.png
    reports/charts/comparison_clean_vs_distractor.png

Jalankan:
    python reports/visualize.py
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — tidak butuh display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPORTS_DIR = Path(__file__).parent
RESULTS_DIR = REPORTS_DIR / "results"
CHARTS_DIR  = REPORTS_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# Warna konsisten
COLOR_PERFECT    = "#2ecc71"   # hijau — score 1.0
COLOR_PARTIAL    = "#f39c12"   # oranye — score 0 < x < 1
COLOR_FAILED     = "#e74c3c"   # merah — score 0.0
COLOR_CLEAN      = "#3498db"   # biru — clean dataset
COLOR_DISTRACTOR = "#e67e22"   # oranye tua — distractor dataset
COLOR_BG         = "#1a1a2e"   # background gelap
COLOR_TEXT       = "#ecf0f1"   # teks terang


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_latest_faithfulness(dataset: str) -> list:
    """Load file faithfulness JSON terbaru untuk dataset tertentu."""
    pattern = f"mistral_{dataset}_"
    files = sorted([
        f for f in RESULTS_DIR.glob(f"{pattern}*_faithfulness.json")
    ])
    if not files:
        print(f"⚠ Tidak ada hasil untuk dataset '{dataset}'")
        return []

    latest = files[-1]
    print(f"Loading: {latest.name}")
    with open(latest) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Chart 1: Faithfulness per case — clean dataset
# ---------------------------------------------------------------------------

def chart_faithfulness_clean(results: list) -> str:
    if not results:
        return ""

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    case_ids = [r["case_id"] for r in results]
    scores   = [r["faithfulness_score"] for r in results]
    colors   = [
        COLOR_PERFECT if s == 1.0 else COLOR_PARTIAL if s > 0 else COLOR_FAILED
        for s in scores
    ]

    bars = ax.bar(case_ids, scores, color=colors, edgecolor="#2c3e50", linewidth=0.8)

    # Tambah nilai di atas bar
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{score:.2f}",
            ha="center", va="bottom",
            color=COLOR_TEXT, fontsize=10, fontweight="bold"
        )

    # Threshold line
    ax.axhline(y=0.75, color="#9b59b6", linestyle="--", linewidth=1.5,
               label="Evidence threshold (0.75)", alpha=0.8)

    # Avg line
    avg = sum(scores) / len(scores)
    ax.axhline(y=avg, color="#1abc9c", linestyle="-.", linewidth=1.5,
               label=f"Avg score ({avg:.3f})", alpha=0.9)

    # Legend
    legend_patches = [
        mpatches.Patch(color=COLOR_PERFECT, label="Perfect (1.0)"),
        mpatches.Patch(color=COLOR_PARTIAL, label="Partial"),
        mpatches.Patch(color=COLOR_FAILED, label="Failed (0.0)"),
    ]
    ax.legend(
        handles=legend_patches +
                [plt.Line2D([0], [0], color="#9b59b6", linestyle="--", label="Threshold"),
                 plt.Line2D([0], [0], color="#1abc9c", linestyle="-.", label=f"Avg ({avg:.3f})")],
        loc="upper right", facecolor="#2c3e50", edgecolor="#7f8c8d",
        labelcolor=COLOR_TEXT, fontsize=9
    )

    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Case ID", color=COLOR_TEXT, fontsize=11)
    ax.set_ylabel("Faithfulness Score", color=COLOR_TEXT, fontsize=11)
    ax.set_title(
        "Faithfulness Score per Case — Mistral, Clean Dataset\n"
        f"Avg: {avg:.3f} | Perfect: {sum(1 for s in scores if s==1.0)}/{len(scores)}",
        color=COLOR_TEXT, fontsize=13, fontweight="bold", pad=15
    )
    ax.tick_params(colors=COLOR_TEXT)
    ax.spines[:].set_color("#2c3e50")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output = str(CHARTS_DIR / "faithfulness_clean.png")
    plt.savefig(output, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close()
    print(f"✅ Saved: {output}")
    return output


# ---------------------------------------------------------------------------
# Chart 2: Faithfulness per case — distractor dataset
# ---------------------------------------------------------------------------

def chart_faithfulness_distractor(results: list) -> str:
    if not results:
        return ""

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    case_ids = [r["case_id"] for r in results]
    scores   = [r["faithfulness_score"] for r in results]
    colors   = [
        COLOR_PERFECT if s == 1.0 else COLOR_PARTIAL if s > 0 else COLOR_FAILED
        for s in scores
    ]

    bars = ax.bar(case_ids, scores, color=colors, edgecolor="#2c3e50", linewidth=0.8)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{score:.2f}",
            ha="center", va="bottom",
            color=COLOR_TEXT, fontsize=11, fontweight="bold"
        )

    avg = sum(scores) / len(scores)
    ax.axhline(y=avg, color="#1abc9c", linestyle="-.", linewidth=1.5,
               label=f"Avg ({avg:.3f})", alpha=0.9)

    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Case ID", color=COLOR_TEXT, fontsize=11)
    ax.set_ylabel("Faithfulness Score", color=COLOR_TEXT, fontsize=11)
    ax.set_title(
        "Faithfulness Score per Case — Mistral, Distractor Dataset\n"
        f"Avg: {avg:.3f} | All cases manipulated context",
        color=COLOR_TEXT, fontsize=13, fontweight="bold", pad=15
    )
    ax.tick_params(colors=COLOR_TEXT)
    ax.spines[:].set_color("#2c3e50")

    legend_patches = [
        mpatches.Patch(color=COLOR_PARTIAL, label="Partial"),
        mpatches.Patch(color=COLOR_FAILED, label="Failed (0.0)"),
        plt.Line2D([0], [0], color="#1abc9c", linestyle="-.", label=f"Avg ({avg:.3f})"),
    ]
    ax.legend(handles=legend_patches, loc="upper right",
              facecolor="#2c3e50", edgecolor="#7f8c8d",
              labelcolor=COLOR_TEXT, fontsize=9)

    plt.tight_layout()
    output = str(CHARTS_DIR / "faithfulness_distractor.png")
    plt.savefig(output, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close()
    print(f"✅ Saved: {output}")
    return output


# ---------------------------------------------------------------------------
# Chart 3: Clean vs Distractor comparison
# ---------------------------------------------------------------------------

def chart_comparison(clean_results: list, distractor_results: list) -> str:
    if not clean_results or not distractor_results:
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(COLOR_BG)

    datasets = {
        "Clean\n(10 cases)": clean_results,
        "Distractor\n(5 cases)": distractor_results,
    }
    colors_ds = [COLOR_CLEAN, COLOR_DISTRACTOR]

    for ax, (label, results), color in zip(axes, datasets.items(), colors_ds):
        ax.set_facecolor(COLOR_BG)
        scores = [r["faithfulness_score"] for r in results]
        avg = sum(scores) / len(scores)

        # Score distribution
        bins = [0, 0.001, 0.5, 0.999, 1.001]
        labels_bin = ["0.0\n(Failed)", "0–0.5\n(Partial)", "0.5–1.0\n(Partial)", "1.0\n(Perfect)"]
        counts = [0, 0, 0, 0]
        for s in scores:
            if s == 0.0:
                counts[0] += 1
            elif s < 0.5:
                counts[1] += 1
            elif s < 1.0:
                counts[2] += 1
            else:
                counts[3] += 1

        bar_colors = [COLOR_FAILED, COLOR_PARTIAL, COLOR_PARTIAL, COLOR_PERFECT]
        bars = ax.bar(labels_bin, counts, color=bar_colors,
                      edgecolor="#2c3e50", linewidth=0.8, alpha=0.9)

        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    str(count),
                    ha="center", va="bottom",
                    color=COLOR_TEXT, fontsize=13, fontweight="bold"
                )

        ax.set_title(
            f"{label}\nAvg Score: {avg:.3f} ({avg*100:.1f}%)",
            color=COLOR_TEXT, fontsize=12, fontweight="bold"
        )
        ax.set_ylabel("Number of Cases", color=COLOR_TEXT)
        ax.tick_params(colors=COLOR_TEXT)
        ax.spines[:].set_color("#2c3e50")
        ax.set_ylim(0, max(counts) + 1.5)

    fig.suptitle(
        "Mistral Faithfulness — Clean vs Distractor Dataset",
        color=COLOR_TEXT, fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    output = str(CHARTS_DIR / "comparison_clean_vs_distractor.png")
    plt.savefig(output, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close()
    print(f"✅ Saved: {output}")
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("LLM Eval Framework — Visualizer")
    print("=" * 50)

    clean_results      = load_latest_faithfulness("clean")
    distractor_results = load_latest_faithfulness("distractor")

    print()
    chart_faithfulness_clean(clean_results)
    chart_faithfulness_distractor(distractor_results)
    chart_comparison(clean_results, distractor_results)

    print()
    print("=" * 50)
    print(f"Charts saved to: reports/charts/")
    print("Tambahkan ke README:")
    print("  ![Clean](reports/charts/faithfulness_clean.png)")
    print("  ![Distractor](reports/charts/faithfulness_distractor.png)")
    print("  ![Comparison](reports/charts/comparison_clean_vs_distractor.png)")
    print("=" * 50)