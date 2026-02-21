"""
visualize.py
------------
Generate visualisasi dari hasil evaluasi.

Jalankan:
    python reports/visualize.py
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

REPORTS_DIR = Path(__file__).parent
RESULTS_DIR = REPORTS_DIR / "results"
CHARTS_DIR  = REPORTS_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

COLOR_PERFECT    = "#2ecc71"
COLOR_PARTIAL    = "#f39c12"
COLOR_FAILED     = "#e74c3c"
COLOR_CLEAN      = "#3498db"
COLOR_DISTRACTOR = "#e67e22"
COLOR_BG         = "#1a1a2e"
COLOR_TEXT       = "#ecf0f1"

def load_latest(dataset):
    files = sorted(RESULTS_DIR.glob(f"mistral_{dataset}_*_faithfulness.json"))
    if not files:
        print(f"Tidak ada hasil untuk '{dataset}'")
        return []
    print(f"Loading: {files[-1].name}")
    return json.loads(files[-1].read_text())

def chart_clean(results):
    if not results: return
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ids    = [r["case_id"] for r in results]
    scores = [r["faithfulness_score"] for r in results]
    colors = [COLOR_PERFECT if s==1.0 else COLOR_PARTIAL if s>0 else COLOR_FAILED for s in scores]
    bars = ax.bar(ids, scores, color=colors, edgecolor="#2c3e50", linewidth=0.8)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f"{s:.2f}", ha="center", va="bottom", color=COLOR_TEXT, fontsize=10, fontweight="bold")
    avg = sum(scores)/len(scores)
    ax.axhline(y=0.75, color="#9b59b6", linestyle="--", linewidth=1.5, alpha=0.8, label="Threshold (0.75)")
    ax.axhline(y=avg,  color="#1abc9c", linestyle="-.", linewidth=1.5, alpha=0.9, label=f"Avg ({avg:.3f})")
    ax.legend(facecolor="#2c3e50", edgecolor="#7f8c8d", labelcolor=COLOR_TEXT, fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.set_title(f"Faithfulness — Mistral, Clean Dataset\nAvg: {avg:.3f} | Perfect: {sum(1 for s in scores if s==1.0)}/{len(scores)}",
                 color=COLOR_TEXT, fontsize=13, fontweight="bold")
    ax.set_xlabel("Case ID", color=COLOR_TEXT)
    ax.set_ylabel("Faithfulness Score", color=COLOR_TEXT)
    ax.tick_params(colors=COLOR_TEXT)
    ax.spines[:].set_color("#2c3e50")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = str(CHARTS_DIR / "faithfulness_clean.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close()
    print(f"Saved: {out}")

def chart_distractor(results):
    if not results: return
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ids    = [r["case_id"] for r in results]
    scores = [r["faithfulness_score"] for r in results]
    colors = [COLOR_PERFECT if s==1.0 else COLOR_PARTIAL if s>0 else COLOR_FAILED for s in scores]
    bars = ax.bar(ids, scores, color=colors, edgecolor="#2c3e50", linewidth=0.8)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
                f"{s:.2f}", ha="center", va="bottom", color=COLOR_TEXT, fontsize=11, fontweight="bold")
    avg = sum(scores)/len(scores)
    ax.axhline(y=avg, color="#1abc9c", linestyle="-.", linewidth=1.5, alpha=0.9, label=f"Avg ({avg:.3f})")
    ax.legend(facecolor="#2c3e50", edgecolor="#7f8c8d", labelcolor=COLOR_TEXT, fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.set_title(f"Faithfulness — Mistral, Distractor Dataset\nAvg: {avg:.3f} | Manipulated context",
                 color=COLOR_TEXT, fontsize=13, fontweight="bold")
    ax.set_xlabel("Case ID", color=COLOR_TEXT)
    ax.set_ylabel("Faithfulness Score", color=COLOR_TEXT)
    ax.tick_params(colors=COLOR_TEXT)
    ax.spines[:].set_color("#2c3e50")
    plt.tight_layout()
    out = str(CHARTS_DIR / "faithfulness_distractor.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close()
    print(f"Saved: {out}")

def chart_comparison(clean, distractor):
    if not clean or not distractor: return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(COLOR_BG)
    for ax, results, label, color in zip(
        axes,
        [clean, distractor],
        ["Clean (10 cases)", "Distractor (5 cases)"],
        [COLOR_CLEAN, COLOR_DISTRACTOR]
    ):
        ax.set_facecolor(COLOR_BG)
        scores = [r["faithfulness_score"] for r in results]
        avg = sum(scores)/len(scores)
        counts = [
            sum(1 for s in scores if s == 0.0),
            sum(1 for s in scores if 0 < s < 1.0),
            sum(1 for s in scores if s == 1.0),
        ]
        labels_bin = ["Failed\n(0.0)", "Partial\n(0–1)", "Perfect\n(1.0)"]
        bar_colors = [COLOR_FAILED, COLOR_PARTIAL, COLOR_PERFECT]
        bars = ax.bar(labels_bin, counts, color=bar_colors, edgecolor="#2c3e50", linewidth=0.8)
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                        str(count), ha="center", va="bottom",
                        color=COLOR_TEXT, fontsize=14, fontweight="bold")
        ax.set_title(f"{label}\nAvg: {avg:.3f} ({avg*100:.1f}%)",
                     color=COLOR_TEXT, fontsize=12, fontweight="bold")
        ax.set_ylabel("Number of Cases", color=COLOR_TEXT)
        ax.tick_params(colors=COLOR_TEXT)
        ax.spines[:].set_color("#2c3e50")
        ax.set_ylim(0, max(counts)+2)
    fig.suptitle("Mistral Faithfulness — Clean vs Distractor",
                 color=COLOR_TEXT, fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = str(CHARTS_DIR / "comparison_clean_vs_distractor.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close()
    print(f"Saved: {out}")

if __name__ == "__main__":
    print("=" * 50)
    print("LLM Eval Framework — Visualizer")
    print("=" * 50)
    clean      = load_latest("clean")
    distractor = load_latest("distractor")
    print()
    chart_clean(clean)
    chart_distractor(distractor)
    chart_comparison(clean, distractor)
    print()
    print("Charts saved to: reports/charts/")