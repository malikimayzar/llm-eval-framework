import json
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

REPORTS_DIR = Path(__file__).parent
RESULTS_DIR = REPORTS_DIR / "results"
CHARTS_DIR  = REPORTS_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# Color palette 
COLOR_PERFECT    = "#2ecc71"
COLOR_PARTIAL    = "#f39c12"
COLOR_FAILED     = "#e74c3c"
COLOR_CLEAN      = "#3498db"
COLOR_DISTRACTOR = "#e67e22"
COLOR_BG         = "#1a1a2e"
COLOR_TEXT       = "#ecf0f1"
COLOR_MISTRAL    = "#3498db"   
COLOR_PHI3       = "#e74c3c"   

# Data loaders 
def load_latest(model: str, dataset: str) -> list:
    safe_model = model.replace(":", "-")
    files = sorted(RESULTS_DIR.glob(f"{safe_model}_{dataset}_*_faithfulness.json"))
    if not files:
        print(f"  [skip] Tidak ada hasil untuk '{model}' + '{dataset}'")
        return []
    print(f"  Loading: {files[-1].name}")
    return json.loads(files[-1].read_text())

def load_latest_summary(model: str, dataset: str) -> dict:
    safe_model = model.replace(":", "-")
    files = sorted(RESULTS_DIR.glob(f"{safe_model}_{dataset}_*_summary.json"))
    if not files:
        return {}
    return json.loads(files[-1].read_text())

# Chart helpers
def _style_ax(ax):
    ax.set_facecolor(COLOR_BG)
    ax.tick_params(colors=COLOR_TEXT)
    ax.spines[:].set_color("#2c3e50")
    ax.xaxis.label.set_color(COLOR_TEXT)
    ax.yaxis.label.set_color(COLOR_TEXT)
    
def _add_value_labels(ax, bars, scores, fmt="{:.2f}", offset=0.02, fontsize=10):
    for bar, s in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            fmt.format(s),
            ha="center", va="bottom",
            color=COLOR_TEXT, fontsize=fontsize, fontweight="bold",
        )
        
# Chart 1
def chart_clean(results):
    if not results:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(COLOR_BG)
    _style_ax(ax)

    ids    = [r["case_id"] for r in results]
    scores = [r["faithfulness_score"] for r in results]
    colors = [COLOR_PERFECT if s == 1.0 else COLOR_PARTIAL if s > 0 else COLOR_FAILED for s in scores]

    bars = ax.bar(ids, scores, color=colors, edgecolor="#2c3e50", linewidth=0.8)
    _add_value_labels(ax, bars, scores)

    avg = sum(scores) / len(scores)
    ax.axhline(y=0.75, color="#9b59b6", linestyle="--", linewidth=1.5, alpha=0.8, label="Threshold (0.75)")
    ax.axhline(y=avg,  color="#1abc9c", linestyle="-.", linewidth=1.5, alpha=0.9, label=f"Avg ({avg:.3f})")
    ax.legend(facecolor="#2c3e50", edgecolor="#7f8c8d", labelcolor=COLOR_TEXT, fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.set_title(
        f"Faithfulness — Mistral, Clean Dataset\n"
        f"Avg: {avg:.3f} | Perfect: {sum(1 for s in scores if s==1.0)}/{len(scores)}",
        color=COLOR_TEXT, fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Case ID", color=COLOR_TEXT)
    ax.set_ylabel("Faithfulness Score", color=COLOR_TEXT)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out = str(CHARTS_DIR / "faithfulness_clean.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close()
    print(f"  Saved: faithfulness_clean.png")


# Chart 2 
def chart_distractor(results):
    if not results:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLOR_BG)
    _style_ax(ax)

    ids    = [r["case_id"] for r in results]
    scores = [r["faithfulness_score"] for r in results]
    colors = [COLOR_PERFECT if s == 1.0 else COLOR_PARTIAL if s > 0 else COLOR_FAILED for s in scores]

    bars = ax.bar(ids, scores, color=colors, edgecolor="#2c3e50", linewidth=0.8)
    _add_value_labels(ax, bars, scores, offset=0.015, fontsize=11)

    avg = sum(scores) / len(scores)
    ax.axhline(y=avg, color="#1abc9c", linestyle="-.", linewidth=1.5, alpha=0.9, label=f"Avg ({avg:.3f})")
    ax.legend(facecolor="#2c3e50", edgecolor="#7f8c8d", labelcolor=COLOR_TEXT, fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.set_title(
        f"Faithfulness — Mistral, Distractor Dataset\n"
        f"Avg: {avg:.3f} | Manipulated context",
        color=COLOR_TEXT, fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Case ID", color=COLOR_TEXT)
    ax.set_ylabel("Faithfulness Score", color=COLOR_TEXT)
    plt.tight_layout()

    out = str(CHARTS_DIR / "faithfulness_distractor.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close()
    print(f"  Saved: faithfulness_distractor.png")


# Chart 3
def chart_comparison(clean, distractor):
    if not clean or not distractor:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(COLOR_BG)

    for ax, results, label, color in zip(
        axes,
        [clean, distractor],
        ["Clean (10 cases)", "Distractor (5 cases)"],
        [COLOR_CLEAN, COLOR_DISTRACTOR],
    ):
        _style_ax(ax)
        scores = [r["faithfulness_score"] for r in results]
        avg    = sum(scores) / len(scores)
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
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    str(count),
                    ha="center", va="bottom",
                    color=COLOR_TEXT, fontsize=14, fontweight="bold",
                )
        ax.set_title(
            f"{label}\nAvg: {avg:.3f} ({avg*100:.1f}%)",
            color=COLOR_TEXT, fontsize=12, fontweight="bold",
        )
        ax.set_ylabel("Number of Cases", color=COLOR_TEXT)
        ax.set_ylim(0, max(counts) + 2)

    fig.suptitle(
        "Mistral Faithfulness — Clean vs Distractor",
        color=COLOR_TEXT, fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    out = str(CHARTS_DIR / "comparison_clean_vs_distractor.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close()
    print(f"  Saved: comparison_clean_vs_distractor.png")


# Chart 4 

def chart_model_comparison(mistral_results: list, phi3_results: list):
    if not mistral_results or not phi3_results:
        print("  [skip] chart_model_comparison: butuh data mistral + phi3:mini")
        return

    def _compute_metrics(results: list) -> dict:
        scores     = [r["faithfulness_score"] for r in results]
        n          = len(results)
        insuf      = [r for r in results if r.get("is_insufficient_context_response") and r.get("has_failure")]
        return {
            "avg_faithfulness" : round(sum(scores) / n, 4),
            "perfect_rate"     : round(sum(1 for s in scores if s == 1.0) / n, 4),
            "failure_rate"     : round(sum(1 for r in results if r["has_failure"]) / n, 4),
            "false_insuf_rate" : round(len(insuf) / n, 4),
            "n"                : n,
        }

    m  = _compute_metrics(mistral_results)
    p  = _compute_metrics(phi3_results)

    metric_labels = [
        "Avg\nFaithfulness",
        "Perfect Score\nRate",
        "Failure\nRate",
        "False INSUF_CTX\nRate",
    ]
    mistral_vals = [m["avg_faithfulness"], m["perfect_rate"], m["failure_rate"], m["false_insuf_rate"]]
    phi3_vals    = [p["avg_faithfulness"], p["perfect_rate"], p["failure_rate"], p["false_insuf_rate"]]

    x     = np.arange(len(metric_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(COLOR_BG)
    _style_ax(ax)

    bars_m = ax.bar(x - width/2, mistral_vals, width,
                    label=f"mistral (n={m['n']})",
                    color=COLOR_MISTRAL, edgecolor="#2c3e50", linewidth=0.8)
    bars_p = ax.bar(x + width/2, phi3_vals, width,
                    label=f"phi3:mini (n={p['n']})",
                    color=COLOR_PHI3, edgecolor="#2c3e50", linewidth=0.8)
    
    for bar, val in zip(bars_m, mistral_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f"{val:.2f}", ha="center", va="bottom",
                color=COLOR_TEXT, fontsize=10, fontweight="bold")
    for bar, val in zip(bars_p, phi3_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f"{val:.2f}", ha="center", va="bottom",
                color=COLOR_TEXT, fontsize=10, fontweight="bold")
        
    for i, (mv, pv) in enumerate(zip(mistral_vals, phi3_vals)):
        higher_is_better = i < 2
        mistral_wins = (mv > pv) if higher_is_better else (mv < pv)
        winner_x = (x[i] - width/2) if mistral_wins else (x[i] + width/2)
        ax.annotate("★", xy=(winner_x, max(mv, pv) + 0.08),
                    ha="center", color="#f1c40f", fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, color=COLOR_TEXT, fontsize=10)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Score / Rate", color=COLOR_TEXT)
    ax.set_title(
        "Model Comparison: Mistral vs phi3:mini — Clean Dataset\n"
        "★ = better performer per metric  |  Lower is better for Failure & False INSUF_CTX",
        color=COLOR_TEXT, fontsize=12, fontweight="bold",
    )
    ax.legend(facecolor="#2c3e50", edgecolor="#7f8c8d", labelcolor=COLOR_TEXT, fontsize=10)
    ax.axhline(y=0.75, color="#9b59b6", linestyle="--", linewidth=1, alpha=0.5, label="Threshold")

    plt.tight_layout()
    out = str(CHARTS_DIR / "model_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close()
    print(f"  Saved: model_comparison.png")


# Chart 5 

def chart_per_case_comparison(mistral_results: list, phi3_results: list):
    if not mistral_results or not phi3_results:
        print("  [skip] chart_per_case_comparison: butuh data mistral + phi3:mini")
        return
    
    mistral_map = {r["case_id"]: r["faithfulness_score"] for r in mistral_results}
    phi3_map    = {r["case_id"]: r["faithfulness_score"] for r in phi3_results}
    common_ids  = sorted(set(mistral_map) & set(phi3_map))

    if not common_ids:
        print("  [skip] chart_per_case_comparison: tidak ada case yang sama di kedua model")
        return

    m_scores = [mistral_map[cid] for cid in common_ids]
    p_scores = [phi3_map[cid]    for cid in common_ids]
    x        = np.arange(len(common_ids))

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(COLOR_BG)
    _style_ax(ax)

    # Line plot
    ax.plot(x, m_scores, color=COLOR_MISTRAL, linewidth=2,
            marker="o", markersize=8, label="mistral", zorder=3)
    ax.plot(x, p_scores, color=COLOR_PHI3, linewidth=2,
            marker="s", markersize=8, label="phi3:mini", zorder=3)

    # Shading
    ax.fill_between(x, m_scores, p_scores,
                    where=[m > p for m, p in zip(m_scores, p_scores)],
                    alpha=0.15, color=COLOR_MISTRAL, label="mistral leads")
    ax.fill_between(x, m_scores, p_scores,
                    where=[p >= m for m, p in zip(m_scores, p_scores)],
                    alpha=0.15, color=COLOR_PHI3, label="phi3:mini leads")
    
    for i, (ms, ps) in enumerate(zip(m_scores, p_scores)):
        ax.text(i, ms + 0.04, f"{ms:.2f}", ha="center", va="bottom",
                color=COLOR_MISTRAL, fontsize=8, fontweight="bold")
        ax.text(i, ps - 0.07, f"{ps:.2f}", ha="center", va="top",
                color=COLOR_PHI3, fontsize=8, fontweight="bold")

    # Threshold line
    ax.axhline(y=0.75, color="#9b59b6", linestyle="--",
               linewidth=1.5, alpha=0.7, label="Threshold (0.75)")

    # Avg lines
    m_avg = sum(m_scores) / len(m_scores)
    p_avg = sum(p_scores) / len(p_scores)
    ax.axhline(y=m_avg, color=COLOR_MISTRAL, linestyle="-.",
               linewidth=1, alpha=0.6, label=f"mistral avg ({m_avg:.3f})")
    ax.axhline(y=p_avg, color=COLOR_PHI3, linestyle="-.",
               linewidth=1, alpha=0.6, label=f"phi3:mini avg ({p_avg:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(common_ids, rotation=45, ha="right", color=COLOR_TEXT, fontsize=9)
    ax.set_ylim(-0.05, 1.25)
    ax.set_ylabel("Faithfulness Score", color=COLOR_TEXT)
    ax.set_xlabel("Case ID", color=COLOR_TEXT)
    ax.set_title(
        "Per-Case Faithfulness: Mistral vs phi3:mini — Clean Dataset\n"
        f"Mistral avg: {m_avg:.3f} | phi3:mini avg: {p_avg:.3f} | "
        f"{len(common_ids)} common cases",
        color=COLOR_TEXT, fontsize=12, fontweight="bold",
    )
    ax.legend(facecolor="#2c3e50", edgecolor="#7f8c8d",
              labelcolor=COLOR_TEXT, fontsize=9, loc="upper right")

    plt.tight_layout()
    out = str(CHARTS_DIR / "per_case_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close()
    print(f"  Saved: per_case_comparison.png")


# Chart 6 
def chart_latency_comparison(mistral_summary: dict, phi3_summary: dict):
    m_lat = mistral_summary.get("inference", {}).get("avg_latency_seconds")
    p_lat = phi3_summary.get("inference", {}).get("avg_latency_seconds")

    if not m_lat or not p_lat:
        print("  [skip] chart_latency_comparison: data latency tidak tersedia di summary")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(COLOR_BG)
    _style_ax(ax)

    models   = ["mistral", "phi3:mini"]
    latencies = [m_lat, p_lat]
    colors   = [COLOR_MISTRAL, COLOR_PHI3]

    bars = ax.bar(models, latencies, color=colors, edgecolor="#2c3e50",
                  linewidth=0.8, width=0.4)

    for bar, lat in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f"{lat:.0f}s\n({lat/60:.1f} min)",
                ha="center", va="bottom",
                color=COLOR_TEXT, fontsize=11, fontweight="bold")

    # Speedup annotation
    speedup = m_lat / p_lat
    ax.annotate(
        f"phi3:mini is {speedup:.1f}x faster",
        xy=(1, p_lat), xytext=(0.5, max(latencies) * 0.85),
        arrowprops=dict(arrowstyle="->", color=COLOR_TEXT, lw=1.5),
        color="#f1c40f", fontsize=11, fontweight="bold", ha="center",
    )

    ax.set_ylabel("Avg Latency per Case (seconds)", color=COLOR_TEXT)
    ax.set_title(
        "Inference Latency: Mistral vs phi3:mini\nCPU-only, 16GB RAM",
        color=COLOR_TEXT, fontsize=12, fontweight="bold",
    )
    ax.set_ylim(0, max(latencies) * 1.3)

    plt.tight_layout()
    out = str(CHARTS_DIR / "latency_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close()
    print(f"  Saved: latency_comparison.png")


# Entry point
def parse_args():
    parser = argparse.ArgumentParser(description="LLM Eval Framework — Visualizer")
    parser.add_argument(
        "--chart",
        choices=["all", "existing", "model_comparison", "per_case", "latency"],
        default="all",
        help="Chart yang di-generate (default: all)",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("=" * 55)
    print("LLM Eval Framework — Visualizer")
    print("=" * 55)

    run_existing        = args.chart in ("all", "existing")
    run_model_compare   = args.chart in ("all", "model_comparison")
    run_per_case        = args.chart in ("all", "per_case")
    run_latency         = args.chart in ("all", "latency")

    # Load data
    print("\n[Loading data...]")
    mistral_clean      = load_latest("mistral", "clean")
    mistral_distractor = load_latest("mistral", "distractor")
    phi3_clean         = load_latest("phi3:mini", "clean")
    mistral_summary    = load_latest_summary("mistral", "clean")
    phi3_summary       = load_latest_summary("phi3:mini", "clean")

    # Generate charts
    print("\n[Generating charts...]")

    if run_existing:
        chart_clean(mistral_clean)
        chart_distractor(mistral_distractor)
        chart_comparison(mistral_clean, mistral_distractor)

    if run_model_compare:
        chart_model_comparison(mistral_clean, phi3_clean)

    if run_per_case:
        chart_per_case_comparison(mistral_clean, phi3_clean)

    if run_latency:
        chart_latency_comparison(mistral_summary, phi3_summary)

    print(f"\nCharts saved to: reports/charts/")
    print("=" * 55)