import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def annotate_bars(ax):
    """
    Annotates each bar with its height value (e.g., 0.123).
    """
    for bar in ax.patches:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
            rotation=0
        )

def plot_global_coverage(csv_file, output_folder):
    """
    Reads `csv_file` (containing 'system' and 'global_coverage' columns),
    and creates a bar plot in the same style as your other metrics.

    Example file structure:
    system,global_coverage
    Baseline,1.0
    MFCC bow,0.9986402486402487
    ...
    """
    # Read the global coverage file
    df_coverage = pd.read_csv(csv_file)

    # Sort systems by coverage (descending)
    df_coverage.sort_values(by="global_coverage", ascending=False, inplace=True)

    # Create a figure
    plt.figure(figsize=(10, 6))
    sns.set_style("darkgrid")  # same Seaborn style

    # Single color (royalblue) for all bars
    plt.bar(
        df_coverage["system"],
        df_coverage["global_coverage"],
        color="royalblue"
    )

    plt.title("Global Coverage by System", fontsize=14, pad=15)
    plt.ylabel("Global Coverage", fontsize=12)
    plt.xlabel("Retrieval System", fontsize=12)

    # Because coverage is typically in [0, 1], you can limit the y-axis if you like:
    # plt.ylim(0, 1.05)

    # Rotate system labels
    plt.xticks(rotation=45, ha="right", fontsize=10)

    # Annotate bars with their coverage values
    annotate_bars(plt.gca())

    # Tight layout and save
    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "global_coverage_by_system.png")
    plt.savefig(output_path, dpi=150)
    plt.show()

def plot_metric(df_results, metric, output_folder):
    """
    Creates a bar plot for a given metric, grouped by 'system',
    using a single color (royalblue).
    """
    grouped = df_results.groupby("system")[metric].mean().reset_index()
    grouped.sort_values(by=metric, ascending=False, inplace=True)

    plt.figure(figsize=(10, 6))
    sns.set_style("darkgrid")

    plt.bar(
        grouped["system"],
        grouped[metric],
        color="royalblue"
    )

    plt.title(f"Average {metric.upper()} by System", fontsize=14, pad=15)
    plt.ylabel(metric.upper(), fontsize=12)
    plt.xlabel("Retrieval System", fontsize=12)

    # Annotate bars
    annotate_bars(plt.gca())

    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"{metric}_by_system.png"), dpi=150)
    plt.show()

def make_plots(df_results, output_folder):
    """
    Plots standard metrics (precision_at_k, recall_at_k, etc.)
    and beyond-accuracy metrics using a single bar color (royalblue).
    """
    sns.set_style("darkgrid")

    # 1) Plot standard metrics
    for metric in ["precision", "recall", "NDCG", "MRR"]:
        plot_metric(df_results, metric, output_folder)

    # 2) Plot beyond-accuracy metrics
    for metric in ["diversity", "novelty", "coverage", "serendipity"]:
        plot_metric(df_results, metric, output_folder)

