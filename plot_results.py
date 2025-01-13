import os
import matplotlib.pyplot as plt
import seaborn as sns


def annotate_bars(ax):
    """
    Annotates each bar with its height value (e.g. 0.123).
    """
    for bar in ax.patches:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,  # tweak font size if you want
            color="black",
            rotation=0  # you can rotate values if they overlap
        )


def plot_metric(df_results, metric, output_folder):
    """
    Creates a bar plot for a given metric, grouped by 'system'.
    """
    # Group the data by 'system' and compute the mean of the metric
    grouped = df_results.groupby("system")[metric].mean().reset_index()

    # Sort systems if you want a specific order; otherwise you can skip
    grouped.sort_values(by=metric, ascending=False, inplace=True)

    # Create a figure and axis with a larger size
    plt.figure(figsize=(10, 6))

    # We'll use a color palette from Seaborn
    # If you have many systems, use e.g. 'tab20' or 'hsv'
    palette = sns.color_palette("tab10", len(grouped))

    # Bar plot
    bars = plt.bar(
        grouped["system"],
        grouped[metric],
        color=palette
    )

    plt.title(f"Average {metric.upper()} by System", fontsize=14, pad=15)
    plt.ylabel(metric.upper(), fontsize=12)
    plt.xlabel("Retrieval System", fontsize=12)

    # Optionally, you might want to limit the y-axis if your metrics go up to 1
    # plt.ylim(0, 1)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha="right", fontsize=10)

    # Annotate bars
    annotate_bars(plt.gca())

    # Tweak layout and save
    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"{metric}_by_system.png"), dpi=150)
    plt.show()


def make_plots(df_results, output_folder):
    """
    Plots standard metrics (precision_at_k, recall_at_k, etc.)
    and beyond-accuracy metrics in a visually improved style.
    """
    # 1) Use Seaborn style
    sns.set_style("darkgrid")

    # 2) Plot standard metrics
    for metric in ["precision_at_k", "recall_at_k", "ndcg_at_k", "mrr"]:
        plot_metric(df_results, metric, output_folder)

    # 3) Plot beyond-accuracy metrics
    for metric in ["diversity", "novelty", "coverage", "serendipity"]:
        plot_metric(df_results, metric, output_folder)
