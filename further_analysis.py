import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def analyze_results(input_csv: str, output_csv: str):
    """
    Reads the CSV file containing system/query metrics.
    Parses the system name -> (system_base, approach).
    Groups by (system_base, approach) to get average metrics over queries.
    Saves the grouped results as a new CSV file.
    Creates side-by-side bar plots (NDCG & Diversity).
    Creates a scatterplot with NDCG on x-axis and Diversity on y-axis.
    """

    # Load the CSV
    df = pd.read_csv(input_csv)

    # List of metric columns to average
    metric_cols = [
        "precision_at_k",
        "recall_at_k",
        "ndcg_at_k",
        "mrr",
        "diversity",
        "novelty",
        "coverage",
        "serendipity",
    ]

    def parse_system_name(sys_name: str):
        if sys_name.endswith("_div_greedy"):
            return sys_name.replace("_div_greedy", ""), "div_greedy"
        elif sys_name.endswith("_div_semi"):
            return sys_name.replace("_div_semi", ""), "div_semi"
        elif sys_name.endswith("_div_cluster"):
            return sys_name.replace("_div_cluster", ""), "div_cluster"
        else:
            return sys_name, "normal"

    # Create two new columns in df
    df["system_base"], df["approach"] = zip(*df["system"].apply(parse_system_name))

    # Group by (system_base, approach) and average numeric metrics
    df_grouped = (
        df.groupby(["system_base", "approach"], as_index=False)[metric_cols]
          .mean()
    )

    # Save results
    df_grouped.to_csv(output_csv, index=False)
    print(f"Grouped results saved to {output_csv}")

    # ide-by-side bar chart for reference
    def side_by_side_plot(df_input, value_col, title_str, ylabel_str, savefig_name):
        df_pivot = df_input.pivot(index="system_base", columns="approach", values=value_col).fillna(0)
        df_pivot.sort_index(inplace=True)

        approach_order = ["normal", "div_greedy", "div_semi", "div_cluster"]
        approach_order = [a for a in approach_order if a in df_pivot.columns]

        system_bases = df_pivot.index.tolist()
        x = np.arange(len(system_bases))
        width = 0.2

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, approach in enumerate(approach_order):
            offset = (i - (len(approach_order)-1)/2) * width
            values = df_pivot[approach].values
            rects = ax.bar(x + offset, values, width, label=approach)
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f"{height:.3f}",
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

        ax.set_title(title_str)
        ax.set_ylabel(ylabel_str)
        ax.set_xticks(x)
        ax.set_xticklabels(system_bases, rotation=45, ha="right")
        ax.legend()
        plt.tight_layout()
        plt.savefig(savefig_name)
        plt.show()

    side_by_side_plot(
        df_input=df_grouped[["system_base", "approach", "ndcg_at_k"]],
        value_col="ndcg_at_k",
        title_str="NDCG by System & Approach",
        ylabel_str="NDCG",
        savefig_name="ndcg_comparison.png"
    )

    side_by_side_plot(
        df_input=df_grouped[["system_base", "approach", "diversity"]],
        value_col="diversity",
        title_str="Diversity by System & Approach",
        ylabel_str="Diversity",
        savefig_name="diversity_comparison.png"
    )

    # Scatterplot: NDCG vs. Diversity
    color_map = {
        "normal": "blue",
        "div_greedy": "darkorange",
        "div_semi": "cyan",
        "div_cluster": "purple",
    }

    plt.figure(figsize=(8, 6))
    for i, row in df_grouped.iterrows():
        x_val = row["ndcg_at_k"]
        y_val = row["diversity"]
        approach = row["approach"]
        system_base = row["system_base"]

        # pick color
        c = color_map.get(approach, "black")

        # plot a single point
        plt.scatter(x_val, y_val, color=c, s=50, alpha=0.7)

        # label the point
        label_str = f"{system_base}-{approach}"
        plt.annotate(label_str, (x_val, y_val+0.001),
                     fontsize=8, ha="center")

    plt.xlabel("NDCG")
    plt.ylabel("Diversity")
    plt.title("Scatterplot of NDCG vs Diversity @20")

    # optionally add a legend for the approach colors
    handles = []
    for a, color in color_map.items():
        handles.append(plt.Line2D([0], [0], marker='o', color='w', label=a,
                      markerfacecolor=color, markersize=8))
    plt.legend(handles=handles, title="Approach")

    plt.tight_layout()
    plt.savefig("results/100q20n/ndcg_vs_diversity_scatter.png")
    plt.show()


if __name__ == "__main__":
    input_csv_path = "results/100q20n/evaluation_results.csv"
    output_csv_path = "results/100q20n/analysis_results.csv"
    analyze_results(input_csv_path, output_csv_path)
