import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_results(input_csv: str, output_csv: str):
    """
    1) Reads the CSV file containing system/query metrics.
    2) Parses the system name -> (system_base, approach).
    3) Groups by (system_base, approach) to get average metrics over queries.
    4) Saves the grouped results as a new CSV file.
    5) Creates side-by-side bar plots (NDCG & Diversity).
    6) Creates a scatterplot with NDCG on x-axis and Diversity on y-axis.
    """

    # 1) Load the CSV
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

    # -- Function to parse system names (to handle normal, div_greedy, etc.) --
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

    # 2) Group by (system_base, approach) and average numeric metrics
    df_grouped = (
        df.groupby(["system_base", "approach"], as_index=False)[metric_cols]
          .mean()
    )

    # 3) Save results
    df_grouped.to_csv(output_csv, index=False)
    print(f"Grouped results saved to {output_csv}")

    # -----------------------------------------
    # 4) Side-by-side bar chart for reference
    # (your existing code to plot bars for NDCG & Diversity, if desired)
    # -----------------------------------------
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

    # Example usage
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

    # -----------------------------------------
    # 5) Scatterplot: NDCG vs. Diversity
    # -----------------------------------------
    # We'll do one point per row in df_grouped.
    # If you want different colors for each approach, we can define a color map:
    color_map = {
        "normal": "blue",
        "div_greedy": "red",
        "div_semi": "green",
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

        # optional: label the point
        label_str = f"{system_base}-{approach}"
        plt.annotate(label_str, (x_val, y_val+0.001),
                     fontsize=8, ha="center")

    plt.xlabel("NDCG")
    plt.ylabel("Diversity")
    plt.title("Scatterplot of NDCG vs Diversity @10")

    # optionally add a legend for the approach colors
    handles = []
    for a, color in color_map.items():
        handles.append(plt.Line2D([0], [0], marker='o', color='w', label=a,
                      markerfacecolor=color, markersize=8))
    plt.legend(handles=handles, title="Approach")

    plt.tight_layout()
    plt.savefig("results/test1/ndcg_vs_diversity_scatter.png")
    plt.show()


# ------------------------------------------------------------------------------
# Example usage if run directly
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    input_csv_path = "results/test1/evaluation_results.csv"      # your CSV file
    output_csv_path = "results/test1/analysis_results.csv"       # aggregated output
    analyze_results(input_csv_path, output_csv_path)
