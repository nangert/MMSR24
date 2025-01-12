import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_results(input_csv: str, output_csv: str):
    """
    1) Reads the CSV file containing system/query metrics.
    2) Groups by 'system' to get one row per system (averaging over queries).
       Excludes non-numeric columns (like query_id).
    3) Saves the grouped results as a new CSV file.
    4) Creates a plot comparing ndcg_at_k and diversity for normal systems vs. _div systems.
    """

    # 1) Load the CSV
    df = pd.read_csv(input_csv)

    # List of metric columns you want to average
    metric_cols = [
        "precision_at_k",
        "recall_at_k",
        "ndcg_at_k",
        "mrr",
        "diversity",
        "novelty",
        "coverage",
        "serendipity"
    ]

    # 2) Group by 'system' and average only the metric columns
    #    This will drop columns like 'query_id' automatically.
    df_grouped = (
        df.groupby("system", as_index=False)[metric_cols]
          .mean()  # numeric columns only
    )

    # 3) Save results
    df_grouped.to_csv(output_csv, index=False)
    print(f"Grouped results saved to {output_csv}")

    # ------------------------------------------------
    # 4) Plot ndcg_at_k & diversity: normal vs. _div
    # ------------------------------------------------

    # Separate normal vs. _div systems
    df_normal = df_grouped[~df_grouped["system"].str.endswith("_div")].copy()
    df_div = df_grouped[df_grouped["system"].str.endswith("_div")].copy()

    # Create a "system_base" column
    df_div["system_base"] = df_div["system"].str.replace("_div", "", regex=False)
    df_normal["system_base"] = df_normal["system"]

    # Merge on that base system name, so we can pair normal vs. div
    df_compare = pd.merge(
        df_normal[["system_base", "ndcg_at_k", "diversity"]],
        df_div[["system_base", "ndcg_at_k", "diversity"]],
        on="system_base",
        suffixes=("_normal", "_div")
    )

    # Sort by system_base alphabetically (optional)
    df_compare.sort_values(by="system_base", inplace=True)

    # Now we have columns:
    # [system_base, ndcg_at_k_normal, diversity_normal, ndcg_at_k_div, diversity_div]

    # Plot side-by-side bar chart for NDCG and diversity
    x = np.arange(len(df_compare))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    # --- Left subplot: NDCG ---
    rects1 = ax[0].bar(x - width/2, df_compare["ndcg_at_k_normal"], width, label="Normal")
    rects2 = ax[0].bar(x + width/2, df_compare["ndcg_at_k_div"], width, label="Diversity")

    ax[0].set_ylabel("NDCG")
    ax[0].set_title("NDCG by System (Normal vs. Div)")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(df_compare["system_base"], rotation=45, ha="right")
    ax[0].legend()

    # Annotate bars for NDCG
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax[0].annotate(f'{height:.3f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    # --- Right subplot: Diversity ---
    rects3 = ax[1].bar(x - width/2, df_compare["diversity_normal"], width, label="Normal")
    rects4 = ax[1].bar(x + width/2, df_compare["diversity_div"], width, label="Diversity")

    ax[1].set_ylabel("Diversity")
    ax[1].set_title("Diversity by System (Normal vs. Div)")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(df_compare["system_base"], rotation=45, ha="right")
    ax[1].legend()

    # Annotate bars for diversity
    for rect in rects3 + rects4:
        height = rect.get_height()
        ax[1].annotate(f'{height:.3f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig("ndcg_diversity_comparison.png")
    plt.show()

# ------------------------------------------------------------------------------
# Example usage if you run this script directly
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    input_csv_path = "results/evaluation_results.csv"     # your existing CSV file
    output_csv_path = "results/analysis_results.csv"      # new grouped file
    analyze_results(input_csv_path, output_csv_path)
