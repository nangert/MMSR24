from plot_results import make_plots, plot_global_coverage
import pandas as pd


def main():
    """
    Example usage of these plotting functions.
    """

    # 1) Plot aggregated evaluation results
    df_eval_results = pd.read_csv("results/mfcc/evaluation_system_results.csv")
    make_plots(df_eval_results, output_folder="results/mfcc/plots")

    # 2) Plot global coverage
    plot_global_coverage(
        csv_file="results/mfcc/global_coverage.csv",
        output_folder="results/mfcc/plots"
    )

if __name__ == "__main__":
    main()
