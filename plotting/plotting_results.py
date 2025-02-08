from plot_results import make_plots, plot_global_coverage
import pandas as pd


def main():
    # 1) Plot aggregated evaluation results
    df_eval_results = pd.read_csv("../results/full10n/evaluation_system_results.csv")
    make_plots(df_eval_results, output_folder="results/full10n/plots")

    # 2) Plot global coverage
    plot_global_coverage(
        csv_file="../results/full10n/global_coverage.csv",
        output_folder="results/full10n/plots"
    )

if __name__ == "__main__":
    main()
