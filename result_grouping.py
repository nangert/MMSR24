import pandas as pd
import matplotlib.pyplot as plt
import os

class DataPlotter:
    def __init__(self,
                 global_coverage_csv="/results/full/global_coverave.csv",
                 evaluation_results_csv="/results/full/evaluation_results.csv"):
        """
        Initialize the DataPlotter with paths to the necessary CSV files.
        """
        self.global_coverage_csv = global_coverage_csv
        self.evaluation_results_csv = evaluation_results_csv

    def plot_global_coverage(self, output_filename="global_coverage_plot.png"):
        """
        Reads `global_coverave.csv` and plots the coverage by system.
        Saves the resulting plot as `global_coverage_plot.png` by default.
        """
        # Read data
        df_coverage = pd.read_csv(self.global_coverage_csv)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(df_coverage['system'], df_coverage['global_coverage'], color='#cce5ff')  # light color
        ax.set_title("Global Coverage by System")
        ax.set_xlabel("System")
        ax.set_ylabel("Global Coverage")
        ax.set_ylim([0, 1.1])  # optionally set y-limits a bit above 1

        # Rotate x-labels for better readability if needed
        plt.xticks(rotation=45, ha='right')

        # Layout and save
        plt.tight_layout()
        plt.savefig(output_filename, dpi=150)
        plt.close()

    def group_evaluation_results(self,
                                 grouped_output_filename="evaluation_system_results.csv"):
        """
        Reads `evaluation_results.csv`, groups by system, averages metrics,
        and saves the aggregated data to `evaluation_system_results.csv`.
        """
        # Read the data
        df_eval = pd.read_csv(self.evaluation_results_csv)

        # List of metrics to be averaged
        metrics = [
            'precision_at_k', 'recall_at_k', 'ndcg_at_k', 'mrr',
            'diversity', 'novelty', 'coverage', 'serendipity'
        ]

        # Group by 'system' and compute mean for the given metrics
        df_grouped = df_eval.groupby('system')[metrics].mean().reset_index()

        # Save to CSV
        df_grouped.to_csv(grouped_output_filename, index=False)
        return df_grouped

    def plot_evaluation_histograms(self,
                                   grouped_data_csv="evaluation_system_results.csv",
                                   output_folder="histograms"):
        """
        Reads the aggregated data (from the grouped CSV file) and creates histograms
        for each metric, using very light colors. Each histogram is saved in a
        specified output folder.
        """
        # Make sure output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Read the grouped data
        df_grouped = pd.read_csv(grouped_data_csv)

        # The metrics we want to plot
        metrics = [
            'precision_at_k', 'recall_at_k', 'ndcg_at_k', 'mrr',
            'diversity', 'novelty', 'coverage', 'serendipity'
        ]

        for metric in metrics:
            # Plot histogram
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df_grouped[metric], bins=5, color='#cce5ff', edgecolor='black')
            ax.set_title(f"Histogram of {metric.title()}")
            ax.set_xlabel(metric)
            ax.set_ylabel("Frequency")

            # Save figure
            output_filename = f"{metric}_hist.png"
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, output_filename), dpi=150)
            plt.close()


if __name__ == "__main__":
    # Example usage
    plotter = DataPlotter(
        global_coverage_csv="results/mfcc/global_coverage.csv",
        evaluation_results_csv="results/mfcc/evaluation_results.csv"
    )

    # 1. Plot global coverage
    plotter.plot_global_coverage("results/full10n/global_coverage_plot.png")

    # 2. Group evaluation results
    df_grouped = plotter.group_evaluation_results("results/mfcc/evaluation_system_results.csv")

    # 3. Plot histograms of aggregated metrics
    plotter.plot_evaluation_histograms("results/mfcc/evaluation_system_results.csv", "histograms")
