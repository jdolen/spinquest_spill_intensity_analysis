# main.py
"""
spill_timing_analyzer main script 
  Analyze the FreqHist_**Hz histograms from 
  ROOT files from multiple runs and spills.
"""
import os
import glob
import pandas as pd
import numpy as np

# Import the custom modules
import config
from analysis import SpillAnalyzer
from plotting import SpillPlotter

def main():
    """
    Main function to find root files, analyze them, and make plots
    """

    # Check if PLOTS_DIR exists and if not create it
    if not os.path.exists(config.PLOTS_DIR):
        os.makedirs(config.PLOTS_DIR)
        print(f"Created output directory: {config.PLOTS_DIR}")

    # Combine the root file directory and the file pattern into a single string   
    file_search_path = os.path.join(config.ROOT_FILE_PATH, config.FILE_PATTERN)

    # Create a list containing the path to every file
    filepaths = glob.glob(file_search_path)

    # Make sure filepaths is not an empty list and therefore False
    if not filepaths:
        print(f"Error: No files found at '{file_search_path}'.")
        return

    print(f"Found {len(filepaths)} files to process...")

    all_results = [] # empty list which will contain dictionaries with summary metrics for each analyzed spill
    analyzers_by_run = {} # empty dictionary which will contain run numbers as keys and per run data

    print("-----Start Analyzer File Loop-----")
    for filepath in sorted(filepaths):
        print(f"\nProcessing: {os.path.basename(filepath)}")
        
        analyzer = SpillAnalyzer(filepath, config)
        
        if analyzer.run_num != -1:
            all_results.append({
                'run': analyzer.run_num,
                'spill': analyzer.spill_num,
                'filepath': os.path.basename(filepath),
                'max_intensity': analyzer.max_intensity,
                'time_of_max': analyzer.time_of_max_intensity,
                'total_intensity': analyzer.total_intensity,
                'mean_intensity': analyzer.mean_intensity,
                'std_intensity': analyzer.std_intensity,
                'total_intensity_0_2_seconds': analyzer.total_intensity_0_2_seconds,
                'total_intensity_2_4_seconds': analyzer.total_intensity_2_4_seconds,
                'duty_factor': analyzer.duty_factor,
                'coefficient_of_variation': analyzer.coefficient_of_variation,
                'kurtosis': analyzer.kurtosis,
                'gini': analyzer.gini,
                'peak_intervals': analyzer.peak_intervals
            })
            
            if analyzer.run_num not in analyzers_by_run:
                analyzers_by_run[analyzer.run_num] = []
            analyzers_by_run[analyzer.run_num].append(analyzer)
    print("-----End File Loop-----")

    if not all_results:
        print("No valid data processed. Exiting.")
        return

    print("-----Start Plotter-----")
    plotter = SpillPlotter(config)
    summary_df = pd.DataFrame(all_results)
    
    # --- Generate Overall Summary Plots ---
    plotter.plot_spill_count_per_run(summary_df)
    plotter.plot_total_intensity_overall(summary_df)
    plotter.plot_max_intensity_overall(summary_df)
    plotter.plot_duty_factor_overall(summary_df)
    plotter.plot_time_of_max_overall(summary_df)
    plotter.plot_overall_time_of_max_histogram(summary_df, weighted=False)
    #plotter.plot_overall_time_of_max_histogram(summary_df, weighted=True)
    
    # --- Generate Per-Run Summary Plots ---
    for run_number, run_data in summary_df.groupby('run'):
        print(f"\n--- Generating summary plots for Run {run_number} ---")
        
        most_intense_analyzer = None
        if run_number in analyzers_by_run:
            most_intense_analyzer = max(analyzers_by_run[run_number], key=lambda x: x.total_intensity if not np.isnan(x.total_intensity) else -1)
            plotter.plot_single_spill(most_intense_analyzer, is_representative_spill=True)
            plotter.plot_single_spill_zoomed(most_intense_analyzer)
            
            # Corrected function calls with both required arguments
            plotter.plot_fft(run_number, most_intense_analyzer)
            plotter.plot_autocorrelation(run_number, most_intense_analyzer)

        plotter.plot_max_intensity_vs_spill(run_number, run_data)
        plotter.plot_total_intensity_vs_spill(run_number, run_data)
        plotter.plot_duty_factor_vs_spill(run_number, run_data)
        plotter.plot_uniformity_vs_spill(run_number, run_data)
        plotter.plot_peak_interval_histogram(run_number, run_data, weighted=False)
        #plotter.plot_peak_interval_histogram(run_number, run_data, weighted=True)
        plotter.plot_time_of_max_vs_spill(run_number, run_data)
        plotter.plot_time_of_max_histogram(run_number, run_data, weighted=False)
        #plotter.plot_time_of_max_histogram(run_number, run_data, weighted=True)
    print("-----End Plotter-----")
    
    # --- Final Text Summary ---
    print("\n========================================================")
    print("Analysis Summary")
    print("-----------------------------------")
    print(f"Total Number of Runs Considered: {summary_df['run'].nunique()}")
    print(f"Total Number of Spills Considered: {len(summary_df)}")
    
    if not summary_df.empty:
        print("\nTop 5 Spills by Total Integrated Intensity:")
        top_5_spills = summary_df.sort_values(by='total_intensity', ascending=False).head(5)
        for index, row in top_5_spills.iterrows():
            print(f"  - File: {row['filepath']:<40} Intensity: {row['total_intensity']:,.0f}")
        
    print("========================================================\n")
    


if __name__ == "__main__":
    main()