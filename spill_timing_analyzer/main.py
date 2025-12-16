# main.py
"""
spill_timing_analyzer main script 

Description: Analyze the FreqHist_**Hz histograms from 
  ROOT files from multiple runs and spills.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re 

# Import the custom modules
import config
from analysis import SpillAnalyzer
from plotting import SpillPlotter, ACNET_METADATA

# --- Set global font sizes for all plots ---
plt.rcParams['axes.labelsize']   = 14 # x and y labels
plt.rcParams['axes.titlesize']   = 16 # main title of a subplot
plt.rcParams['figure.titlesize'] = 18 # main title of a figure
plt.rcParams['xtick.labelsize']  = 12 # x-axis tick labels
plt.rcParams['ytick.labelsize']  = 12 # y-axis tick labels
plt.rcParams['legend.fontsize']  = 12 # legend


def main():
    """
    Main function to find root files, analyze them, and make plots
    """

    ###-----------------------------------------------------
    ### FIND FILES

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

    print("-----Start Pre-flight File Check-----")
    validated_filepaths = []
    skipped_count = 0
    for filepath in sorted(filepaths):
        filename = os.path.basename(filepath)
        match = re.search(r"run(\d+)\.spill(\d+)\.root", filename)
        
        if match:
            spill_num = int(match.group(2))
            # Construct the expected TSV filename
            spill_num_padded = f"{spill_num:09d}"
            tsv_filename = f"spill_{spill_num_padded}_Acnet.tsv"
            #tsv_filepath = os.path.join(os.path.dirname(filepath), tsv_filename)
            tsv_filepath = os.path.join(config.ACNET_TSV_PATH, tsv_filename)

            # Check if the corresponding TSV file exists
            if os.path.exists(tsv_filepath):
                validated_filepaths.append(filepath)
            else:
                print(f"  - SKIPPING: Missing required TSV file:\n    {tsv_filepath}")
                skipped_count += 1
        else:
            print(f"  - SKIPPING: Filename '{filename}' does not match expected pattern.")
            skipped_count += 1

    print(f"Check complete. Found {len(validated_filepaths)} valid file pairs.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} ROOT files due to missing TSV files or incorrect naming.")
    print("------------------------------------")

    if not validated_filepaths:
        print("Error: No valid ROOT/TSV file pairs found to process.")
        return

    ###-----------------------------------------------------
    ### INITIALIZATION

    # List which will contain dictionaries with summary metrics for each analyzed spill
    all_results = [] 

    # Dictionary which will store SpillAnalyzer objects (key=run number)
    analyzers_by_run = {} #

    # Pandas series to accumlate intensity from each spill in different time bins (to create a weighted histogram later)
    summed_spill_data = None 

    # Collect fft peaks to count how often they occur
    all_fft_peaks = []
    all_ranked_fft_peaks = []
    all_magnitude_fft_peaks = []

    all_intensity_profiles = {} # For Stability Plot
    all_spectra = []            # For 2D Spectrogram
    fft_frequencies = None      # To store the frequency axis

    ###-----------------------------------------------------
    ### FILE PROCESSING

    print("-----Start Analyzer File Loop-----")

    # Loop over files (loop over each individual spill)

    # for filepath in sorted(filepaths):

    for filepath in validated_filepaths:

        print(f"\nProcessing input : {os.path.basename(filepath)}")
        
        # Create an instance of the SpillAnalyzer class for this spill
        analyzer = SpillAnalyzer(filepath, config)
        
        # Check if the SpillAnalyzer instance was created properly for this file
        if analyzer.run_num != -1:


            # Create the dictionary of per-spill metrics
            result_dict = {
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
                'top_fft_freq': analyzer.top_fft_freq,
                'acnet_timestamp': analyzer.acnet_timestamp 
            }

            # Merge the Acnet data into the main results dictionary.
            # This will add a new column to the final DataFrame for each variable
            # found in the TSV file (e.g., 'F:NS2FLO', 'S:G2SEM', etc.).
            if analyzer.acnet_data:
                result_dict.update(analyzer.acnet_data)

            # Append the combined dictionary to the all_results list
            all_results.append(result_dict)

            print(f"--- ACNET Data for Spill: {analyzer.spill_num} ---")
            # Loop through the metadata dictionary to get variable names and titles
            for variable, metadata in ACNET_METADATA.items():
                # Get the value from the analyzer's acnet_data dictionary
                # .get() is used to safely retrieve the value, returning 'N/A' if not found
                value = analyzer.acnet_data.get(variable, 'N/A')
                title = metadata['title'].split(' vs.')[0] # Get the descriptive title
                print(f"  - {title:<45} ({variable}): {value}")


            if analyzer.fft_peaks:
                all_fft_peaks.extend(analyzer.fft_peaks)
            
            if analyzer.ranked_fft_peaks:
                all_ranked_fft_peaks.extend(analyzer.ranked_fft_peaks)

            if analyzer.magnitude_fft_peaks:
                all_magnitude_fft_peaks.extend(analyzer.magnitude_fft_peaks)
            
            # --- Collect data for Stability Plot ---
            # Use a unique key for each spill
            spill_key = f"run{analyzer.run_num}_spill{analyzer.spill_num}"
            all_intensity_profiles[spill_key] = analyzer.data.set_index('time_s')['intensity']

            # --- Collect data for 2D Spectrogram ---
            freqs, spectrum = analyzer.get_fft_power_spectrum()
            if spectrum is not None:
                all_spectra.append(spectrum)
                if fft_frequencies is None:
                    # Store the frequency axis from the first valid spill
                    fft_frequencies = freqs 

            # Logic to create plots integrating multiple spills
            if summed_spill_data is None:
                # For the first valid spill, use its data to set up the accumulator.
                # Set the time values as the index.
                summed_spill_data = analyzer.data.set_index('time_s')['intensity']
            else:
                # For subsequent spills, add their intensity to the sum.
                # We align on the index (time_s) to ensure we're adding correctly.
                current_spill_series = analyzer.data.set_index('time_s')['intensity']
                summed_spill_data = summed_spill_data.add(current_spill_series, fill_value=0)
            
            if analyzer.run_num not in analyzers_by_run:
                analyzers_by_run[analyzer.run_num] = []
            analyzers_by_run[analyzer.run_num].append(analyzer)
    print("-----End File Loop-----")

    if not all_results:
        print("No valid data processed. Exiting.")
        return

    ### PLOTTING

    print("-----Start Plotter-----")
    plotter = SpillPlotter(config)
    summary_df = pd.DataFrame(all_results)


    # Temporarily display all columns without truncation
    pd.set_option('display.max_columns', None) 
    
    print("\n--- DEBUG: Displaying first 5 rows of summary_df ---")
    print(summary_df.head(5))
    print("-----------------------------------------------------\n")
    print("\n--- DEBUG: Displaying last 5 rows of summary_df ---")
    print(summary_df.tail(5))
    print("-----------------------------------------------------\n")
    
    
    # --- Create Stability Plot Data ---
    if all_intensity_profiles:
        # Combine all intensity series into a single DataFrame
        intensity_df = pd.DataFrame(all_intensity_profiles).fillna(0)
        # Calculate mean and std deviation across all spills (columns)
        mean_profile = intensity_df.mean(axis=1)
        std_profile = intensity_df.std(axis=1)
        # Call the new plotting function
        plotter.plot_stability_profile(mean_profile, std_profile)

    # --- Call Spectrogram Plot ---
    if all_spectra and fft_frequencies is not None:
        plotter.plot_fft_spectrogram(all_spectra, fft_frequencies)
        plotter.plot_fft_spectrogram_zoomed(all_spectra, fft_frequencies, freq_max=800, major_marker=100)
        plotter.plot_fft_spectrogram_zoomed(all_spectra, fft_frequencies, freq_max=500, major_marker=50)
        plotter.plot_fft_spectrogram_zoomed(all_spectra, fft_frequencies, freq_max=250, major_marker=25)

    # --- Generate Overall Summary Plots (combine all considered spills) ---
    plotter.plot_spill_count_per_run(summary_df)
    plotter.plot_total_intensity_overall(summary_df)
    plotter.plot_total_intensity_half_spill_overall(summary_df)
    plotter.plot_max_intensity_overall(summary_df)
    plotter.plot_duty_factor_overall(summary_df)
    plotter.plot_duty_factor_comparison(summary_df)
    plotter.plot_time_of_max_overall(summary_df)
    plotter.plot_overall_time_of_max_histogram(summary_df, weighted=False)
    plotter.plot_overall_top_fft_frequency_histogram(summary_df)
    plotter.plot_ranked_choice_fft_histogram(all_ranked_fft_peaks)
    plotter.plot_weighted_fft_peak_histogram(all_magnitude_fft_peaks)

    plotter.plot_overall_peak_width_histogram(analyzers_by_run)
    plotter.plot_overall_peak_interval_histogram(analyzers_by_run)

    print("\n--- Generating ACNET Summary Plots ---")
    acnet_variables_to_plot = [
        'G:TURN13', 'G:BNCH13', 'G:NBSYD', 'I:FTSDF', 'S:F1SEM', 'S:G2SEM',
        'F:NM2ION', 'F:NM3ION', 'F:NM3SEM', 'E:M3TGHM', 'E:M3TGHS',
        'E:M3TGVM', 'E:M3TGVS'
    ]

    for variable in acnet_variables_to_plot:
        if variable in ACNET_METADATA:
            metadata = ACNET_METADATA[variable]
            plotter.plot_acnet_variable_vs_spill(
                summary_df=summary_df,
                variable_name=variable,
                plot_title=metadata['title'],
                yaxis_label=f"{metadata['title'].split(' vs.')[0]} ({metadata['unit']})"
            )
        else:
            print(f"  - WARNING: No metadata found for ACNET variable '{variable}'. Skipping plot.")

    plotter.plot_acnet_over_time(summary_df, "E:M3TGHM") 
    plotter.plot_acnet_over_time(summary_df, "E:M3TGHS") # Sigma of Horizontal Beam Profile
    plotter.plot_acnet_over_time(summary_df, "E:M3TGVM") 
    plotter.plot_acnet_over_time(summary_df, "E:M3TGVS") 
    plotter.plot_acnet_over_time(summary_df, "S:F1SEM")  
    plotter.plot_acnet_over_time(summary_df, "S:G2SEM")  # Protons at SEM G2
    plotter.plot_acnet_over_time(summary_df, "I:FTSDF")  # 53kHz Duty Factor
    plotter.plot_acnet_over_time(summary_df, "F:NM2ION")  
    plotter.plot_acnet_over_time(summary_df, "F:NM3ION")  
    plotter.plot_acnet_over_time(summary_df, "F:NM3SEM")  


    print("\n--- Generating ACNET Grid Plots ---")
    
    # Define the list of variables you want to plot together
    beam_intensity_monitors = [
        'S:F1SEM',
        'S:G2SEM',
        'F:NM2ION',
        'F:NM3SEM'
    ]

    # Call the new plotting function
    plotter.plot_acnet_grid_vs_spill(
        summary_df=summary_df, 
        variables_to_plot=beam_intensity_monitors, 
        plot_title="Beam Intensity Monitor Readings vs. Spill Number",
        output_filename="Overall_Beam_Monitors_vs_Spill.png"
    )


    # Define the list of variables you want to plot together
    beam_intensity_related_ = [
        'total_intensity',
        'G:NBSYD',
        'S:F1SEM',
        #'S:G2SEM',
        #'F:NM2ION',
        #'F:NM3SEM',
        #'max_intensity',
        #'mean_intensity',
        #'std_intensity'
    ]



    # Call the new plotting function
    plotter.plot_acnet_grid_vs_spill(
        summary_df=summary_df, 
        variables_to_plot=beam_intensity_related_, 
        plot_title="Beam Intensity Related vs. Spill Number",
        output_filename="Overall_Beam_Intensity_Related_vs_Spill.png"
    )


    # --- START: NEW SECTION FOR CROSS-CORRELATION ---
    print("\n--- Generating Cross-Correlation Plots ---")
    
    # Call the new function with the desired variables
    plotter.plot_cross_correlation(
        summary_df=summary_df, 
        var1='total_intensity', 
        var2='G:NBSYD'
    )

    plotter.plot_cross_correlation(
        summary_df=summary_df, 
        var1='total_intensity', 
        var2='S:F1SEM'
    )

    plotter.plot_cross_correlation(
        summary_df=summary_df, 
        var1='total_intensity', 
        var2='S:G2SEM'
    )

    if summed_spill_data is not None:
        plotter.plot_overall_integrated_spill(summed_spill_data)

    # --- Generate Per-Run Summary Plots and Plot Details of Most Intense Spill in a Given Run  ---
    for run_number, run_data in summary_df.groupby('run'):
        print(f"\n--- Generating summary plots and most intense spill plots for Run {run_number} ---")
        
        most_intense_analyzer = None
        if run_number in analyzers_by_run:
            
            #Find the most intense spill for a given run
            most_intense_analyzer = max(analyzers_by_run[run_number], key=lambda x: x.total_intensity if not np.isnan(x.total_intensity) else -1)
            
            # Plot intensity vs time and corresponding FFT
            plotter.plot_single_spill(most_intense_analyzer, is_representative_spill=True)
            plotter.plot_fft(run_number, most_intense_analyzer)

            # These are slow when there are many bins (only do them for certain histograms)
            if config.HISTOGRAM_NAME == "FreqHist_10kHz" or config.HISTOGRAM_NAME == "FreqHist_7_5kHz" or config.HISTOGRAM_NAME == "FreqHist_1kHz": 
                plotter.plot_single_spill_zoomed(most_intense_analyzer, 0, 0.5)
                plotter.plot_single_spill_zoomed(most_intense_analyzer, 0.5, 1)
                plotter.plot_single_spill_zoomed(most_intense_analyzer, 1, 1.5)
                plotter.plot_single_spill_zoomed(most_intense_analyzer, 1.5, 2)
                plotter.plot_single_spill_zoomed(most_intense_analyzer, 2, 2.5)
                plotter.plot_single_spill_zoomed(most_intense_analyzer, 2.5, 3)
                plotter.plot_single_spill_zoomed(most_intense_analyzer, 3, 3.5)
                plotter.plot_single_spill_zoomed(most_intense_analyzer, 3.5, 4)
                plotter.plot_autocorrelation(run_number, most_intense_analyzer)
                plotter.plot_fft_of_autocorrelation(run_number, most_intense_analyzer) 
                most_intense_analyzer.calculate_duty_factor_vs_time()
                most_intense_analyzer.calculate_intensity_vs_time()
                plotter.plot_duty_factor_and_intensity_vs_time(most_intense_analyzer)


        plotter.plot_max_intensity_vs_spill(run_number, run_data)
        plotter.plot_total_intensity_vs_spill(run_number, run_data)
        plotter.plot_duty_factor_vs_spill(run_number, run_data)
        plotter.plot_uniformity_vs_spill(run_number, run_data)
        #plotter.plot_peak_interval_histogram(run_number, run_data, weighted=False)
        #plotter.plot_peak_width_histogram(run_number, run_data) 
        plotter.plot_time_of_max_vs_spill(run_number, run_data)
        plotter.plot_time_of_max_histogram(run_number, run_data, weighted=False)
        # These aren't particularly enlightening 
        #plotter.plot_peak_interval_histogram(run_number, run_data, weighted=True)
        #plotter.plot_time_of_max_histogram(run_number, run_data, weighted=True)

        # --- UPDATE THIS SECTION ---
        # Get the list of analyzers for the current run
        analyzers_for_run = analyzers_by_run.get(run_number, [])
        if analyzers_for_run:
            # Pass the list of analyzers to the updated functions
            plotter.plot_peak_interval_histogram(run_number, analyzers_for_run)
            plotter.plot_peak_width_histogram(run_number, analyzers_for_run)
        # --- END UPDATE ---


    # ----- NEW SECTION: Duty Factor vs. Time for High-Intensity Spills -----
    print("\n-----Start Duty Factor vs. Time Analysis for High-Intensity Spills-----")
    if not summary_df.empty and config.DUTY_FACTOR_INTERVAL_S > 0:
        # Determine the intensity threshold based on the configured percentile
        intensity_threshold = summary_df['total_intensity'].quantile(config.HIGH_INTENSITY_SPILL_PERCENTILE / 100.0)
        high_intensity_spills_df = summary_df[summary_df['total_intensity'] >= intensity_threshold]

        print(f"Found {len(high_intensity_spills_df)} high-intensity spills (top {100-config.HIGH_INTENSITY_SPILL_PERCENTILE}%) to analyze...")

        for _, spill_info in high_intensity_spills_df.iterrows():
            run_num = spill_info['run']

            # Find the corresponding analyzer object from our processed data
            analyzer = next((a for a in analyzers_by_run.get(run_num, []) if a.spill_num == spill_info['spill']), None)

            if analyzer:
                analyzer.calculate_duty_factor_vs_time()
                #plotter.plot_duty_factor_vs_time(analyzer)
                analyzer.calculate_intensity_vs_time()
                #plotter.plot_intensity_vs_time_barchart(analyzer)
                plotter.plot_duty_factor_and_intensity_vs_time(analyzer)
                plotter.plot_single_spill(analyzer)
                if config.HISTOGRAM_NAME in ["FreqHist_10kHz", "FreqHist_7_5kHz", "FreqHist_1kHz"]:
                    plotter.plot_single_spill_zoomed(most_intense_analyzer, 0, 0.5)
                    plotter.plot_single_spill_zoomed(most_intense_analyzer, 0.5, 1)
                    plotter.plot_single_spill_zoomed(most_intense_analyzer, 1, 1.5)
                    plotter.plot_single_spill_zoomed(most_intense_analyzer, 1.5, 2)
                    plotter.plot_single_spill_zoomed(most_intense_analyzer, 2, 2.5)
                    plotter.plot_single_spill_zoomed(most_intense_analyzer, 2.5, 3)
                    plotter.plot_single_spill_zoomed(most_intense_analyzer, 3, 3.5)
                    plotter.plot_single_spill_zoomed(most_intense_analyzer, 3.5, 4)

    # ----- NEW SECTION: Duty Factor vs. Time for High-Spike Spills -----
    print("\n-----Start Duty Factor vs. Time Analysis for High-Spike Spills-----")
    if not summary_df.empty and config.HIGH_SPIKE_SPILL_PERCENTILE > 0:
        # Determine the spike threshold based on the configured percentile for max_intensity
        spike_threshold = summary_df['max_intensity'].quantile(config.HIGH_SPIKE_SPILL_PERCENTILE / 100.0)
        high_spike_spills_df = summary_df[summary_df['max_intensity'] >= spike_threshold]

        print(f"Found {len(high_spike_spills_df)} high-spike spills (top {100-config.HIGH_SPIKE_SPILL_PERCENTILE}%) to analyze...")

        for _, spill_info in high_spike_spills_df.iterrows():
            run_num = spill_info['run']

            # Find the corresponding analyzer object
            analyzer = next((a for a in analyzers_by_run.get(run_num, []) if a.spill_num == spill_info['spill']), None)

            if analyzer:
                # We can reuse the same functions as before
                analyzer.calculate_duty_factor_vs_time()
                #plotter.plot_duty_factor_vs_time(analyzer)
                analyzer.calculate_intensity_vs_time()
                #plotter.plot_intensity_vs_time_barchart(analyzer)
                plotter.plot_duty_factor_and_intensity_vs_time(analyzer)
                plotter.plot_single_spill(analyzer)
                if config.HISTOGRAM_NAME in ["FreqHist_10kHz", "FreqHist_7_5kHz", "FreqHist_1kHz"]:
                    plotter.plot_single_spill_zoomed(most_intense_analyzer, 0, 0.5)
                    plotter.plot_single_spill_zoomed(most_intense_analyzer, 0.5, 1)
                    plotter.plot_single_spill_zoomed(most_intense_analyzer, 1, 1.5)
                    plotter.plot_single_spill_zoomed(most_intense_analyzer, 1.5, 2)
                    plotter.plot_single_spill_zoomed(most_intense_analyzer, 2, 2.5)
                    plotter.plot_single_spill_zoomed(most_intense_analyzer, 2.5, 3)
                    plotter.plot_single_spill_zoomed(most_intense_analyzer, 3, 3.5)
                    plotter.plot_single_spill_zoomed(most_intense_analyzer, 3.5, 4)
            
    print("-----End Plotter-----")

    """
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
    """    
    
    # --- Final Text Summary ---
    print("\n========================================================")
    print("Analysis Summary")
    print("-----------------------------------")
    print(f"Total Number of Runs Considered: {summary_df['run'].nunique()}")
    print(f"Total Number of Spills Considered: {len(summary_df)}")

    if not summary_df.empty:
        # (This is the existing code for Top 5 Spills)
        print("\nTop 5 Spills by Total Integrated Intensity:")
        top_5_spills = summary_df.sort_values(by='total_intensity', ascending=False).head(5)
        for index, row in top_5_spills.iterrows():
            print(f"  - File: {row['filepath']:<40} Intensity: {row['total_intensity']:,.0f}")

        # ----- REVISED SECTION: Find and Plot Top Spikes -----

        # 1. Initialize the list BEFORE the loop.
        all_peaks_list = []

        # 2. Loop through every analyzer object to collect all peak data.
        for run_num in analyzers_by_run:
            for analyzer in analyzers_by_run[run_num]:
                if not analyzer.peaks_df.empty:
                    analyzer.peaks_df['run'] = analyzer.run_num
                    analyzer.peaks_df['spill'] = analyzer.spill_num
                    all_peaks_list.append(analyzer.peaks_df)



        # 1. Collect the peaks_df from every analyzer object.
        if all_peaks_list:
            # 2. Combine all peaks into a single DataFrame.
            all_peaks_df = pd.concat(all_peaks_list, ignore_index=True)

            #plotter.plot_spike_width_vs_time_scatter(all_peaks_df)
            plotter.plot_spike_width_vs_time_hist2d(all_peaks_df)
            plotter.plot_spike_width_by_spill_half(all_peaks_df)

            plotter.plot_width_vs_intensity_hist2d(all_peaks_df)
            plotter.plot_width_vs_prominence_hist2d(all_peaks_df)
            plotter.plot_intensity_vs_time_hist2d(all_peaks_df)
            plotter.plot_spike_intensity_by_spill_half(all_peaks_df)

            plotter.plot_overall_spike_time_histogram(all_peaks_df)

            # --- 3a. Find and Plot Top Spikes by ABSOLUTE INTENSITY ---
            top_n_by_intensity = all_peaks_df.sort_values(by='intensity', ascending=False).head(config.N_TOP_SPIKES_TO_PLOT)

            print("\nTop Spikes by Absolute Intensity (Most Likely to Cause Inhibit):")
            for i, (_, spike) in enumerate(top_n_by_intensity.iterrows()):
                print(f"  {i+1:2d}. Run: {spike['run']}, Spill: {spike['spill']}, "
                      f"Intensity: {spike['intensity']:<12,.0f} at {spike['time']:.4f} s")

            print("\n----- Generating Zoomed Plots for Top Spikes by Intensity -----")
            for i, (_, spike) in enumerate(top_n_by_intensity.iterrows()):
                analyzer = next((a for a in analyzers_by_run.get(spike['run'], []) if a.spill_num == spike['spill']), None)
                if analyzer:
                    plotter.plot_zoomed_spike(analyzer, spike, i, sort_key="Intensity")

            # --- 3b. Find and Plot Top Spikes by PROMINENCE ---
            top_n_by_prominence = all_peaks_df.sort_values(by='prominence', ascending=False).head(config.N_TOP_SPIKES_TO_PLOT)

            print("\nTop Spikes by Prominence (Most Structurally Unstable):")
            for i, (_, spike) in enumerate(top_n_by_prominence.iterrows()):
                 print(f"  {i+1:2d}. Run: {spike['run']}, Spill: {spike['spill']}, "
                      f"Intensity: {spike['intensity']:<12,.0f} at {spike['time']:.4f} s (Prominence: {spike['prominence']:.0f})")

            print("\n----- Generating Zoomed Plots for Top Spikes by Prominence -----")
            for i, (_, spike) in enumerate(top_n_by_prominence.iterrows()):
                analyzer = next((a for a in analyzers_by_run.get(spike['run'], []) if a.spill_num == spike['spill']), None)
                if analyzer:
                    plotter.plot_zoomed_spike(analyzer, spike, i, sort_key="Prominence")
        print("========================================================\n")


    """
    # ----- REVISED SECTION: Find Top Spikes with NumPy -----

    # 1. Create an empty list to hold all candidate spikes.
    all_spikes = []

    # 2. Loop through every analyzer object to find its local top spikes.
    for run_num in analyzers_by_run:
        for analyzer in analyzers_by_run[run_num]:
            if not analyzer.data.empty:
                # Convert pandas columns to NumPy arrays for direct indexing.
                intensities = analyzer.data['intensity'].to_numpy()
                times = analyzer.data['time_s'].to_numpy()

                # Get the integer indices of the top 10 intensities for this spill.
                # np.argsort sorts from smallest to largest, so we take the last 10.
                top_indices_in_spill = np.argsort(intensities)[-10:]

                # Add these top spikes to our master list.
                for idx in top_indices_in_spill:
                    all_spikes.append({
                        'run': analyzer.run_num,
                        'spill': analyzer.spill_num,
                        'intensity': intensities[idx],
                        'time_s': times[idx]
                    })

    # 3. Sort the master list of all spikes by intensity to find the overall top spikes.
    sorted_spikes = sorted(all_spikes, key=lambda x: x['intensity'], reverse=True)

    # 4. Get the top N spikes based on the config setting.
    top_n_spikes = sorted_spikes[:config.N_TOP_SPIKES_TO_PLOT]

    # 5. Print the formatted results.
    print("\nTop 10 Highest Intensity Spikes (Single Bins):")
    # To print 10, we'll slice the first 10, regardless of the plot setting
    for i, spike in enumerate(sorted_spikes[:10]):
        print(f"  {i+1:2d}. Run: {spike['run']}, Spill: {spike['spill']}, "
              f"Intensity: {spike['intensity']:<12,.0f} at {spike['time_s']:.4f} s")

    # ----- Generating Zoomed Plots for Top Spikes -----
    print("\n----- Generating Zoomed Plots for Top Spikes -----")
    for i, spike in enumerate(top_n_spikes):
        # Find the full analyzer object corresponding to the spill.
        analyzer = next((a for a in analyzers_by_run.get(spike['run'], []) if a.spill_num == spike['spill']), None)

        if analyzer:
            # Call the plotting function, which now receives the corrected spike data.
            plotter.plot_zoomed_spike(analyzer, spike['time_s'], spike['intensity'], i)

    print("========================================================\n")
    """
if __name__ == "__main__":
    main()