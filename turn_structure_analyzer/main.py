# main.py
"""
turn_structure_analyzer main script

Description: Main script to run the 53 Mhz analysis loop for a specific list of files.
"""
import os
import glob
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the custom modules
import config
from analysis import TurnStructureAnalyzer
from plotting import TurnStructurePlotter

# --- Set global font sizes for all plots ---
plt.rcParams['axes.labelsize']   = 14 # x and y labels
plt.rcParams['axes.titlesize']   = 16 # main title of a subplot
plt.rcParams['figure.titlesize'] = 18 # main title of a figure
plt.rcParams['xtick.labelsize']  = 12 # x-axis tick labels
plt.rcParams['ytick.labelsize']  = 12 # y-axis tick labels
plt.rcParams['legend.fontsize']  = 12 # legend




def plot_filled_batches_summary(df, plots_dir):
    """
    SUMMARY PLOT: Plots the number of filled batches per spill vs. spill number 
    (chronological, not sequential) for all runs, styled with one color per run.
    """
    # --- 1. Data Preparation ---
    if df.empty or df['filled_batches'].isnull().all():
        return # Nothing to plot

    # Ensure data is numeric and sorted chronologically by spill number
    df['run'] = pd.to_numeric(df['run'])
    df['spill'] = pd.to_numeric(df['spill'])
    df_sorted = df.sort_values(by='spill').dropna(subset=['filled_batches']).reset_index(drop=True)

    # --- 2. Plot Setup ---
    fig, ax = plt.subplots(figsize=(15, 10))
    unique_runs = sorted(df_sorted['run'].unique()) # Sort runs for a clean legend
    colormap = plt.get_cmap('hsv', len(unique_runs))

    # --- 3. Plotting Logic (Loop over each run) ---
    for i, run_number in enumerate(unique_runs):
        run_data = df_sorted[df_sorted['run'] == run_number]
        ax.plot(
            run_data.index, 
            run_data['filled_batches'], 
            'o', 
            ms=4, 
            color=colormap(i), 
            label=f'Run {run_number}'
        )

    # --- 4. Styling and Labels ---
    ax.set_title("Number of Filled Batches vs. Spill Number")
    ax.set_xlabel("Spill Number (chronological, not sequential)")
    ax.set_ylabel("Number of Filled Batches")
    ax.grid(True, linestyle='--')
    ax.set_xlim(left=-1, right=len(df_sorted))
    
    # Set Y-axis for discrete integer values (0-7 batches)
    ax.set_yticks(range(8))
    ax.set_ylim(bottom=-0.5, top=7.5)

    # --- 5. Custom X-Axis Ticks (to prevent overcrowding) ---
    tick_spacing = 40
    if len(df_sorted) < tick_spacing: # Adjust for smaller datasets
        tick_spacing = len(df_sorted) // 4 if len(df_sorted) > 4 else 1
    if tick_spacing == 0: tick_spacing = 1
    
    tick_locs = range(0, len(df_sorted), tick_spacing)
    tick_labels = df_sorted['spill'].iloc[tick_locs]
    
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.tick_params(axis='x', which='major', labelsize=8)

    # --- 6. Legend and Layout ---
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=10, title="Run Number", fontsize='small')
    plt.subplots_adjust(bottom=0.3) # Make room for legend and labels

    # --- 7. Save Figure ---
    output_filename = f"{plots_dir}/_Overall_Filled_Batches_vs_Spill_Summary.png"
    plt.savefig(output_filename, dpi=200)
    plt.close(fig)
    print(f"\nOverall Filled Batches plot saved to: {output_filename}")



def main():
    """
    Main function to process and analyze all specified ROOT files.
    """
    # --- Create Plots directory if it doesn't exist ---
    if not os.path.exists(config.PLOTS_DIR):
        os.makedirs(config.PLOTS_DIR)
        print(f"Created directory: {config.PLOTS_DIR}")


    if config.PROCESS_ALL_FILES_IN_DIR:
        # Find all .root files in the directory
        search_path = os.path.join(config.ROOT_FILE_PATH, "*.root")
        # Get just the filenames, not the full path, to match the existing loop structure
        files_to_process = [os.path.basename(f) for f in glob.glob(search_path)]
        print(f"Processing all {len(files_to_process)} files in the directory...")
    else:
        # Use the specific list from the config file
        files_to_process = config.FILENAMES
        print(f"Processing {len(files_to_process)} specific files from config.FILENAMES...")
    

    # Initialize lists to hold cross-file statistics
    top_buckets_overall = []
    top_batches_overall = []
    top_turns_overall = []
    filled_batch_counts = []

    overall_batch_maxes = {i: [] for i in range(1, config.NUM_BATCHES_PER_TURN + 1)}
 
    # --- Main Execution Loop Over Files specified in config ---
    for filename in files_to_process:
        # Construct the full path to the file
        #filepath = os.path.join(config.ROOT_FILE_SUBFOLDER, filename)

        # Combine the root file directory and the file pattern into a single string   
        filepath = os.path.join(config.ROOT_FILE_PATH, filename)


        # Check if the file actually exists before processing
        if not os.path.exists(filepath):
            print(f"\nWARNING: File not found, skipping: {filepath}")
            continue

        print("\n========================================================")
        print(f"Processing File: {filename}")
        print("========================================================")
        
        try:
            # Combine general and file-specific configurations
            analyzer_config = config.FILE_PARAMETERS.get(filename, {})
            analyzer_config['BUCKETS_PER_BATCH'] = config.BUCKETS_PER_BATCH
            analyzer_config['NUM_BATCHES_PER_TURN'] = config.NUM_BATCHES_PER_TURN
            analyzer_config['HISTOGRAM_NAME'] = config.HISTOGRAM_NAME
            analyzer_config['ACNET_TSV_PATH'] = getattr(config, 'ACNET_TSV_PATH', None)
            analyzer_config['AUTO_FIND_START_BUCKET'] = config.AUTO_FIND_START_BUCKET
            analyzer_config['START_FINDER_NOISE_WINDOW'] = config.START_FINDER_NOISE_WINDOW
            analyzer_config['START_FINDER_THRESHOLD_STD'] = config.START_FINDER_THRESHOLD_STD
            analyzer_config['REFINE_START_BUCKET_ITERATIVELY'] = config.REFINE_START_BUCKET_ITERATIVELY
            analyzer_config['REFINEMENT_SEARCH_WINDOW'] = config.REFINEMENT_SEARCH_WINDOW
            analyzer_config['REFINEMENT_SAMPLE_TURNS'] = config.REFINEMENT_SAMPLE_TURNS

            # 1. Initialize and run the analysis
            analyzer = TurnStructureAnalyzer(filepath=filepath, config=analyzer_config)
            analyzer.run_analysis()

            # 2. Collect cross-file stats if available
            if 'aggregate_stats' in analyzer.results:
                agg_stats = analyzer.results['aggregate_stats']

                # Collect info for top N bucket list
                if agg_stats.get('spill_max_bucket'):
                    top_buckets_overall.append(agg_stats['spill_max_bucket'])

                # Collect info for top N batch integral list
                if agg_stats.get('spill_max_batch_integral'):
                    top_batches_overall.append(agg_stats['spill_max_batch_integral'])
                
                # Collect info for top N turn integral list
                if agg_stats.get('spill_max_turn_integral'):
                    top_turns_overall.append(agg_stats['spill_max_turn_integral'])

                # Collect max intensity for each batch for this file
                batch_maxes = agg_stats.get('batch_maxes_overall', [])
                print("\n--- Per-Batch Max Intensities for this file ---")
                for i, b_max in enumerate(batch_maxes):
                    batch_num = i + 1
                    if pd.notna(b_max):
                        overall_batch_maxes[batch_num].append(b_max)
                        print(f"  Batch {batch_num} Max: {b_max:.2f}")
                print("---------------------------------------------")

                # Properly count if batches are filled in this spill
                count = agg_stats.get('robust_filled_batch_count')
                if count is not None:
                    filled_batch_counts.append((analyzer.run_num, analyzer.spill_num, count))

            # 3. Generate all plots for this file
            plotter_config = {
                'PLOTS_DIR': config.PLOTS_DIR,
                'PLOT_COLORS': config.PLOT_COLORS,
                'PLOT_FIRST_N_TURNS': config.PLOT_FIRST_N_TURNS,
                'PLOT_INTENSITY_THRESHOLD_FRAC': config.PLOT_INTENSITY_THRESHOLD_FRAC,
                'NUM_BATCHES_PER_TURN': config.NUM_BATCHES_PER_TURN,
                'BUCKETS_PER_BATCH': config.BUCKETS_PER_BATCH,
                'start_bucket': analyzer.config.get('start_bucket', 87)
            }
            plotter = TurnStructurePlotter(analyzer=analyzer, config=plotter_config)
            plotter.plot_all()

            print("\n--- Acnet Spill Info ---")
            acnet_data = analyzer.results.get('acnet_data', {})
            
            # Use .get() to safely retrieve values, providing a default if the key is missing
            turn13_val = acnet_data.get('G:TURN13', 'Not Found')
            bnch13_val = acnet_data.get('G:BNCH13', 'Not Found')
            nbsyd_val = acnet_data.get('G:NBSYD', 'Not Found')
            ftsdf_val = acnet_data.get('I:FTSDF', 'Not Found') 
            f1sem_val = acnet_data.get('S:F1SEM', 'Not Found')
            g2sem_val = acnet_data.get('S:G2SEM', 'Not Found')
            nm2ion_val = acnet_data.get('F:NM2ION', 'Not Found')
            nm3ion_val = acnet_data.get('F:NM3ION', 'Not Found')
            nm3sem_val = acnet_data.get('F:NM3SEM', 'Not Found')
            m3tghm_val = acnet_data.get('E:M3TGHM', 'Not Found')
            m3tghs_val = acnet_data.get('E:M3TGHS', 'Not Found')
            m3tgvm_val = acnet_data.get('E:M3TGVM', 'Not Found')
            m3tgvs_val = acnet_data.get('E:M3TGVS', 'Not Found')

            print(f"  G:TURN13 (Number of turns where a set of protons are injected into single RF bucket.): {turn13_val}")
            print(f"  G:BNCH13 (Filled buckets per cycle): {bnch13_val}")
            print(f"  G:NBSYD  (Filled buckets every 7 RF buckets): {nbsyd_val}")
            print(f"  I:FTSDF  (Duty factor sampled at 53 kHz.): {ftsdf_val}")
            print(f"  S:F1SEM  (Number of protons measured with SEM at F1?): {f1sem_val} ppp")
            print(f"  S:G2SEM  (Number of protons measured with SEM at G2.): {g2sem_val} ppp")
            print(f"  F:NM2ION (Number of protons measured with Ion Chamber at NM2.): {nm2ion_val} ppp")
            print(f"  F:NM3ION (Number of protons measured with Ion Chamber at NM3.): {nm3ion_val} ppp")
            print(f"  F:NM3SEM (Number of protons measured with SEM at NM3.): {nm3sem_val} ppp")
            print(f"  E:M3TGHM (Mean of horizontal beam profile measured at NM3.): {m3tghm_val} mm")
            print(f"  E:M3TGHS (Sigma of horizontal beam profile measured at NM3.): {m3tghs_val} mm")
            print(f"  E:M3TGVM (Mean of vertical beam profile measured at NM3.): {m3tgvm_val} mm")
            print(f"  E:M3TGVS (Sigma of vertical beam profile measured at NM3.): {m3tgvs_val} mm")



            print("------------------------")


        except Exception as e:
            print(f"FATAL: An error occurred while processing {filename}: {e}")
            traceback.print_exc()
        
        print(f"Finished processing {filename}")

    # --- Final Plotting Across All Files ---

    if filled_batch_counts:
        results_df = pd.DataFrame(filled_batch_counts, columns=['run', 'spill', 'filled_batches'])
        #results_df = pd.DataFrame(filled_batch_counts, columns=['spill_num', 'filled_batches'])
        results_df['run'] = pd.to_numeric(results_df['run'])
        results_df['spill'] = pd.to_numeric(results_df['spill'])
        results_df = results_df.sort_values('spill').reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(14, 8))
        x_positions = np.arange(len(results_df))
        ax.bar(x_positions, results_df['filled_batches'])

        ax.set_xlabel("Spill Number")
        ax.set_ylabel("Number of Filled Batches (Robust Count)")
        ax.set_title("Robustly Counted Filled Booster Batches per Spill")
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(results_df['spill'], rotation=90, ha='center', fontsize='small')
        ax.grid(True, axis='y', linestyle='--')
        ax.set_yticks(range(8))
        ax.set_ylim(bottom=0)
        fig.tight_layout()

        plot_filename = os.path.join(config.PLOTS_DIR, "_Overall_Filled_Batches_vs_Spill.png")
        plt.savefig(plot_filename, dpi=150)
        plt.close(fig)
        print(f"\nRobust batch count plot saved to:\n{plot_filename}")

        plot_filled_batches_summary(results_df, config.PLOTS_DIR)


    # --- Final Summary Across All Files ---
    print("\n========================================================")
    print("Finished processing all files.")

    # Sort and print top 20 most intense buckets
    top_buckets_overall.sort(key=lambda x: x['intensity'], reverse=True)
    print("\n--- Top 20 Most Intense Buckets (Across All Files) ---")
    for i, bucket in enumerate(top_buckets_overall[:20]):
        print(f"  {i+1:2d}. Intensity: {bucket['intensity']:<10.2f} | "
              f"Run: {bucket['run_num']}, Spill: {bucket['spill_num']}, "
              f"Turn: {bucket['turn_num']}, Batch: {bucket['batch_num']}")
    print("------------------------------------------------------")

    # Sort and print top 20 highest integrated intensity batches
    top_batches_overall.sort(key=lambda x: x['integral'], reverse=True)
    print("\n--- Top 20 Highest Integrated Intensity Batches (Across All Files) ---")
    for i, batch in enumerate(top_batches_overall[:20]):
        print(f"  {i+1:2d}. Integral: {batch['integral']:<12.2f} | "
              f"Run: {batch['run_num']}, Spill: {batch['spill_num']}, "
              f"Turn: {batch['turn_num']}, Batch: {batch['batch_num']}")
    print("--------------------------------------------------------------------")

    # Sort and print top 20 highest integrated intensity turns
    top_turns_overall.sort(key=lambda x: x['integral'], reverse=True)
    print("\n--- Top 20 Highest Integrated Intensity Turns (Across All Files) ---")
    for i, turn in enumerate(top_turns_overall[:20]):
        print(f"  {i+1:2d}. Integral: {turn['integral']:<12.2f} | "
              f"Run: {turn['run_num']}, Spill: {turn['spill_num']}, Turn: {turn['turn_num']}")
    print("------------------------------------------------------------------")

    # Print overall max intensity for each batch
    print("\n--- Overall Max Intensity per Batch (Across All Files) ---")
    for batch_num, intensities in overall_batch_maxes.items():
        if intensities:
            print(f"  Batch {batch_num}: {max(intensities):.4f}")
        else:
            print(f"  Batch {batch_num}: No data")
    print("----------------------------------------------------------")

    now = pd.Timestamp.now(tz='America/Chicago')
    print(f"\nAnalysis completed on: {now.strftime('%A, %B %d, %Y at %I:%M:%S %p %Z')}")
    print(f"Location: Batavia, Illinois, United States")
    print("========================================================\n")
    # --- 🔼 END OF MODIFIED/NEW SECTION 🔼 ---
if __name__ == "__main__":
    main()