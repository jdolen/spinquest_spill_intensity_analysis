"""
Batch Filler Analyzer

Description:
This script loops through all ROOT files in the specified directory to analyze
the batch filling pattern of each spill. It automatically finds the start of the
beam, processes the turn structure, and counts how many of the 7 booster batches
are "filled" vs. "empty" based on their integrated intensity.

It produces a plot showing the number of filled batches versus the spill number
for all analyzed files.
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


def analyze_spill_filling(filepath):
    """
    Analyzes a single spill file to count the number of filled batches.
    ...
    """
    print(f"\n--- Analyzing {os.path.basename(filepath)} ---")
    try:
        # Configuration for the analyzer
        analyzer_config = {
            'BUCKETS_PER_BATCH': config.BUCKETS_PER_BATCH,
            'NUM_BATCHES_PER_TURN': config.NUM_BATCHES_PER_TURN,
            'HISTOGRAM_NAME': config.HISTOGRAM_NAME,
            'AUTO_FIND_START_BUCKET': True,
            'START_FINDER_NOISE_WINDOW': config.START_FINDER_NOISE_WINDOW,
            'START_FINDER_THRESHOLD_STD': config.START_FINDER_THRESHOLD_STD,
        }

        # 1. Initialize analyzer
        analyzer = TurnStructureAnalyzer(filepath=filepath, config=analyzer_config)

        # 2. Determine the threshold for a "filled" batch based on noise
        noise_window = config.START_FINDER_NOISE_WINDOW
        if len(analyzer.intensity_series) < noise_window:
            print("  - Not enough data to establish noise baseline. Skipping.")
            return None

        noise_batch_region = analyzer.intensity_series.iloc[:config.BUCKETS_PER_BATCH]
        integrated_noise_per_batch = noise_batch_region.sum()
        filled_threshold = integrated_noise_per_batch * config.FILLED_BATCH_NOISE_MULTIPLIER
        print(f"  - Integrated noise per batch: {integrated_noise_per_batch:.2f}")
        print(f"  - Intensity threshold for a 'filled' batch: {filled_threshold:.2f}")

        # 3. Process turns to get the turn_df
        analyzer._process_turns(max_turns=config.FILLER_ANALYZER_MAX_TURNS)
        turn_df = analyzer.results.get('turn_df')

        if turn_df is None or turn_df.empty:
            print("  - No turn data found. Skipping.")
            return None

        # --- 🔽 ADDED THE MISSING LINE HERE 🔽 ---
        # 4. Count filled batches for each turn using the new threshold
        batch_integral_cols = [f'integrated_intensity_batch_{i+1}' for i in range(config.NUM_BATCHES_PER_TURN)]
        # --- 🔼 END OF FIX 🔼 ---
        is_filled_df = turn_df[batch_integral_cols] > filled_threshold
        turn_df['filled_batch_count'] = is_filled_df.sum(axis=1)
        
        # 5. Find the most common number of filled batches (the mode) for this spill
        if turn_df['filled_batch_count'].empty:
            return (analyzer.spill_num, 0)
            
        spill_mode_filled_batches = turn_df['filled_batch_count'].mode()[0]
        print(f"  - Most common number of filled batches per turn: {spill_mode_filled_batches}")
        
        return (analyzer.spill_num, spill_mode_filled_batches)

    except Exception as e:
        print(f"  - FATAL: An error occurred: {e}")
        traceback.print_exc()
        return None



def main():
    """
    Main function to process all ROOT files and generate the summary plot.
    """
    # Create Plots directory if it doesn't exist
    if not os.path.exists(config.PLOTS_DIR):
        os.makedirs(config.PLOTS_DIR)
        print(f"Created directory: {config.PLOTS_DIR}")

    if config.PROCESS_ALL_FILES_IN_DIR:
        # Find all .root files in the directory
        search_path = os.path.join(config.ROOT_FILE_PATH, "*.root")
        files_to_process = glob.glob(search_path)
        print(f"Processing all {len(files_to_process)} files in the directory...")
    else:
        # Use the specific list from the config file and construct full paths
        files_to_process = [os.path.join(config.ROOT_FILE_PATH, f) for f in config.FILENAMES]
        print(f"Processing {len(files_to_process)} specific files from config.FILENAMES...")

    if not files_to_process:
        print("No files found to process. Exiting.")
        return

    spill_results = []

    # Main execution loop
    for filepath in sorted(files_to_process):
        result = analyze_spill_filling(filepath)
        if result:
            spill_results.append(result)

    # --- Generate the final plot ---
    if not spill_results:
        print("\nNo valid spill data was processed. Cannot generate plot.")
        return

    # Convert results to a DataFrame for easier plotting
    results_df = pd.DataFrame(spill_results, columns=['spill_num', 'filled_batches'])
    results_df['spill_num'] = pd.to_numeric(results_df['spill_num'])
    results_df = results_df.sort_values('spill_num')

    fig, ax = plt.subplots(figsize=(14, 8)) # Increased height for labels

    # Use the DataFrame index for the x-positions to create evenly spaced bars
    x_positions = np.arange(len(results_df))
    ax.bar(x_positions, results_df['filled_batches'])

    ax.set_xlabel("Spill Number")
    ax.set_ylabel("Number of Filled Batches")
    ax.set_title("Number of Filled Booster Batches per Spill")
    
    # Set the x-axis ticks to the bar positions, and the labels to the spill numbers
    ax.set_xticks(x_positions)
    ax.set_xticklabels(results_df['spill_num'], rotation=90, ha='center', fontsize='small')

    # Make the y-axis grid visible for easier reading
    ax.grid(True, axis='y', linestyle='--')
    
    # Ensure y-axis shows integer ticks from 0 to 7
    ax.set_yticks(range(8))
    ax.set_ylim(bottom=0) # Start y-axis at 0 for bar charts

    # Adjust layout to prevent labels from being cut off
    fig.tight_layout()

    plot_filename = os.path.join(config.PLOTS_DIR, "Filled_Batches_vs_Spill.png")
    plt.savefig(plot_filename, dpi=150)
    plt.close(fig)


    print(f"\nAnalysis complete. Summary plot saved to:\n{plot_filename}")
    
    now = pd.Timestamp.now(tz='America/Chicago')
    print(f"\nAnalysis completed on: {now.strftime('%A, %B %d, %Y at %I:%M:%S %p %Z')}")
    print(f"Location: Batavia, Illinois, United States")

if __name__ == "__main__":
    main()