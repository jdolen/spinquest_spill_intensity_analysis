# check_bins.py
"""
A utility script to loop through ROOT files and check the number of bins
in a specific histogram ('FreqHist_53MHz'). It highlights files
that have an unusually high number of bins.
"""

import os
import glob
import ROOT

def check_histogram_bins():
    """
    Main function to find ROOT files, open them, and check the bin count
    of the 'FreqHist_53MHz' histogram.
    """

    # --- Configuration ---
    # You can modify these paths and patterns to match your setup.
    # These are taken from your config.py file.
    #ROOT_FILE_PATH = "/Users/jdolen/Google Drive/My Drive/00_SpinQuest_DarkQuest/spinquest_rootfiles"
    ROOT_FILE_PATH = "/Users/jdolen/Code/SQDQ_Code/REFACTOR/AnalyzeBIM_53MHz/empty_spill_files"

    FILE_PATTERN = "histograms*.root"
    HISTOGRAM_NAME = "FreqHist_53MHz"
    BIN_THRESHOLD = 2000000
    # --- End Configuration ---

    # Combine the root file directory and the file pattern
    file_search_path = os.path.join(ROOT_FILE_PATH, FILE_PATTERN)

    # Create a list containing the path to every file
    filepaths = glob.glob(file_search_path)

    # Check if any files were found
    if not filepaths:
        print(f"Error: No files found matching the pattern '{file_search_path}'.")
        print("Please check your ROOT_FILE_PATH and FILE_PATTERN.")
        return

    print(f"Found {len(filepaths)} files to check. Processing now...\n")

    # Loop through each found file
    for filepath in sorted(filepaths):
        filename = os.path.basename(filepath)
        
        try:
            # Open the ROOT file in read-only mode
            root_file = ROOT.TFile.Open(filepath)

            # Check if the file is valid
            if not root_file or root_file.IsZombie():
                print(f"{filename:<50} -> ERROR: Could not open file or file is corrupted.")
                continue

            # Get the histogram from the file
            hist = root_file.Get(HISTOGRAM_NAME)

            # Check if the histogram exists
            if not hist:
                print(f"{filename:<50} -> WARNING: Histogram '{HISTOGRAM_NAME}' not found.")
                root_file.Close()
                continue

            # Get the number of bins
            n_bins = hist.GetNbinsX()

            # Check against the threshold and print the result
            if n_bins > BIN_THRESHOLD:
                # Use ANSI escape codes for color (red) to make it stand out
                print(f"\033[91m{filename:<50} -> Bins: {n_bins}  <-- !!! EXCEEDS THRESHOLD !!!\033[0m")
            else:
                print(f"{filename:<50} -> Bins: {n_bins}")

            # Close the file
            root_file.Close()

        except Exception as e:
            print(f"{filename:<50} -> An unexpected error occurred: {e}")

    print("\nCheck complete.")

if __name__ == "__main__":
    # This ensures the script can be run directly from the command line
    check_histogram_bins()