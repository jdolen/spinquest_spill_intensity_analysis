# analyze_root_file.py
#
# A starter script for to interact with ROOT files using Python.
# This script demonstrates two common ways to access ROOT data:
#   1. Using PyROOT
#   2. Using uproot & pandas
# 
# If you have python installed on your computer,
#  you can try the following recipe to run this script for the first time:
"""
python3 -m venv my_env
source my_env/bin/activate
python3.13 -m pip install --upgrade pip
pip3  install pandas numpy matplotlib seaborn scipy uproot
source $(brew --prefix root)/bin/thisroot.sh
python3 analyze_root_file.py
"""

import ROOT
import uproot
import pandas as pd
import os
import matplotlib.pyplot as plt

# --- 1. SETUP: Define file paths and names ---
# -----------------------------------------------------------------------------
# NOTE FOR STUDENTS: You will need to change this to the correct path on your computer.
ROOT_FILE_PATH = "./"
ROOT_FILENAME = "histograms_run6178.spill1951167.root"
FULL_FILE_PATH = os.path.join(ROOT_FILE_PATH, ROOT_FILENAME)

# Define the name of the specific histogram we want to convert to a DataFrame.
HIST_TO_CONVERT = "FreqHist_10kHz"

# Define a directory where we will save our output images.
OUTPUT_DIR = "histogram_plots"


def save_all_histograms_as_png(filename, output_dir):
    """
    Uses PyROOT to open a ROOT file, find all histograms,
    and save each one as a separate PNG image.
    """
    print(f"\n--- Using PyROOT to save all histograms from {os.path.basename(filename)} ---")
    
    # Create the output directory if it doesn't already exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Open the ROOT file in "READ" mode.
    root_file = ROOT.TFile.Open(filename, "READ")
    
    # Check if the file was opened successfully.
    if not root_file or root_file.IsZombie():
        print(f"Error: Could not open file {filename}")
        return

    # To see what's in the file, we get the "list of keys" (a list of all objects).
    for key in root_file.GetListOfKeys():
        # Get the actual object from the key.
        obj = key.ReadObj()
        
        # Check if the object is a 1D or 2D histogram.
        # TH1 and TH2 are the base classes for all ROOT histograms.
        if isinstance(obj, (ROOT.TH1, ROOT.TH2)):
            hist_name = obj.GetName()
            print(f"Found histogram: {hist_name}. Saving as PNG...")
            
            # Create a "canvas" to draw the histogram on.
            # Think of it as a blank piece of paper.
            canvas = ROOT.TCanvas(f"canvas_{hist_name}", "canvas", 800, 600)
            
            # Draw the histogram onto the canvas.
            obj.Draw()
            
            # Save the canvas as a PNG file in our output directory.
            output_path = os.path.join(output_dir, f"{hist_name}.png")
            canvas.SaveAs(output_path)

    # Good practice to close the file when you're done.
    root_file.Close()
    print("--- PyROOT task complete. ---")


def convert_hist_to_dataframe(filename, hist_name):
    """
    Uses uproot and pandas to read a specific histogram and convert it
    into a pandas DataFrame.
    """
    print(f"\n--- Using uproot to convert '{hist_name}' to a pandas DataFrame ---")
    
    try:
        # uproot provides a very "Python-like" way to open files.
        # The 'with' statement automatically closes the file for us.
        with uproot.open(filename) as root_file:
            # Check if the histogram exists in the file.
            if hist_name not in root_file:
                print(f"Error: Histogram '{hist_name}' not found in {os.path.basename(filename)}")
                return None
                
            # Access the histogram like a dictionary key.
            hist = root_file[hist_name]
            
            # uproot can easily convert the histogram's data to NumPy arrays.
            # This gives us the contents (y-axis) and the bin edges (x-axis).
            intensities, bin_edges = hist.to_numpy()
            
            # The x-axis of a histogram is defined by its bin edges. To get a single
            # value for each bin, we calculate the center of each bin.
            # We do this by averaging the left and right edges of each bin.
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Now, create a pandas DataFrame. A DataFrame is like a spreadsheet.
            # We'll have two columns: one for the time and one for the intensity.
            df = pd.DataFrame({
                'time_s': bin_centers,
                'intensity': intensities
            })
            
            print("Successfully created DataFrame. Here are the first 5 rows:")
            print(df.head())
            
            print("\nDataFrame Info:")
            df.info()
            
            return df
            
    except Exception as e:
        print(f"An error occurred with uproot: {e}")
        return None

def create_example_plot(df, hist_name, output_dir):
    """
    Creates and saves a simple line plot from the intensity DataFrame.
    """
    print("\n--- Creating an example plot from the DataFrame ---")

    # First, make sure we have a valid DataFrame to plot.
    if df is None or df.empty:
        print("DataFrame is empty. Skipping plot creation.")
        return

    # --- Plotting with pandas and Matplotlib ---
    # The .plot() method on a DataFrame is a quick way to create plots.
    # We'll create a line plot of intensity vs. time.
    ax = df.plot(
        x='time_s',
        y='intensity',
        kind='line',         # A line plot is good for time-series data.
        figsize=(12, 7),     # Set the figure size (width, height in inches).
        legend=False,        # We only have one line, so no legend is needed.
        title=f"Spill Intensity vs. Time (from {hist_name})"
    )

    # Customize the labels to be more descriptive.
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Linearized Intensity (arbitrary units)")

    # Add a grid for better readability.
    ax.grid(True, linestyle='--')

    # Ensure the x-axis starts at 0.
    ax.set_xlim(left=0)

    # Get the figure object from the axes to save it.
    fig = ax.get_figure()

    # Define the output path and save the figure.
    output_path = os.path.join(output_dir, "intensity_vs_time_plot.png")
    fig.savefig(output_path, dpi=150) # dpi sets the resolution.

    # It's good practice to close the figure to free up memory.
    plt.close(fig)

    print(f"Successfully saved plot to: {output_path}")


# --- 3. MAIN EXECUTION ---
# -----------------------------------------------------------------------------
# This is the main part of our script that calls the functions we defined above.
if __name__ == "__main__":
    # First, check if the specified ROOT file actually exists.
    if not os.path.exists(FULL_FILE_PATH):
        print(f"FATAL ERROR: The specified ROOT file does not exist at:\n{FULL_FILE_PATH}")
    else:
        # Call the first function to save all histograms as images.
        save_all_histograms_as_png(FULL_FILE_PATH, OUTPUT_DIR)
        
        # Call the second function to convert our specific histogram to a DataFrame.
        intensity_df = convert_hist_to_dataframe(FULL_FILE_PATH, HIST_TO_CONVERT)

        # If the DataFrame was created successfully, make a plot from it.
        if intensity_df is not None:
            create_example_plot(intensity_df, HIST_TO_CONVERT, OUTPUT_DIR)

        # You can add more code here to work with the 'intensity_df' DataFrame!
        
        print("\nScript finished.")