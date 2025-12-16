# config.py
"""
spill_timing_analyzer config 
   
Description: Configuration for the spill timing analyzer.
"""

#ACNET_TSV_PATH = "/Users/jdolen/Google Drive/My Drive/00_SpinQuest_DarkQuest/spinquest_slowcontrol/all_acnet_tsv_files/"
ACNET_TSV_PATH = "/Users/jdolen/NoBackup/ACNET/all_acnet_tsv_files/"

# The full path to the directory where the ROOT files are located.
#ROOT_FILE_PATH = "/Users/jdolen/Code/SQDQ/spinquest_rootfiles/"
ROOT_FILE_PATH = "/Users/jdolen/Google Drive/My Drive/00_SpinQuest_DarkQuest/spinquest_rootfiles"
#ROOT_FILE_PATH = "/Users/jdolen/Code/SQDQ_Code/REFACTOR/AnalyzeBIM_53MHz/empty_spill_files"

# The pattern to match for the ROOT files.
FILE_PATTERN = "histograms*.root"
#FILE_PATTERN = "histograms*run6140*.root"
#FILE_PATTERN = "histograms*run6170*.root"
#FILE_PATTERN = "histograms*run6171*.root"
#FILE_PATTERN = "histograms*run6097*.root"

# The specific histogram to analyze in each file.
#  - FreqHist_53MHz  (0.03 s duration, 1593118 bins, 19 ns per bin [one bucket])
#  - FreqHist_1MHz   (0.1 s duration, 100000 bins, 1 microsec per bin)
#  - FreqHist_100kHz (1 s duration, 100000 bins, 10 microsec per bin)
#  - FreqHist_10kHz  (full 4 s spill, 40000 bins, 100 microsec per bin)
#  - FreqHist_7_5kHz (full 4 s spill, 30000 bins, 133.3 microsec per bin)
#  - FreqHist_1kHz   (full 4 s spill, 4000 bins, 1 millisec per bin)
HISTOGRAM_NAME = "FreqHist_10kHz"

# The time interval in seconds for calculating duty factor within a spill.
DUTY_FACTOR_INTERVAL_S = 1#0.5

# The percentile for selecting high-intensity spills for the detailed analysis.
# For example, 90 means we analyze spills in the top 10% of total intensity.
HIGH_INTENSITY_SPILL_PERCENTILE = 95

# The percentile for selecting spills with the highest single-bin intensity (spike).
# For example, 95 means we analyze spills in the top 5% with the largest spikes.
HIGH_SPIKE_SPILL_PERCENTILE = 95


# The number of top spikes (from all spills) to generate a zoomed-in plot for.
N_TOP_SPIKES_TO_PLOT = 20

# The total width of the zoomed-in time window in seconds.
SPIKE_ZOOM_WINDOW_S = 0.15

# Number of peaks in the FFT distribution to consider when performing ranked choice
N_RANKED_PEAKS = 10 

# If you wish to rebin the input histogram set it here (set the number of bins to be some number greater than 1)
#   If you don't wish to rebin keep this <1
REBIN_N_BINS = -1

# The directory where output plots will be saved.
if REBIN_N_BINS >1:
   PLOTS_DIR = "plots_spill_timing_"+HISTOGRAM_NAME+"_nbins_"+str(REBIN_N_BINS) 
else:
   PLOTS_DIR = "plots_spill_timing_"+HISTOGRAM_NAME 



   

