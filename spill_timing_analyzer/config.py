# config.py
"""
Configuration for the spill timing analyzer.
"""

# The full path to the directory where the ROOT files are located.
ROOT_FILE_PATH = "/Users/jdolen/Code/SQDQ/spinquest_rootfiles/"

# The pattern to match for the ROOT files.
FILE_PATTERN = "histograms*.root"

# The specific histogram to analyze in each file.
HISTOGRAM_NAME = "FreqHist_10kHz"
#- FreqHist_53MHz  (0.03 s duration)
#- FreqHist_1MHz   (0.1 s duration)
#- FreqHist_100kHz (1 s duration)
#- FreqHist_10kHz  (full 4 s spill)
#- FreqHist_7_5kHz (full 4 s spill)
#- FreqHist_1kHz   (full 4 s spill)

# The directory where output plots will be saved.
PLOTS_DIR = "plots_"+HISTOGRAM_NAME 
