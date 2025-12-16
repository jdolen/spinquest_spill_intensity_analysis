# config.py
"""
turn_structure_analyzer config

Description: Configuration 53 MHz analysis of the SpinQuest BIM data sampled from a main injector slow spill. 
"""

# --- File Locations ---
ROOT_FILE_PATH = "/Users/jdolen/Google Drive/My Drive/00_SpinQuest_DarkQuest/spinquest_rootfiles"
#ROOT_FILE_PATH = "/Users/jdolen/Code/SQDQ_Code/REFACTOR/AnalyzeBIM_53MHz/empty_spill_files"


ACNET_TSV_PATH = "/Users/jdolen/Google Drive/My Drive/00_SpinQuest_DarkQuest/spinquest_slowcontrol/all_acnet_tsv_files/" 
#ACNET_TSV_PATH = "/Users/jdolen/Google Drive/My Drive/00_SpinQuest_DarkQuest/spinquest_slowcontrol/" 
#ACNET_TSV_PATH = "/Users/jdolen/Google\ Drive/My\ Drive/00_SpinQuest_DarkQuest/slowcontrol"

# --- Choose if you want to process selected files or all of the files ---
# Set to True to loop over all .root files in ROOT_FILE_PATH.
# Set to False to loop over only the files specified in the FILENAMES list below.
PROCESS_ALL_FILES_IN_DIR = False

# --- List of selected files to process ---

FILENAMES = [
    "histograms_run6127.spill1937428.root",
    "histograms_run6080.spill1933868.root",
    "histograms_run6111.spill1936211.root",
    "histograms_run6118.spill1936619.root",
    #"histograms_run6135.spill1938471.root",
    #"histograms_run6136.spill1938513.root",
    #"histograms_run6137.spill1938639.root",
    #"histograms_run6138.spill1938831.root",
    #"histograms_run6140.spill1940023.root",
    #"histograms_run6140.spill1940024.root",
    #"histograms_run6140.spill1940025.root",
    #"histograms_run6140.spill1940029.root",
    #"histograms_run6146.spill1940363.root",
    #"histograms_run6153.spill1941850.root",
    #"histograms_run6155.spill1941914.root",
    #"histograms_run6155.spill1941930.root",
    #"histograms_run6155.spill1941944.root",
    #"histograms_run6156.spill1942474.root",
    #"histograms_run6168.spill1949016.root",
    #"histograms_run6168.spill1949006.root",
    #"histograms_run6170.spill1950276.root",
    #"histograms_run6170.spill1950272.root",
    #"histograms_run6170.spill1950248.root",
    #"histograms_run6174.spill1951103.root",
    #"histograms_run6175.spill1951113.root",
    #"histograms_run6176.spill1951132.root",
    #"histograms_run6177.spill1951158.root",
    #"histograms_run6178.spill1951185.root",
]

"""
FILENAMES = [
    "histograms_run6172.spill1950835.root",
    "histograms_run6169.spill1950050.root",
    "histograms_run6155.spill1942435.root",
    "histograms_run6163.spill1948648.root",
    "histograms_run6155.spill1942368.root",
    "histograms_run6139.spill1939952.root",
    "histograms_run6129.spill1937469.root",
    "histograms_run6118.spill1936700.root",
    "histograms_run6110.spill1936091.root",
    "histograms_run6105.spill1935787.root",
    "histograms_run6094.spill1934431.root",
    "histograms_run6081.spill1933911.root",
    "histograms_run6176.spill1951131.root",
]
"""
# --- File-specific parameters ---
# Keys should match the filenames in the list above.
FILE_PARAMETERS = {
    
    "histograms_run6127.spill1937428.root": {"start_bucket": 86, "cond_df_threshold": 100.0},

    #"histograms_run6080.spill1933868.root": {"start_bucket": 86, "cond_df_threshold": 100.0},
    #"histograms_run6111.spill1936211.root": {"start_bucket": 86, "cond_df_threshold": 100.0},
    #"histograms_run6118.spill1936619.root": {"start_bucket": 86, "cond_df_threshold": 100.0},
    #"histograms_run6135.spill1938471.root": {"start_bucket": 86, "cond_df_threshold": 100.0},
    #"histograms_run6136.spill1938513.root": {"start_bucket": 86, "cond_df_threshold": 100.0},
    #"histograms_run6137.spill1938639.root": {"start_bucket": 86, "cond_df_threshold": 100.0},
    #"histograms_run6138.spill1938831.root": {"start_bucket": 86, "cond_df_threshold": 100.0},
    #"histograms_run6140.spill1940023.root": {"start_bucket": 81, "cond_df_threshold": 100.0},
    #"histograms_run6140.spill1940024.root": {"start_bucket": 81, "cond_df_threshold": 100.0},
    #"histograms_run6140.spill1940025.root": {"start_bucket": 81, "cond_df_threshold": 100.0},
    #"histograms_run6140.spill1940029.root": {"start_bucket": 81, "cond_df_threshold": 100.0},
    #"histograms_run6146.spill1940363.root": {"start_bucket": 81, "cond_df_threshold": 100.0},
    #"histograms_run6153.spill1941850.root": {"start_bucket": 80, "cond_df_threshold": 150.0},
    #"histograms_run6155.spill1941914.root": {"start_bucket": 80, "cond_df_threshold": 150.0},
    #"histograms_run6155.spill1941930.root": {"start_bucket": 80, "cond_df_threshold": 150.0},
    #"histograms_run6155.spill1941944.root": {"start_bucket": 80, "cond_df_threshold": 150.0},
    #"histograms_run6156.spill1942474.root": {"start_bucket": 80, "cond_df_threshold": 150.0},
    #"histograms_run6168.spill1949016.root": {"start_bucket": 80, "cond_df_threshold": 150.0},
    #"histograms_run6168.spill1949006.root": {"start_bucket": 80, "cond_df_threshold": 150.0},
    #"histograms_run6170.spill1950276.root": {"start_bucket": 80, "cond_df_threshold": 150.0},
    #"histograms_run6170.spill1950272.root": {"start_bucket": 80, "cond_df_threshold": 150.0},
    #"histograms_run6170.spill1950248.root": {"start_bucket": 80, "cond_df_threshold": 150.0},
    #"histograms_run6174.spill1951103.root": {"start_bucket": 80, "cond_df_threshold": 150.0},
    #"histograms_run6175.spill1951113.root": {"start_bucket": 80, "cond_df_threshold": 150.0},
    #"histograms_run6176.spill1951132.root": {"start_bucket": 80, "cond_df_threshold": 150.0},
    #"histograms_run6177.spill1951158.root": {"start_bucket": 80, "cond_df_threshold": 150.0},
    #"histograms_run6178.spill1951185.root": {"start_bucket": 80, "cond_df_threshold": 150.0},
   
}


# --- Analysis Parameters ---
BUCKETS_PER_BATCH = 84   # Number of RF Buckets in a booster batch
NUM_BATCHES_PER_TURN = 7 # Number of booster batches that fit within the main injector. Every turn a fraction of the protons in each of these seven batches are extracted during a slow spill. Often some of the batches are empty.
HISTOGRAM_NAME = "FreqHist_53MHz" # Name of the TH1 histogram in the BIM ROOT file with full bucket information

AUTO_FIND_START_BUCKET = False         # Set to True to enable automatic start bucket finding
START_FINDER_NOISE_WINDOW = 60      # Number of initial buckets to use for noise calculation
START_FINDER_THRESHOLD_STD = 5.0      # How many standard deviations above noise to set the threshold
START_FINDER_CONFIRM_WINDOW = 5  # Require the average of this many buckets to cross the threshold (to avoid one noisy bucket starting the batch finder)

# Set to True to enable the iterative refinement of the start bucket
REFINE_START_BUCKET_ITERATIVELY = True
# Number of buckets on either side of the initial guess to search for a better start
REFINEMENT_SEARCH_WINDOW = 15 
# Number of turns to average over when checking the intensity of Batch 7
REFINEMENT_SAMPLE_TURNS = 10

# --- Specific Parameters to batch_filler_analyzer (quick check of which buckets are full)
#FILLED_BATCH_INTENSITY_THRESHOLD_FRAC = 0.25 # A batch is "filled" if its integral is > 5% of the spill's average batch integral
FILLED_BATCH_NOISE_MULTIPLIER = 2.0 # A batch is "filled" if its integral is > 10x the integrated noise of a batch-sized window.
FILLER_ANALYZER_MAX_TURNS = 200 # Number of turns to analyze in the batch_filler_analyzer script for speed.


# --- Plotting Parameters ---
PLOTS_DIR = "Plots_TurnStructureAna_AUTO_FIND" # Directory to save output plots
PLOT_COLORS = ['red', 'blue', 'orange', 'green', 'purple', 'cyan', 'magenta'] # Unique color for each batch (1-7)

# --- Per-Turn Plotting Conditions ---
# Plot full bucket by bucket plots only for specific turns:
#  Plot the first N turns regardless of intensity
PLOT_FIRST_N_TURNS = 3
#  Plot any turn whose max intensity is above this fraction of the spill's max intensity
PLOT_INTENSITY_THRESHOLD_FRAC = 0.85
