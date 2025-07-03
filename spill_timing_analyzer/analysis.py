# analysis.py
"""
spill_timing_analyzer analysis 
  Contains the SpillAnalyzer class for processing a single 
  FreqHist_**Hz histogram from a ROOT file for a specific run and spill
"""

import os
import re
import ROOT
import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from scipy.signal import find_peaks, peak_prominences, peak_widths #find_peaks_cwt

class SpillAnalyzer:
    """
    Analyzes a single spill.
    """
    def __init__(self, filepath, config):
        """
        Initializes the analyzer for a single file.
        """
        self.config = config
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.run_num, self.spill_num = self._extract_run_spill()
        
        # Get the time and intensity data from each histogram bin and save them in a pandas dataframe
        self.data = self._load_data()
        
        if self.data.empty or self.data['intensity'].sum() == 0:
            self.max_intensity = np.nan
            self.time_of_max_intensity = np.nan
            self.total_intensity = np.nan
            self.mean_intensity = np.nan
            self.std_intensity = np.nan
            self.duty_factor = np.nan
            self.coefficient_of_variation = np.nan
            self.kurtosis = np.nan
            self.gini = np.nan
            self.peak_intervals = []
            print(f"  - WARNING: No data or zero intensity for {self.filename}.")
        else:
            intensity_values = self.data['intensity'].values # numpy array of intensities
            
            # --- Standard Metrics ---

            # maximum intensity bin value and time
            max_idx = self.data['intensity'].idxmax() # index of the first occurrence of the maximum value
            self.max_intensity = self.data.loc[max_idx, 'intensity']
            self.time_of_max_intensity = self.data.loc[max_idx, 'time_s']

            # Check if there are ties in maximum intensity
            #  using a list of all index labels where the intensity is at its maximum
            max_indices = self.data.index[self.data['intensity'] == self.data['intensity'].max()]
            if len(max_indices) > 1:
                print(f"CHECK: Found {len(max_indices)} occurrences of the max value at indices: {max_indices.tolist()}")

            # Intensity sum, mean, and std    
            self.total_intensity = intensity_values.sum()
            self.mean_intensity  = intensity_values.mean()
            self.std_intensity   = intensity_values.std()

            n_points = len(intensity_values)
            midpoint = n_points // 2 # Find the index for the middle of the intensity array
            self.total_intensity_0_2_seconds = intensity_values[:midpoint].sum() # Sum of the first half of the spill
            self.total_intensity_2_4_seconds = intensity_values[midpoint:].sum() # Sum of the second half of the spill

            # --- Non-Uniformity Metrics ---

            self.duty_factor = self._calculate_duty_factor(intensity_values)
            self.coefficient_of_variation = self.std_intensity / self.mean_intensity if self.mean_intensity > 0 else 0 
            self.kurtosis = kurtosis(intensity_values, fisher=False)
            self.gini = self._calculate_gini(intensity_values)
            
            # --- Spike Frequency Analysis ---
            peak_threshold = self.mean_intensity + 2 * self.std_intensity
            peak_indices, _ = find_peaks(intensity_values, height=peak_threshold)
            
            if len(peak_indices) > 1:
                peak_times = self.data['time_s'].iloc[peak_indices].values
                self.peak_intervals = np.diff(peak_times)
            else:
                self.peak_intervals = []


    def _extract_run_spill(self):
        """Extracts run and spill numbers from the filename."""
        match = re.search(r"run(\d+)\.spill(\d+)\.root", self.filename)
        if match:
            return int(match.group(1)), int(match.group(2))
        return -1, -1

    def _load_data(self):
        """Loads data from the FreqHist_*Hz histogram."""
        try:
            root_file = ROOT.TFile.Open(self.filepath)
            if not root_file or root_file.IsZombie():
                print(f"  - ERROR: Could not open file {self.filepath}")
                return pd.DataFrame()
            hist = root_file.Get(self.config.HISTOGRAM_NAME)
            if not hist:
                print(f"  - ERROR: Histogram '{self.config.HISTOGRAM_NAME}' not found in {self.filename}")
                root_file.Close()
                return pd.DataFrame()

            n_bins = hist.GetNbinsX()
            times = [hist.GetBinCenter(i) for i in range(1, n_bins + 1)]
            intensities = [hist.GetBinContent(i) for i in range(1, n_bins + 1)]
            root_file.Close()
            return pd.DataFrame({'time_s': times, 'intensity': intensities})
        except Exception as e:
            print(f"  - An unexpected error occurred loading data from {self.filename}: {e}")
            return pd.DataFrame()

    @staticmethod
    def _calculate_duty_factor(intensity_chunk):
        """Calculates the non-conditional duty factor: (<I>)^2 / <I^2>."""
        if not isinstance(intensity_chunk, np.ndarray):
            intensity_chunk = np.array(intensity_chunk)
        if intensity_chunk.size == 0:
            return np.nan
        
        mean_intensity_sq = np.mean(np.square(intensity_chunk))
        if np.abs(mean_intensity_sq) < 1e-12: return np.nan
        
        mean_intensity = np.mean(intensity_chunk)
        return np.square(mean_intensity) / mean_intensity_sq
        
    @staticmethod
    def _calculate_gini(array):
        """Calculates the Gini coefficient of a numpy array.
              Alternative measure of non-uniformity. 0 uniform, 1 very non-uniform (one spike).
        """
        if np.amin(array) < 0:
            array -= np.amin(array) #Ensure non-negative values
        array += 0.0000001 # Prevent division by zero 
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))