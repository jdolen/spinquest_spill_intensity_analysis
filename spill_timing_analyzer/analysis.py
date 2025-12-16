# analysis.py
"""
spill_timing_analyzer analysis 
  
Description: Contains the SpillAnalyzer class for processing a single 
  FreqHist_**Hz histogram from a ROOT file for a specific run and spill
"""

import os
import re
import ROOT
import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from scipy.signal import find_peaks, peak_prominences, peak_widths #find_peaks_cwt



def get_frequency_from_name(hist_name):
    """Extracts a formatted frequency string from the histogram name."""
    # Find numbers and units (k or M) in the histogram name
    match = re.search(r"(\d+)_?(\d*)\s*([kM])Hz", hist_name.replace(".", "_"))
    if match:
        num1, num2, unit = match.groups()
        # Combine numbers if there's a separator like in "7_5kHz"
        number = num1 + ("." + num2 if num2 else "")
        return f"{number}{unit}Hz"
    return hist_name # Return the original name if no match



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
        self.frequency_label = get_frequency_from_name(config.HISTOGRAM_NAME)
        self.run_num, self.spill_num = self._extract_run_spill()
        self.acnet_data = {}
        self.acnet_timestamp = None 
        self.duty_factor_vs_time = pd.DataFrame() 
        self.intensity_vs_time = pd.DataFrame() 
        self.peaks_df = pd.DataFrame()
        self.top_fft_freq = np.nan
        self.top_fft_mag = np.nan
        self.fft_peaks = []
        self.ranked_fft_peaks = []
        self.magnitude_fft_peaks = []
        self._load_acnet_data()

        # Get the time and intensity data from each histogram bin and save them in a pandas dataframe
        self.data = self._load_data()
        
        if self.data.empty or self.data['intensity'].sum() == 0:
            self.max_intensity = np.nan
            self.time_of_max_intensity = np.nan
            self.total_intensity = np.nan
            self.total_intensity_0_2_seconds = np.nan
            self.total_intensity_2_4_seconds = np.nan
            self.mean_intensity = np.nan
            self.std_intensity = np.nan
            self.duty_factor = np.nan
            self.duty_factor_uncertainty = np.nan
            self.coefficient_of_variation = np.nan
            self.kurtosis = np.nan
            self.gini = np.nan
            self.peak_threshold = np.nan
            self.peak_intervals = []
            self.peak_widths = []
            self.peak_integrals = []
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
            self.duty_factor_uncertainty = self._calculate_duty_factor_uncertainty(intensity_values)

            self.coefficient_of_variation = self.std_intensity / self.mean_intensity if self.mean_intensity > 0 else 0 
            self.kurtosis = kurtosis(intensity_values, fisher=False)
            self.gini = self._calculate_gini(intensity_values)
            
           
            # --- Spike Frequency Analysis ---

            self._analyze_fft()

            #self.peaks_df = pd.DataFrame() 
            # Define a threshold to find significant peaks
            peak_threshold = self.mean_intensity + 2 * self.std_intensity
            self.peak_threshold = peak_threshold 

            # Use find_peaks to get indices and a dictionary of all properties
            peak_indices, properties = find_peaks(intensity_values, 
                                                  height=peak_threshold, 
                                                  width=1, 
                                                  rel_height=0.5,
                                                  prominence=self.std_intensity 
                                                  )

            if len(peak_indices) > 0:
                # Calculate the time step (duration of a single bin)
                time_step = self.data['time_s'].iloc[1] - self.data['time_s'].iloc[0] if len(self.data) > 1 else 0

                # Assemble the DataFrame with all peak properties from find_peaks
                self.peaks_df = pd.DataFrame({
                    'index': peak_indices,
                    'time': self.data['time_s'].iloc[peak_indices].values,
                    'intensity': intensity_values[peak_indices],
                    'prominence': properties['prominences'],
                    'width_s': properties['widths'] * time_step,
                    'width_height': properties['width_heights'],
                    'integral': [intensity_values[int(round(l)):int(round(r))].sum() for l, r in zip(properties['left_ips'], properties['right_ips'])]
                })
            """
            peak_threshold = self.mean_intensity + 2 * self.std_intensity
            peak_indices, properties = find_peaks(intensity_values, height=peak_threshold, width=1)
            
            # Calculate the time step (duration of a single bin)
            time_step = self.data['time_s'].iloc[1] - self.data['time_s'].iloc[0] if len(self.data) > 1 else 0
            
            # If peaks are found
            if len(peak_indices) > 1:
                peak_times = self.data['time_s'].iloc[peak_indices].values
                self.peak_intervals = np.diff(peak_times)
                # The 'widths' are returned in units of bins; convert to seconds
                self.peak_widths = properties["widths"] * time_step

                # --- NEW: Calculate Peak Integrals ---
                integrals = []
                # Get the left and right boundaries for each peak
                left_boundaries = properties["left_ips"]
                right_boundaries = properties["right_ips"]

                # Loop through each peak to calculate its integral
                for i in range(len(peak_indices)):
                    start_index = int(round(left_boundaries[i]))
                    end_index = int(round(right_boundaries[i]))
                    
                    # Sum the intensity values within the peak's boundaries
                    spike_integral = intensity_values[start_index:end_index].sum()
                    integrals.append(spike_integral)
                
                self.peak_integrals = integrals


            # If no peaks are found
            else:
                self.peak_intervals = []
                self.peak_widths = []
                self.peak_integrals = []
            """

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

            n_bins_original = hist.GetNbinsX()

            # Rebin the input histogram if requested
            if self.config.REBIN_N_BINS > 1:
                # Check if the number of bins is divisible by self.config.REBIN_N_BINS 
                if n_bins_original % self.config.REBIN_N_BINS == 0:
                    rebin_factor = n_bins_original // self.config.REBIN_N_BINS
                    hist.Rebin(rebin_factor) 
                else:
                    print(f"  - WARNING: Original bin count ({n_bins_original}) is not a multiple of {self.config.REBIN_N_BINS}.")
                    rebin_factor = n_bins_original // self.config.REBIN_N_BINS
                    if rebin_factor > 0:
                        hist.Rebin(rebin_factor)

            n_bins = hist.GetNbinsX()
            times = [hist.GetBinCenter(i) for i in range(1, n_bins + 1)]
            intensities = [hist.GetBinContent(i) for i in range(1, n_bins + 1)]
            root_file.Close()
            return pd.DataFrame({'time_s': times, 'intensity': intensities})
        except Exception as e:

            # Check if it is a control-c 
            if isinstance(e, KeyboardInterrupt):
                print("\n\nProcess interrupted by user. Exiting gracefully.")
                raise 
            
            # Handle all other exceptions
            print(f"  - An unexpected error occurred loading data from {self.filename}: {e}")
            return pd.DataFrame()


    def _load_acnet_data(self):
        """Loads data from the corresponding Acnet TSV file, reading the timestamp only once."""
        if self.spill_num == -1:
            return

        try:
            # Construct the full path to the TSV file
            spill_num_padded = f"{self.spill_num:09d}"
            tsv_filename = f"spill_{spill_num_padded}_Acnet.tsv"
            #directory = os.path.dirname(self.filepath)
            directory = self.config.ACNET_TSV_PATH

            tsv_filepath = os.path.join(directory, tsv_filename)

            if os.path.exists(tsv_filepath):
                # STEP 1: Read only the first row to get the timestamp.
                first_row_df = pd.read_csv(tsv_filepath, sep='\t', header=None, nrows=1, usecols=[1])
                if not first_row_df.empty:
                    # Convert Unix timestamp (seconds) to a proper datetime object
                    unix_ts = first_row_df.iloc[0, 0]
                    self.acnet_timestamp = pd.to_datetime(unix_ts, unit='s', utc=True)

                # STEP 2: Read the key-value pairs, skipping the redundant timestamp column.
                data_df = pd.read_csv(
                    tsv_filepath,
                    sep='\t',
                    header=None,
                    usecols=[0, 2],
                    names=['key', 'value']
                )
                self.acnet_data = pd.Series(data_df.value.values, index=data_df.key).to_dict()

        except Exception as e:
            print(f"  - WARNING: Could not read or parse TSV file for spill {self.spill_num}: {e}")


    def _calculate_duty_factor_uncertainty(self, intensity_values, n_trials=100):
        """
        Calculates the uncertainty of the duty factor using a Monte Carlo simulation.
        """
        if intensity_values.size == 0:
            return np.nan

        # The uncertainty for each bin is sqrt(intensity)
        # We replace any negative or zero intensity values with a small number to avoid errors in sqrt.
        uncertainties = np.sqrt(np.maximum(intensity_values, 1e-9))

        duty_factors_from_toys = []
        for _ in range(n_trials):
            # Create a "toy" spill by smearing each bin by its uncertainty
            toy_intensities = np.random.normal(loc=intensity_values, scale=uncertainties)

            # Ensure toy intensities are non-negative for the calculation
            toy_intensities[toy_intensities < 0] = 0

            # Calculate the duty factor for this toy spill
            toy_df = self._calculate_duty_factor(toy_intensities)
            if not np.isnan(toy_df):
                duty_factors_from_toys.append(toy_df)

        # The uncertainty is the standard deviation of the results from all the toy spills
        return np.std(duty_factors_from_toys) if duty_factors_from_toys else np.nan

    def _analyze_fft(self):
        """
        Calculates the FFT of the spill intensity, finds the most prominent frequency peak,
        and calculates ranked-choice scores for the top N peaks.
        """
        if self.data.empty:
            return

        intensity = self.data['intensity'].values
        n_points = len(intensity)
        if n_points < 2: return

        time_step = self.data['time_s'].iloc[1] - self.data['time_s'].iloc[0]

        # Calculate FFT and frequencies
        fft_vals = np.fft.fft(intensity)
        fft_freq = np.fft.fftfreq(n_points, d=time_step)

        # Consider only the positive frequencies
        positive_freq_mask = fft_freq > 0
        if not np.any(positive_freq_mask): return

        fft_magnitudes = np.abs(fft_vals)[positive_freq_mask]
        fft_frequencies = fft_freq[positive_freq_mask]

        if 119 < self.top_fft_freq < 121:
            print(f"DEBUG (Spill {self.spill_num}): Found peak near 120 Hz. Raw value: {self.top_fft_freq:.15f}")



        if len(fft_magnitudes) == 0: return

        # --- Find all significant peaks ---
        peaks, properties = find_peaks(fft_magnitudes,
                                       height=np.mean(fft_magnitudes),
                                       prominence=1,
                                       distance=5)

        if len(peaks) == 0: return

        peak_freqs = fft_frequencies[peaks]
        peak_mags = properties['peak_heights']

        # --- Original "Top Peak" Calculation ---
        max_mag_idx = np.argmax(peak_mags)
        self.top_fft_freq = peak_freqs[max_mag_idx]
        self.top_fft_mag = peak_mags[max_mag_idx]

        if 119 < self.top_fft_freq < 121:
            print(f"DEBUG (Spill {self.spill_num}): Found peak near 120 Hz. Raw value: {self.top_fft_freq:.15f}")

        # --- NEW: Ranked-Choice Scoring (Method 1) ---
        # Combine magnitudes and frequencies, then sort by magnitude descending
        sorted_peaks = sorted(zip(peak_mags, peak_freqs), key=lambda x: x[0], reverse=True)

        n_ranked = self.config.N_RANKED_PEAKS
        for i, (mag, freq) in enumerate(sorted_peaks[:n_ranked]):
            # The #1 peak gets N points, #2 gets N-1, etc.
            score = n_ranked - i
            self.ranked_fft_peaks.append((freq, score))

        self.magnitude_fft_peaks = list(zip(peak_freqs, peak_mags))

    def get_fft_power_spectrum(self, max_freq=2500):
            """
            Calculates and returns the FFT power spectrum and corresponding frequencies.
            This is used for building the 2D spectrogram.
            """
            if self.data.empty or len(self.data) < 2:
                return None, None

            intensity = self.data['intensity'].values
            n_points = len(intensity)
            time_step = self.data['time_s'].iloc[1] - self.data['time_s'].iloc[0]

            fft_vals = np.fft.fft(intensity)
            fft_freq = np.fft.fftfreq(n_points, d=time_step)

            # Calculate power spectrum (magnitude squared)
            power_spectrum = np.abs(fft_vals)**2

            # Return only the positive frequencies up to max_freq
            positive_freq_mask = (fft_freq > 0) & (fft_freq <= max_freq)
            
            return fft_freq[positive_freq_mask], power_spectrum[positive_freq_mask]

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

    def calculate_duty_factor_vs_time(self):
        """
        Calculates the duty factor and its uncertainty over discrete time intervals.
         The results are stored in the self.duty_factor_vs_time Pandas DataFrame.
        """
        if self.data.empty or self.config.DUTY_FACTOR_INTERVAL_S <= 0:
            self.duty_factor_vs_time = pd.DataFrame()
            return

        # Over what time interval in seconds do we want to calculate the duty factor?    
        interval_s = self.config.DUTY_FACTOR_INTERVAL_S

        # Create a Pandas Series from the Series self.data['time_s'] where each series value is an integer group ID number
        #  group ID number for each intensity bin are calculated by dividing its time by the interval width and using floor
        #    Example for interval_s = 1 s
        #    0.7 s / 1.0 s = 0.7  -> floor(0.7) is 0
        #    1.3 s / 1.0 s = 1.3  -> floor(1.3) is 1
        time_groups = np.floor(self.data['time_s'] / interval_s)

        # The time_groups Series originally has name time_s since that is where the data comes from. 
        #   We need to change this to a correct label.
        time_groups.name = 'time_group_index'

        # Calculate both the duty factor and its uncertainty for each time group
        #   groupby() does the work of grouping together all of the intensity bins with a given group ID
        #   and apply() allows us to run the calculation function on each of those groups
        # We end up with two pandas series, each with index=group ID 
        duty_factors = self.data.groupby(time_groups)['intensity'].apply(self._calculate_duty_factor)
        duty_factor_uncert = self.data.groupby(time_groups)['intensity'].apply(self._calculate_duty_factor_uncertainty)

        # Combine both series into a DataFrame
        #   Note .reset_index() turns the index 'time_group_index' into a column 
        dataframe_grouped_by_time = pd.DataFrame({
            'duty_factor': duty_factors,
            'duty_factor_uncertainty': duty_factor_uncert
        }).reset_index()

        # Calculate the time_center for each group and save it as a dataframe column
        dataframe_grouped_by_time['time_center'] = (dataframe_grouped_by_time['time_group_index'] + 0.5) * interval_s

        # Store the instance attribute dataframe with four columns (group index, DF, DF uncert, time_center)
        self.duty_factor_vs_time = dataframe_grouped_by_time

    def calculate_intensity_vs_time(self):
        """
        Calculates the total integrated intensity and its uncertainty over discrete time intervals.
        The results are stored in the self.intensity_vs_time DataFrame.
        """
        if self.data.empty or self.config.DUTY_FACTOR_INTERVAL_S <= 0:
            self.intensity_vs_time = pd.DataFrame()
            return

        # Over what time interval in seconds do we want to integrate the intensity?    
        interval_s = self.config.DUTY_FACTOR_INTERVAL_S

        # Create a Pandas Series from the Series self.data['time_s'] where each series value is an integer group ID number
        #  group ID number for each intensity bin are calculated by dividing its time by the interval width and using floor
        #    Example for interval_s = 1 s
        #    0.7 s / 1.0 s = 0.7  -> floor(0.7) is 0
        #    1.3 s / 1.0 s = 1.3  -> floor(1.3) is 1
        time_groups = np.floor(self.data['time_s'] / interval_s)
        time_groups.name = 'time_group_index'

        # Group the intensity data into the time_groups then sum together the intensity in that group 
        #   Then rename the index and reset_index() to turn the index into a column and in doing so turn the Series into a Dataframe
        dataframe_grouped_by_time = self.data.groupby(time_groups)['intensity'].sum().rename('total_intensity').reset_index()

        # For counting stats, the uncertainty of a sum is the square root of the sum
        dataframe_grouped_by_time['total_intensity_err'] = np.sqrt(dataframe_grouped_by_time['total_intensity'])

        # Calculate the time_center for each group and save it as a dataframe column
        dataframe_grouped_by_time['time_center'] = (dataframe_grouped_by_time['time_group_index'] + 0.5) * interval_s

        # Store the instance attribute dataframe with four columns (group index, summed intensity, intensity uncert, time_center)
        self.intensity_vs_time = dataframe_grouped_by_time

   