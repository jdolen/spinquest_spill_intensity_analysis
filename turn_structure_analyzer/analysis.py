# analysis.py
"""
turn_structure_analyzer analysis script

Contains the TurnStructureAnalyzer class responsible for loading data from a ROOT file
from the SpinQuest BIM and performing all statistical calculations for a single slow spill.
"""
import ROOT
import pandas as pd
import numpy as np
import re
import traceback
import os
from scipy.signal import find_peaks

class TurnStructureAnalyzer:
    """Handles data loading and analysis for a single slow spill ROOT file."""

    def __init__(self, filepath, config):
        """
        Initializes the analyzer for a single file.

        Args:
            filepath (str): The full path to the ROOT file.
            config (dict): A dictionary of configuration parameters.
        """
        self.filepath = filepath
        self.filename = os.path.basename(filepath) # Gets just the filename

        print(f"  Initializing analyzer for {self.filename}...")


        self.config = config
        self.run_num, self.spill_num = self._extract_run_spill()
        
        # This dictionary will hold all calculated results
        self.results = {}
        self.results['start_finder_stats'] = {}
        self.results['acnet_data'] = self._load_acnet_data()
        self.intensity_series = self._load_data()

        if self.intensity_series.empty:
            raise ValueError("Intensity series is empty after loading.")

        # 1. First, try to find the start bucket automatically if enabled
        if self.config.get('AUTO_FIND_START_BUCKET', False):
            #found_start = self._find_start_bucket()
            #if found_start is not None:
            #    print(f"  Initial start bucket guess: {found_start}")
            #    # Use the automatically found start bucket
            #    self.config['start_bucket'] = found_start
            #else:
            #    print("  - Auto start finder failed. Using start_bucket from config if available.")
            found_start = 80   # this may be crazy   
        
        # 2. Second, run the refinement if enabled and we have ANY start bucket to work with
        if self.config.get('REFINE_START_BUCKET_ITERATIVELY', False):
            # Check if we have a start bucket from either the auto-finder or the manual config
            initial_guess = 80
            if initial_guess is not None:
                refined_start = self._refine_start_bucket_iteratively()
                if refined_start is not None:
                    print(f"  Refined start bucket to: {refined_start} (was {initial_guess})")
                    self.config['start_bucket'] = refined_start
            else:
                print("  - Skipping refinement: No initial start bucket available.")
       

    def _find_start_bucket(self):
        """
        Finds the first bucket of the spill by detecting a leading edge.
        This version uses a rolling average to confirm the edge is not a single noise spike.
        Returns the integer position of the start bucket, or None if not found.
        """
        try:
            noise_window = self.config.get('START_FINDER_NOISE_WINDOW', 2000)
            std_factor = self.config.get('START_FINDER_THRESHOLD_STD', 5.0)
            confirm_window = self.config.get('START_FINDER_CONFIRM_WINDOW', 3)

            # 1. Establish the baseline noise
            noise_region = self.intensity_series.iloc[:noise_window]
            noise_mean = noise_region.mean()
            noise_std = noise_region.std()

            # 2. Set a threshold
            threshold = noise_mean + (std_factor * noise_std)

            print(f"  [Debug] Noise Mean: {noise_mean:.2f}")
            print(f"  [Debug] Noise Std Dev: {noise_std:.2f}")
            print(f"  [Debug] Calculated Threshold: {threshold:.2f}")

            # Store these stats for the debug plot
            self.results['start_finder_stats'] = {
                'noise_mean': noise_mean,
                'threshold': threshold,
                'noise_std': noise_std
            }

            # 3. Create a rolling average of the intensity
            rolling_avg = self.intensity_series.rolling(window=confirm_window).mean()

            # 4. Find the first position where this rolling average crosses the threshold
            # .fillna(False) handles the NaN values at the start of the rolling average
            boolean_values = (rolling_avg > threshold).fillna(False).values
            end_of_window_pos = np.argmax(boolean_values)

            # Check if a valid edge was found
            if end_of_window_pos == 0 and not boolean_values[0]:
                print("  - Warning: Could not find a confirmed leading edge. Automatic start finder failed.")
                return None
            
            # --- 🔽 CORRECTED LOGIC 🔽 ---
            # The start bucket is the beginning of the first window that crossed the threshold.
            start_bucket_pos = end_of_window_pos - (confirm_window - 1)
            
            # Ensure the position isn't negative (a rare edge case)
            start_bucket_pos = max(0, start_bucket_pos)
            
            return int(start_bucket_pos)
            # --- 🔼 END OF CORRECTION 🔼 ---

        except Exception as e:
            print(f"  - Warning: Error during automatic start bucket finding: {e}")
            return None

    def _refine_start_bucket_iteratively(self):
        """
        Iteratively searches around an initial guess for the start bucket
        that maximizes a Figure of Merit (FOM = Intensity B1 / Intensity B7).
        """
        initial_guess = self.config.get('start_bucket')
        if initial_guess is None: return None

        search_window = self.config.get('REFINEMENT_SEARCH_WINDOW', 50)
        sample_turns = self.config.get('REFINEMENT_SAMPLE_TURNS', 200)
        
        buckets_per_batch = self.config['BUCKETS_PER_BATCH']
        num_batches = self.config['NUM_BATCHES_PER_TURN']
        turn_len = buckets_per_batch * num_batches

        search_range = range(initial_guess - search_window, initial_guess + search_window)
        
        # Get a small value to prevent division by zero, based on noise level
        noise_stats = self.results.get('start_finder_stats', {})
        epsilon = noise_stats.get('noise_mean', 1e-9)
        if epsilon <= 0: # Ensure epsilon is positive
            epsilon = 1e-9

        batch7_intensities = []
        fom_values = []

        # --- CORRECTED LOOP STRUCTURE ---
        # This entire loop calculates the FOM for each candidate start bucket.
        for test_start in search_range:
            
            # Ensure the search range is valid within the data
            if test_start < 0 or (test_start + (sample_turns * turn_len)) > len(self.intensity_series):
                batch7_intensities.append(np.inf) # Use infinity for invalid start points
                fom_values.append(-1) # Use a negative FOM for invalid points
                continue

            # Initialize accumulators for this specific test_start
            total_b1_intensity = 0
            total_b7_intensity = 0
            
            # Sum the intensity of Batch 1 and Batch 7 over the sample of turns
            for i in range(sample_turns):
                turn_start = test_start + (i * turn_len)
                
                # Batch 1 calculation
                b1_start_idx = turn_start
                b1_end_idx = b1_start_idx + buckets_per_batch
                total_b1_intensity += self.intensity_series.iloc[b1_start_idx:b1_end_idx].sum()

                # Batch 7 calculation
                b7_start_idx = turn_start + (num_batches - 1) * buckets_per_batch
                b7_end_idx = b7_start_idx + buckets_per_batch
                total_b7_intensity += self.intensity_series.iloc[b7_start_idx:b7_end_idx].sum()
            
            # *** CRITICAL FIX: These lines must be indented inside this loop ***
            # They append the results for the current test_start.
            batch7_intensities.append(total_b7_intensity)
            
            fom = total_b1_intensity / (total_b7_intensity + epsilon)
            fom_values.append(fom)

        # --- End of corrected loop ---

        if not fom_values: return None
        
        max_fom_index = np.argmax(fom_values)
        best_start_bucket = search_range[max_fom_index]

        # Save data for the debug plots
        self.results['refinement_debug_data'] = {
            'search_range': list(search_range),
            'batch7_intensities': batch7_intensities,
            'fom_values': fom_values,
            'best_start': best_start_bucket,
            'initial_guess': initial_guess
        }
        
        return best_start_bucket

    def run_analysis(self):
        """Public method to run the full analysis pipeline."""
        print("  Running turn-by-turn analysis...")
        self._process_turns()
        self._find_intensity_peaks() 
        self._calculate_aggregate_stats()
        self._count_filled_batches()

        print("  Analysis complete.")

    # --------------------------------------------------------------------------
    # Data Loading and Preparation
    # --------------------------------------------------------------------------

    def _extract_run_spill(self):
        """Extracts run and spill numbers from the filename using regex."""
        match = re.search(r"run(\d+)\.spill(\d+)\.root", self.filename)
        if match:
            return match.group(1), match.group(2)
        print(f"Warning: Could not extract run/spill from '{self.filename}'.")
        return "UNK", "UNK"

    def _load_data(self):
        """Loads intensity data from the specified histogram in the ROOT file."""
        try:
            root_file = ROOT.TFile(self.filepath)
            if not root_file or root_file.IsZombie():
                raise FileNotFoundError(f"Error opening ROOT file: {self.filename}")

            hist = root_file.Get(self.config['HISTOGRAM_NAME'])
            if not hist:
                raise ValueError(f"Histogram '{self.config['HISTOGRAM_NAME']}' not found.")

            n_bins = hist.GetNbinsX()
            bin_centers = np.array([hist.GetBinCenter(i) for i in range(1, n_bins + 1)], dtype=np.float64)
            bin_contents = np.array([hist.GetBinContent(i) for i in range(1, n_bins + 1)], dtype=np.float64)
            
            root_file.Close()

            relative_times_ns = (bin_centers - bin_centers[0]) * 1e9 if len(bin_centers) > 0 else []
            series = pd.Series(bin_contents, index=relative_times_ns, name="Intensity")
            print(f"  Loaded {series.size} data points.")
            return series
        except Exception as e:
            print(f"An error occurred during data loading: {e}")
            traceback.print_exc()
            return pd.Series(dtype=np.float64)

    def _load_acnet_data(self):
        """Loads corresponding Acnet TSV data for the spill."""
        acnet_path = self.config.get('ACNET_TSV_PATH')
        if not acnet_path:
            print("  - Warning: ACNET_TSV_PATH not set in config. Skipping TSV load.")
            return {}

        if self.spill_num == "UNK":
            print("  - Warning: Could not determine spill number. Skipping TSV load.")
            return {}

        try:
            # Format spill number to be 9 digits with leading zeros (e.g., 1937428 -> 001937428)
            padded_spill_num = self.spill_num.zfill(9)
            tsv_filename = f"spill_{padded_spill_num}_Acnet.tsv"
            tsv_filepath = os.path.join(acnet_path, tsv_filename)

            if not os.path.exists(tsv_filepath):
                #print(f"  - Warning: Acnet TSV file not found: {tsv_filename}")
                print(f"  - Warning: Acnet TSV file not found. Full path searched:\n    {tsv_filepath}")

                return {}
            
            df = pd.read_csv(
                tsv_filepath, 
                sep='\t', 
                header=None, 
                names=['parameter', 'timestamp', 'value', 'status']
            )

            # Convert the DataFrame to a {parameter: value} dictionary for easy access
            acnet_dict = df.set_index('parameter')['value'].to_dict()
            
            print(f"  Successfully loaded data for {len(acnet_dict)} parameters from {tsv_filename}.")
            return acnet_dict

        except Exception as e:
            print(f"  - Warning: Error loading or parsing Acnet TSV file: {e}")
            return {}


    # --------------------------------------------------------------------------
    # Core Turn Processing
    # --------------------------------------------------------------------------

    def _process_turns(self, max_turns=None):
        """
        Groups data into batches and turns and calculates per-turn metrics.

        Args:
            max_turns (int, optional): If provided, stops processing after this many turns.
                                       Defaults to None, which processes the whole spill.
        """
        start_bucket = self.config.get("start_bucket", 87)
        buckets_per_batch = self.config['BUCKETS_PER_BATCH']
        num_batches = self.config['NUM_BATCHES_PER_TURN']
        expected_turn_len = buckets_per_batch * num_batches   # 7 batches * 84 buckets/batch = 588 buckets
        
        all_turn_metrics = []
        all_intensities_2d = []
        all_bucket_indices_2d = []
        all_batch_dfs_by_turn = []
        all_batch_max_intensities_by_turn = []
        all_batch_data_agg = {f"Batch {i+1}": [] for i in range(num_batches)}

        current_bucket = start_bucket
        turn_counter = 0

        while current_bucket + expected_turn_len <= len(self.intensity_series):
            
            if max_turns is not None and turn_counter >= max_turns:
                print(f"  Reached max_turns limit of {max_turns}. Stopping turn processing.")
                break


            # Select all buckets from one individual turn and store them in a series
            turn_series = self.intensity_series.iloc[current_bucket : current_bucket + expected_turn_len].copy()
            turn_series.index = np.arange(expected_turn_len) # Reset index to be 0 to N-1

            # --- Calculate metrics for this single turn ---
            metrics = {'turn': turn_counter}
            # Split the turn into seven batches 
            batch_data_chunks = np.array_split(turn_series.values, num_batches)

            # Append batch data for aggregate statistics
            for i, chunk in enumerate(batch_data_chunks):
                all_batch_data_agg[f"Batch {i+1}"].extend(chunk)

            # Basic turn stats
            metrics['mean'] = turn_series.mean()
            metrics['median'] = turn_series.median()
            metrics['q1'] = turn_series.quantile(0.25)
            metrics['q3'] = turn_series.quantile(0.75)
            metrics['max'] = turn_series.max()

            # Duty factors for combined batches
            metrics['df_overall'] = self._calculate_duty_factor(turn_series.values)
            cond_threshold = self.config.get('cond_df_threshold', 100.0)
            metrics['df_cond'] = self._calculate_conditional_duty_factor(turn_series.values, cond_threshold)           
            metrics['df_b12'] = self._calculate_duty_factor(turn_series.iloc[0 : 2 * buckets_per_batch].values)
            metrics['df_b14'] = self._calculate_duty_factor(turn_series.iloc[0 : 4 * buckets_per_batch].values)
            metrics['fom_b12'] = self._calculate_figure_of_merit(turn_series.iloc[0 : 2 * buckets_per_batch].values)
            metrics['fom_b14'] = self._calculate_figure_of_merit(turn_series.iloc[0 : 4 * buckets_per_batch].values)

            # Integrated intensity for the whole turn
            metrics['integrated_intensity'] = self._calculate_integrated_intensity(turn_series.values)

            # Per-batch integrated intensities
            batch_integrated_intensities = [self._calculate_integrated_intensity(chunk) for chunk in batch_data_chunks]
            for i, integral in enumerate(batch_integrated_intensities):
                metrics[f'integrated_intensity_batch_{i+1}'] = integral
            
            # Find which batch has the highest integrated intensity
            metrics['most_intense_batch_by_sum'] = np.nanargmax(batch_integrated_intensities) + 1 if not np.all(np.isnan(batch_integrated_intensities)) else np.nan
        

            # Per-batch metrics
            batch_dfs = [self._calculate_duty_factor(chunk) for chunk in batch_data_chunks]
            batch_max_intensities = [chunk.max() if chunk.size > 0 else np.nan for chunk in batch_data_chunks]
            
            # Add batch DFs to the main metrics dict for this turn
            for i, df in enumerate(batch_dfs):
                metrics[f'df_batch_{i+1}'] = df
            
            metrics['batch_with_max_bucket'] = np.nanargmax(batch_max_intensities) + 1 if not np.all(np.isnan(batch_max_intensities)) else np.nan
            all_turn_metrics.append(metrics)
            
            # Store data for aggregate plots
            all_intensities_2d.extend(turn_series.values)
            all_bucket_indices_2d.extend(turn_series.index)
            all_batch_dfs_by_turn.append(batch_dfs)
            all_batch_max_intensities_by_turn.append(batch_max_intensities)

            current_bucket += expected_turn_len
            turn_counter += 1
        
        # Store results in the main dictionary
        self.results['turn_df'] = pd.DataFrame(all_turn_metrics).set_index('turn')
        self.results['all_batch_data'] = all_batch_data_agg 
        self.results['raw_intensity_series'] = self.intensity_series
        self.results['intensity_vs_bucket_data'] = {
            'indices': all_bucket_indices_2d,
            'intensities': all_intensities_2d
        }
        # Transpose to get shape (num_batches, num_turns)
        self.results['batch_df_heatmap_data'] = np.array(all_batch_dfs_by_turn).T
        self.results['batch_max_intensity_heatmap_data'] = np.array(all_batch_max_intensities_by_turn).T

    def _find_intensity_peaks(self):
        """Finds significant peaks in the integrated intensity per turn."""
        turn_df = self.results.get('turn_df')
        if turn_df is None or 'integrated_intensity' not in turn_df.columns:
            self.results['peaks_df'] = pd.DataFrame() # Ensure key exists
            return

        intensity_data = turn_df['integrated_intensity']
        
        # Set a prominence threshold to avoid finding noisy peaks
        # A peak must stand out from its surroundings by at least the median intensity value
        prominence_threshold = intensity_data.median()
        if pd.isna(prominence_threshold): prominence_threshold = 0

        # Find peaks
        peaks, properties = find_peaks(
            intensity_data.values, 
            prominence=prominence_threshold,
            width=1 # Calculate width
        )

        if peaks.size > 0:
            # Create a DataFrame with peak properties
            peaks_df = pd.DataFrame({
                'turn': peaks,
                'intensity': intensity_data.values[peaks], 
                'width_turns': properties['widths'],
                'width_height': properties['width_heights'],
                'left_turn': properties['left_ips'],
                'right_turn': properties['right_ips']
            })
            # Sort by intensity to find the biggest peaks
            peaks_df = peaks_df.sort_values(by='intensity', ascending=False).reset_index(drop=True)
            self.results['peaks_df'] = peaks_df
            print(f"  Found {len(peaks_df)} significant peaks.")
        else:
            self.results['peaks_df'] = pd.DataFrame() # No peaks found


    """    
    def _calculate_aggregate_stats(self):
        num_batches = self.config['NUM_BATCHES_PER_TURN']
        buckets_per_batch = self.config['BUCKETS_PER_BATCH']
        start_bucket = self.config.get("start_bucket", 87)
        
        # Group all data by batch number across all turns
        all_batch_data = {f"Batch {i+1}": [] for i in range(num_batches)}
        
        current_bucket = start_bucket
        expected_turn_len = buckets_per_batch * num_batches
        while current_bucket + expected_turn_len <= len(self.intensity_series):
            for i in range(num_batches):
                batch_start = current_bucket + i * buckets_per_batch
                batch_end = batch_start + buckets_per_batch
                batch_data = self.intensity_series.iloc[batch_start:batch_end].values
                all_batch_data[f"Batch {i+1}"].extend(batch_data)
            current_bucket += expected_turn_len
            
        self.results['aggregate_stats'] = {
            'all_batch_data': all_batch_data,
            'batch6_max_overall': np.max(all_batch_data['Batch 6']) if all_batch_data['Batch 6'] else np.nan
        }


    def _calculate_aggregate_stats(self):
        
        num_batches = self.config['NUM_BATCHES_PER_TURN']
        buckets_per_batch = self.config['BUCKETS_PER_BATCH']
        start_bucket = self.config.get("start_bucket", 87)
        
        # Group all data by batch number across all turns
        all_batch_data = {f"Batch {i+1}": [] for i in range(num_batches)}
        
        current_bucket = start_bucket
        expected_turn_len = buckets_per_batch * num_batches
        while current_bucket + expected_turn_len <= len(self.intensity_series):
            for i in range(num_batches):
                batch_start = current_bucket + i * buckets_per_batch
                batch_end = batch_start + buckets_per_batch
                batch_data = self.intensity_series.iloc[batch_start:batch_end].values
                all_batch_data[f"Batch {i+1}"].extend(batch_data)
            current_bucket += expected_turn_len

        # --- 🔽 MODIFIED/NEW SECTION 🔽 ---

        # Find max intensity for each batch across the entire spill
        batch_maxes = [np.max(all_batch_data[f'Batch {i+1}']) if all_batch_data[f'Batch {i+1}'] else np.nan for i in range(num_batches)]

        # Find info for the single most intense bucket in the spill
        spill_max_bucket_info = {}
        if not self.results['turn_df'].empty and self.results['turn_df']['max'].notna().any():
            turn_with_max_idx = self.results['turn_df']['max'].idxmax()
            max_intensity_val = self.results['turn_df']['max'].loc[turn_with_max_idx]
            
            turn_len = buckets_per_batch * num_batches
            start = start_bucket + turn_with_max_idx * turn_len
            turn_series = self.intensity_series.iloc[start : start + turn_len].copy()
            turn_series.index = np.arange(turn_len)
            
            bucket_in_turn = turn_series.idxmax()
            #Bug: batch_num = (bucket_in_turn // buckets_per_batch) + 1
            # Calculate the 0-indexed batch number and clip it to the valid range [0, 6]
            batch_index = np.clip(bucket_in_turn // buckets_per_batch, 0, num_batches - 1)
            batch_num = batch_index + 1
            spill_max_bucket_info = {
                "intensity": max_intensity_val, "batch_num": batch_num,
                "turn_num": turn_with_max_idx, "run_num": self.run_num, "spill_num": self.spill_num,
            }

        # Find info for the batch with the highest integrated intensity in the spill
        spill_max_batch_info = {}
        integral_cols = [f'integrated_intensity_batch_{i+1}' for i in range(num_batches)]
        if not self.results['turn_df'].empty and all(c in self.results['turn_df'].columns for c in integral_cols):
            integrals_df = self.results['turn_df'][integral_cols]
            if not integrals_df.stack().empty:
                max_integral = integrals_df.max().max()
                turn_num, col_name = integrals_df.stack().idxmax()
                batch_num = int(col_name.split('_')[-1])
                spill_max_batch_info = {
                    "integral": max_integral, "batch_num": batch_num,
                    "turn_num": turn_num, "run_num": self.run_num, "spill_num": self.spill_num,
                }
        # Find info for the turn with the highest integrated intensity in the spill
        spill_max_turn_info = {}
        if not self.results['turn_df'].empty and 'integrated_intensity' in self.results['turn_df'].columns:
            turn_integrals = self.results['turn_df']['integrated_intensity']
            if turn_integrals.notna().any():
                turn_num = turn_integrals.idxmax()
                max_integral_val = turn_integrals.loc[turn_num]
                spill_max_turn_info = {
                    "integral": max_integral_val,
                    "turn_num": turn_num,
                    "run_num": self.run_num,
                    "spill_num": self.spill_num,
                }        
        
        self.results['aggregate_stats'] = {
            'all_batch_data': all_batch_data,
            'batch_maxes_overall': batch_maxes,
            'spill_max_bucket': spill_max_bucket_info,
            'spill_max_batch_integral': spill_max_batch_info,
            'spill_max_turn_integral': spill_max_turn_info  # <-- Combined here
        }
    """

    def _calculate_aggregate_stats(self):
        """Calculates statistics that summarize the entire spill."""
        num_batches = self.config['NUM_BATCHES_PER_TURN']
        buckets_per_batch = self.config['BUCKETS_PER_BATCH']
        start_bucket = self.config.get("start_bucket", 87)
        
        # --- MODIFIED SECTION ---
        # The redundant loop has been removed. We now retrieve the batch data 
        # that was collected during the _process_turns() function.
        all_batch_data = self.results.get('all_batch_data')
        if not all_batch_data:
             # Handle case where no data was processed
             self.results['aggregate_stats'] = {}
             return
        # --- END OF MODIFICATION ---

        # Find max intensity for each batch across the entire spill
        batch_maxes = [np.max(all_batch_data[f'Batch {i+1}']) if all_batch_data[f'Batch {i+1}'] else np.nan for i in range(num_batches)]

        # Find info for the single most intense bucket in the spill
        spill_max_bucket_info = {}
        if not self.results['turn_df'].empty and self.results['turn_df']['max'].notna().any():
            turn_with_max_idx = self.results['turn_df']['max'].idxmax()
            max_intensity_val = self.results['turn_df']['max'].loc[turn_with_max_idx]
            
            turn_len = buckets_per_batch * num_batches
            start = start_bucket + turn_with_max_idx * turn_len
            turn_series = self.intensity_series.iloc[start : start + turn_len].copy()
            turn_series.index = np.arange(turn_len)
            
            bucket_in_turn = turn_series.idxmax()
            # Calculate the 0-indexed batch number and clip it to the valid range [0, 6]
            batch_index = np.clip(bucket_in_turn // buckets_per_batch, 0, num_batches - 1)
            batch_num = batch_index + 1
            spill_max_bucket_info = {
                "intensity": max_intensity_val, "batch_num": batch_num,
                "turn_num": turn_with_max_idx, "run_num": self.run_num, "spill_num": self.spill_num,
            }

        # Find info for the batch with the highest integrated intensity in the spill
        spill_max_batch_info = {}
        integral_cols = [f'integrated_intensity_batch_{i+1}' for i in range(num_batches)]
        if not self.results['turn_df'].empty and all(c in self.results['turn_df'].columns for c in integral_cols):
            integrals_df = self.results['turn_df'][integral_cols]
            if not integrals_df.stack().empty:
                max_integral = integrals_df.max().max()
                turn_num, col_name = integrals_df.stack().idxmax()
                batch_num = int(col_name.split('_')[-1])
                spill_max_batch_info = {
                    "integral": max_integral, "batch_num": batch_num,
                    "turn_num": turn_num, "run_num": self.run_num, "spill_num": self.spill_num,
                }
                
        # Find info for the turn with the highest integrated intensity in the spill
        spill_max_turn_info = {}
        if not self.results['turn_df'].empty and 'integrated_intensity' in self.results['turn_df'].columns:
            turn_integrals = self.results['turn_df']['integrated_intensity']
            if turn_integrals.notna().any():
                turn_num = turn_integrals.idxmax()
                max_integral_val = turn_integrals.loc[turn_num]
                spill_max_turn_info = {
                    "integral": max_integral_val,
                    "turn_num": turn_num,
                    "run_num": self.run_num,
                    "spill_num": self.spill_num,
                }

        self.results['aggregate_stats'] = {
            'all_batch_data': all_batch_data,
            'batch_maxes_overall': batch_maxes,
            'spill_max_bucket': spill_max_bucket_info,
            'spill_max_batch_integral': spill_max_batch_info,
            'spill_max_turn_integral': spill_max_turn_info
        }

    def _count_filled_batches(self):
        """
        Uses a statistical outlier detection method on the full spill data
        to provide a robust count of filled batches.
        """
        all_batch_data = self.results.get('aggregate_stats', {}).get('all_batch_data')
        if not all_batch_data:
            self.results['aggregate_stats']['robust_filled_batch_count'] = 0
            return

        # 1. Create a flat list of all non-zero intensities for global stats
        all_intensities = [
            item for sublist in all_batch_data.values() for item in sublist if item > 0
        ]
        if not all_intensities:
            self.results['aggregate_stats']['robust_filled_batch_count'] = 0
            return

        # 2. Calculate a global outlier threshold using the IQR method
        q3 = np.percentile(all_intensities, 75)
        iqr = q3 - np.percentile(all_intensities, 25)
        outlier_threshold = q3 + 1.5 * iqr

        # 3. Check each batch against the threshold
        num_filled = 0
        for i in range(self.config['NUM_BATCHES_PER_TURN']):
            batch_label = f"Batch {i+1}"
            batch_data = all_batch_data.get(batch_label, [])
            if batch_data:
                # A batch is "filled" if its 99th percentile is above the outlier threshold
                p99 = np.percentile(batch_data, 99)
                if p99 > outlier_threshold:
                    num_filled += 1
        
        # 4. Store the results
        self.results['aggregate_stats']['robust_filled_batch_count'] = num_filled
        self.results['aggregate_stats']['robust_threshold'] = outlier_threshold
        print(f"  Found {num_filled} filled batches (Threshold={outlier_threshold:.2f})")
    



    # --------------------------------------------------------------------------
    # Static Calculation Methods
    # --------------------------------------------------------------------------

    @staticmethod
    def _prepare_chunk(intensity_chunk):
        """Helper to ensure input is a clean numpy float64 array."""
        if isinstance(intensity_chunk, (pd.Series, pd.DataFrame)):
            intensity_chunk = intensity_chunk.values.flatten()
        if not isinstance(intensity_chunk, np.ndarray):
            intensity_chunk = np.array(intensity_chunk)
        if intensity_chunk.size == 0:
            return None
        try:
            return intensity_chunk.astype(np.float64)
        except ValueError:
            return None

    @classmethod
    def _calculate_integrated_intensity(cls, intensity_chunk):
        """Calculates the integrated (summed) intensity of a data chunk."""
        chunk = cls._prepare_chunk(intensity_chunk)
        if chunk is None:
            return np.nan
        return np.sum(chunk)


    @classmethod
    def _calculate_duty_factor(cls, intensity_chunk):
        """Calculates the non-conditional duty factor: (<I>)^2 / <I^2>."""
        chunk = cls._prepare_chunk(intensity_chunk)
        if chunk is None: return np.nan
        
        mean_intensity_sq = np.mean(np.square(chunk))
        if np.abs(mean_intensity_sq) < 1e-12: return np.nan
        
        mean_intensity = np.mean(chunk)
        return np.square(mean_intensity) / mean_intensity_sq

    @classmethod
    def _calculate_conditional_duty_factor(cls, intensity_chunk, threshold):
        """Calculates conditional duty factor for intensities > threshold."""
        chunk = cls._prepare_chunk(intensity_chunk)
        if chunk is None: return np.nan
        
        filtered_chunk = chunk[chunk > threshold]
        if filtered_chunk.size == 0: return np.nan
        
        mean_intensity_sq_cond = np.mean(np.square(filtered_chunk))
        if np.abs(mean_intensity_sq_cond) < 1e-12: return np.nan
            
        mean_intensity_cond = np.mean(filtered_chunk)
        return np.square(mean_intensity_cond) / mean_intensity_sq_cond

    @classmethod
    def _calculate_figure_of_merit(cls, intensity_chunk, norm_factor=500):
        """Calculates a figure of merit: (<I>)^3 / <I^2> / norm."""
        chunk = cls._prepare_chunk(intensity_chunk)
        if chunk is None: return np.nan

        mean_intensity_sq = np.mean(np.square(chunk))
        if np.abs(mean_intensity_sq) < 1e-12: return np.nan

        mean_intensity = np.mean(chunk)
        return np.power(mean_intensity, 3) / mean_intensity_sq / norm_factor