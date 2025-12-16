# plotting.py
"""
spill_timing_analyzer plotting 
  
Description: Contains the SpillPlotter class for creating all visualizations
  related to the spill timing analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as mticker 
from matplotlib.ticker import MaxNLocator
from scipy.signal import find_peaks
from matplotlib.colors import LogNorm

ACNET_METADATA = {
    'G:TURN13': {'title': 'Number of Injected Turns vs. Spill', 'unit': 'Turns'},
    'G:BNCH13': {'title': 'Number of Filled RF Buckets per Cycle vs. Spill', 'unit': 'Buckets (bnch)'},
    'G:NBSYD':  {'title': 'Number of Filled Booster Batches vs. Spill', 'unit': 'Batches'},
    'I:FTSDF':  {'title': '53kHz Duty Factor vs. Spill', 'unit': 'Duty Factor (%)'},
    'S:F1SEM':  {'title': 'Protons at SEM F1 vs. Spill', 'unit': 'Protons (Ptns)'},
    'S:G2SEM':  {'title': 'Protons at SEM G2 vs. Spill', 'unit': 'Protons per Pulse (ppp)'},
    'F:NM2ION': {'title': 'Protons at Ion Chamber NM2 vs. Spill', 'unit': 'Protons per Pulse (ppp)'},
    # typuically 0 'F:NM3ION': {'title': 'Protons at Ion Chamber NM3 vs. Spill', 'unit': 'Protons per Pulse (ppp)'},
    'F:NM3SEM': {'title': 'Protons at SEM NM3 vs. Spill', 'unit': 'Protons per Pulse (ppp)'},
    'E:M3TGHM': {'title': 'Mean of Horizontal Beam Profile at NM3 vs. Spill', 'unit': 'Position (mm)'},
    'E:M3TGHS': {'title': 'Sigma of Horizontal Beam Profile at NM3 vs. Spill', 'unit': 'Width (mm)'},
    'E:M3TGVM': {'title': 'Mean of Vertical Beam Profile at NM3 vs. Spill', 'unit': 'Position (mm)'},
    'E:M3TGVS': {'title': 'Sigma of Vertical Beam Profile at NM3 vs. Spill', 'unit': 'Width (mm)'},
}


class SpillPlotter:
    """
    Handles plotting for the spill analysis.
    """
    def __init__(self, config):
        """
        Initializes the plotter with the configuration and proper plots directory
        """
        self.config = config
        self.plots_dir = config.PLOTS_DIR
    """    
    def _add_run_spill_text(self, fig, run_num, spill_num, total_intensity=None):
        text = f"Run: {run_num}\nSpill: {spill_num}"
        if total_intensity is not None:
            text += f"\nTotal Intensity: {total_intensity:,.0f}"
        
        fig.text(0.99, 0.99, text, transform=fig.transFigure,
                 horizontalalignment='right', verticalalignment='top',
                 fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    """
    
    def _add_stats_box(self, fig, analyzer):
        """Adds a stats box with key spill metrics to a figure."""
        # Calculate median on the fly from the analyzer's data
        median_intensity = analyzer.data['intensity'].median()

        # The number of spikes is the number of entries in the peaks_df
        num_spikes = len(analyzer.peaks_df)

        total_intensity = analyzer.total_intensity

        # --- Find Highest Spike Properties ---
        # Find the single most intense spike from the peaks_df for this spill
        if not analyzer.peaks_df.empty:
            highest_spike = analyzer.peaks_df.loc[analyzer.peaks_df['intensity'].idxmax()]
            highest_spike_intensity = highest_spike['intensity']
            highest_spike_integral = highest_spike['integral']
        else:
            highest_spike_intensity = 0
            highest_spike_integral = 0

        # Build the text string for the stats box
        text = (
            f"Total Intensity: {total_intensity:,.0f}\n"
            f"Mean Intensity: {analyzer.mean_intensity:,.0f}\n"
            f"Median Intensity: {median_intensity:,.0f}\n"
            f"Std Dev: {analyzer.std_intensity:,.0f}\n"
            f"Duty Factor: {analyzer.duty_factor:.3f}\n"
            f"Peak Intensity: {highest_spike_intensity:,.0f}\n"
            f"Peak Spike Integral: {highest_spike_integral:,.0f}\n"
            f"Num. Spikes Found: {num_spikes}\n"
            f"Spike Threshold: {analyzer.peak_threshold:,.0f}"
        )

        # Place the text box on the figure
        fig.text(0.9, 0.97, text, transform=fig.transFigure,
                 horizontalalignment='right', verticalalignment='top',
                 fontsize=10, bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

    def _calculate_autocorrelation(self, intensity_values):
        """Calculates and returns the autocorrelation of a signal."""
        autocorr = np.correlate(intensity_values - intensity_values.mean(), intensity_values - intensity_values.mean(), mode='full')
        return autocorr[autocorr.size // 2:] # Return the second half (positive time lag) and not the first half (negative time lag)

    def _plot_intensity_profile(self, ax, analyzer, is_representative_spill, is_zoomed, highlight_peaks, xlow=None, xhigh=None):
        """Helper function to perform spike analysis and plot the intensity profile."""
        intensity, times = analyzer.data['intensity'], analyzer.data['time_s']
        if not analyzer.peaks_df.empty:
            peak_indices = analyzer.peaks_df['index'].to_numpy()
        else:
            peak_indices = np.array([]) # Use an empty array if no peaks were found


        if is_representative_spill:
            print(f"  - Diagnostic for Spill {analyzer.spill_num}: Found {len(peak_indices)} significant peaks.")

        if highlight_peaks:

            ax.plot(times, intensity, marker='.', markersize=1, linestyle='', color='black', label='All Data')

            freq_to_highlight = 73.75
            period = 1.0 / freq_to_highlight
            tolerance = 0.05
            
            highlighted_indices = set()
            connected_pairs = set()

            if len(peak_indices) > 1:
                peak_times = times.iloc[peak_indices].values
                for i in range(len(peak_indices)):
                    current_peak_time = peak_times[i]
                    expected_next_time = current_peak_time + period
                    partners = np.where(np.abs(peak_times - expected_next_time) < period * tolerance)[0]
                    
                    if len(partners) > 0:
                        for partner_idx in partners:
                            p1_idx, p2_idx = peak_indices[i], peak_indices[partner_idx]
                            highlighted_indices.add(p1_idx)
                            highlighted_indices.add(p2_idx)
                            connected_pairs.add(tuple(sorted((p1_idx, p2_idx))))

            if is_representative_spill:
                print(f"  - Diagnostic for Spill {analyzer.spill_num}: Found {len(peak_indices)} significant peaks.")
                print(f"    - Found {len(connected_pairs)} pairs matching ~{freq_to_highlight:.1f} Hz pattern.")

            
            if connected_pairs:
                for idx1, idx2 in connected_pairs:
                    t1, i1 = times.iloc[idx1], intensity.iloc[idx1]
                    t2, i2 = times.iloc[idx2], intensity.iloc[idx2]

                    # Draw the connecting line
                    ax.plot([t1, t2], [i1, i2], color='orange', linestyle='-', linewidth=0.8, alpha=0.6)

                    # Calculate time difference and frequency
                    time_diff = abs(t2 - t1)
                    if time_diff > 1e-9: # Avoid division by zero
                        frequency = 1.0 / time_diff

                        # Create the text label
                        label_text = f"{time_diff * 1000:.1f} ms\n({frequency:.1f} Hz)"

                        # Calculate the midpoint to place the text
                        x_mid = (t1 + t2) / 2
                        y_mid = (i1 + i2) / 2 

                        # Add the text to the plot slightly above the line's midpoint
                        ax.text(x_mid, y_mid, label_text, fontsize=7, color='orange', ha='center', va='bottom',
                                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

            if highlight_peaks and highlighted_indices:
                ax.plot(times.iloc[list(highlighted_indices)], intensity.iloc[list(highlighted_indices)], 
                        'o', ms=6, mec='orangered', mfc='none', label=f'~{freq_to_highlight:.1f} Hz Spikes')
                ax.legend()
        else:
            ax.plot(times, intensity, linestyle='-', linewidth=0.5, color='black', label='All Data')


        if is_zoomed and xlow is not None and xhigh is not None:
            title_suffix = f" (Zoomed {xlow}-{xhigh}s)"
            ax.set_xlim(xlow, xhigh)
        else:
            title_suffix = ""
            ax.set_xlim(left=0)

        ax.set_title(f"Full Spill Linearized Intensity ({analyzer.frequency_label}) for Run {analyzer.run_num}, Spill {analyzer.spill_num}{title_suffix}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Linearized Intensity")
        ax.grid(True, linestyle='--')


    def plot_overall_integrated_spill(self, summed_data):
        """Creates a bar chart of the total integrated intensity for all spills."""
        fig, ax = plt.subplots(figsize=(12, 7))

        # 1. Calculate the width of each time bin
        # This assumes all bins are the same width
        bin_width = summed_data.index[1] - summed_data.index[0]

        # 2. Calculate the left edge of each bar for proper alignment
        left_edges = summed_data.index - (bin_width / 2.0)
        
        # 3. Plot using the left edges, full width, and 'edge' alignment
        ax.bar(left_edges, summed_data.values, 
               width=bin_width, 
               align='edge', 
               edgecolor='black')
        
        ax.set_title('Total Integrated Linearized Intensity vs. Time (All Spills)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Total Integrated Linearized Intensity')
        ax.grid(True, axis='y', linestyle='--')
        # Ensure the x-axis limits are tight to the bars
        ax.set_xlim(left_edges[0], left_edges[-1] + bin_width)
        
        output_filename = f"{self.config.PLOTS_DIR}/Overall_Integrated_Spill.png"
        plt.savefig(output_filename, dpi=150)
        plt.close(fig)
        #print(f"\nOverall integrated spill plot saved to: {output_filename}")

    def plot_single_spill(self, analyzer, is_representative_spill=False):
        """Intensity vs time for a single spill"""    
        if analyzer.data.empty: return
        fig, ax = plt.subplots(figsize=(15, 7))
        self._plot_intensity_profile(ax, analyzer, is_representative_spill, is_zoomed=False, highlight_peaks=False)
        #self._add_run_spill_text(fig, analyzer.run_num, analyzer.spill_num, analyzer.total_intensity)
        self._add_stats_box(fig, analyzer)
        output_filename = f"{self.plots_dir}/Intensity_vs_Time_run{analyzer.run_num}_spill{analyzer.spill_num}.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight'), plt.close(fig)
        #print(f"Intensity plot for most intense spill of run {analyzer.run_num} saved.")

    def plot_single_spill_zoomed(self, analyzer, xlow, xhigh):
        """Intensity vs time for a single spill but zoomed in to a abitrary 0.4 second window from 1-1.4s"""    
        if analyzer.data.empty: return
        fig, ax = plt.subplots(figsize=(15, 7))

        self._plot_intensity_profile(ax, analyzer, is_representative_spill=False, is_zoomed=True, highlight_peaks=True, xlow=xlow, xhigh=xhigh)

        #self._add_stats_box(fig, analyzer)
        #plt.tight_layout()
        output_filename = f"{self.plots_dir}/Intensity_vs_Time_Zoomed_run{analyzer.run_num}_spill{analyzer.spill_num}_x_{str(xlow).replace('.', 'p')}_{str(xhigh).replace('.', 'p')}.png"
        plt.savefig(output_filename, dpi=300)#, bbox_inches='tight'), 
        plt.close(fig)
        #print(f"Zoomed intensity plot for most intense spill of run {analyzer.run_num} saved.")

    def plot_spill_count_per_run(self, all_run_data):
        """Simple plot of number of spills recorded in each data taking run"""    
        if all_run_data.empty: return
        spill_counts = all_run_data['run'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(12, 7))
        bars = ax.bar(spill_counts.index, spill_counts.values, color='cornflowerblue', edgecolor='black')
        ax.set_title("Number of Analyzed Spills per Run")
        ax.set_xlabel("Run Number"), ax.set_ylabel("Number of Spills")
        ax.grid(True, axis='y', linestyle='--')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xticks(rotation=45), ax.bar_label(bars)
        output_filename = f"{self.plots_dir}/Spills_Per_Run_Summary.png"
        plt.tight_layout(), plt.savefig(output_filename, dpi=150), plt.close(fig)
        #print(f"\nSpill count per run plot saved to: {output_filename}")

    def plot_max_intensity_vs_spill(self, run_number, run_data):
        """maximum intensity bin in each spill vs spill number in a given run"""    
        if run_data.empty or run_data['max_intensity'].isnull().all(): return
        df = run_data.sort_values(by='spill').dropna(subset=['max_intensity'])
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(df['spill'], df['max_intensity'], marker='o', linestyle='--')
        ax.set_title(f"Peak Linearized Intensity vs. Spill Number for Run {run_number}")
        ax.set_xlabel("Spill Number"), ax.set_ylabel("Peak 100µs Integrated Linearized Intensity")
        ax.grid(True, linestyle='--')
        ax.get_xaxis().get_major_formatter().set_useOffset(False), ax.get_xaxis().get_major_formatter().set_scientific(False)
        output_filename = f"{self.plots_dir}/Max_Intensity_vs_Spill_run_{run_number}_Summary.png"
        plt.savefig(output_filename, dpi=150), plt.close(fig)
        #print(f"\nPeak Intensity plot for Run {run_number} saved.")

    def plot_total_intensity_vs_spill(self, run_number, run_data):
        """total intensity in each spill vs spill number in a given run"""    
        if run_data.empty or run_data['total_intensity'].isnull().all(): return
        df = run_data.sort_values(by='spill').dropna(subset=['total_intensity'])
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(df['spill'], df['total_intensity'], marker='o', linestyle='--')
        ax.set_title(f"Total Integrated Linearized Intensity vs. Spill Number for Run {run_number}")
        ax.set_xlabel("Spill Number"), ax.set_ylabel("Total Integrated Linearized Intensity per Spill")
        ax.grid(True, linestyle='--')
        ax.get_xaxis().get_major_formatter().set_useOffset(False), ax.get_xaxis().get_major_formatter().set_scientific(False)
        output_filename = f"{self.plots_dir}/Total_Intensity_vs_Spill_run_{run_number}_Summary.png"
        plt.savefig(output_filename, dpi=150), plt.close(fig)
        #print(f"Total Intensity plot for Run {run_number} saved.")
        
    def plot_duty_factor_vs_spill(self, run_number, run_data):
        """Duty factor for each spill vs spill number in a given run"""    
        if run_data.empty or run_data['duty_factor'].isnull().all(): return
        df = run_data.sort_values(by='spill').dropna(subset=['duty_factor'])
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(df['spill'], df['duty_factor'], marker='o', linestyle='--')
        ax.set_title(f"Duty Factor vs. Spill Number for Run {run_number}")
        ax.set_xlabel("Spill Number"), ax.set_ylabel("Spill Duty Factor")
        ax.grid(True, linestyle='--'), ax.set_ylim(0, 1.05)
        ax.get_xaxis().get_major_formatter().set_useOffset(False), ax.get_xaxis().get_major_formatter().set_scientific(False)
        output_filename = f"{self.plots_dir}/Duty_Factor_vs_Spill_run_{run_number}_Summary.png"
        plt.savefig(output_filename, dpi=150), plt.close(fig)
        #print(f"Duty Factor plot for Run {run_number} saved.")

    def plot_uniformity_vs_spill(self, run_number, run_data):
        """Plot three uniformity metrics (CV, Kurtosis, Gini) for each spill vs spill number in a given run"""    
        if run_data.empty: return
        df = run_data.sort_values(by='spill').dropna()
        if df.empty: return
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'Spill Uniformity Metrics for Run {run_number}', fontsize=16)
        axes[0].plot(df['spill'], df['coefficient_of_variation'], 'o-', ms=4)
        axes[0].set_ylabel('Coefficient of Variation'), axes[0].grid(True)
        axes[1].plot(df['spill'], df['kurtosis'], 'o-', ms=4, color='green')
        axes[1].set_ylabel('Kurtosis'), axes[1].grid(True)
        axes[2].plot(df['spill'], df['gini'], 'o-', ms=4, color='red')
        axes[2].set_ylabel('Gini Coefficient'), axes[2].set_xlabel('Spill Number'), axes[2].grid(True)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_filename = f"{self.plots_dir}/Uniformity_Metrics_run_{run_number}.png"
        plt.savefig(output_filename, dpi=150), plt.close(fig)
        #print(f"Uniformity metrics plot for Run {run_number} saved.")
    """    
    def plot_peak_interval_histogram(self, run_number, run_data, weighted=False):
        
        if run_data.empty: return
        intervals, weights = [], []
        for _, row in run_data.iterrows():
            if len(row['peak_intervals']) > 0:
                intervals.extend(row['peak_intervals'])
                weights.extend([row['total_intensity']] * len(row['peak_intervals']))
        if not intervals: return
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.hist(intervals, bins=100, range=(0, max(0.1, np.percentile(intervals, 99))), weights=weights if weighted else None)
        title = 'Intensity-Weighted ' if weighted else ''
        title += f'Distribution of Time Between Spikes for Run {run_number}'
        ax.set_title(title)
        ax.set_xlabel('Time Between Consecutive Spikes (s)'), ax.set_ylabel('Summed Spill Intensity' if weighted else 'Count')
        ax.grid(True, axis='y')
        filename_suffix = "_weighted" if weighted else ""
        output_filename = f"{self.plots_dir}/Peak_Interval_Hist_run_{run_number}{filename_suffix}.png"
        plt.savefig(output_filename, dpi=150), plt.close(fig)
        #print(f"Peak interval histogram (weighted={weighted}) for Run {run_number} saved.")
    
    def plot_fft(self, run_number, spill_analyzer):
        if spill_analyzer.data.empty: return
        intensity = spill_analyzer.data['intensity'].values
        n_points, time_step = len(intensity), spill_analyzer.data['time_s'].iloc[1] - spill_analyzer.data['time_s'].iloc[0]
        fft_vals, fft_freq = np.fft.fft(intensity), np.fft.fftfreq(n_points, d=time_step)
        positive_freq_mask = fft_freq > 0
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.suptitle(f'Frequency Analysis for Spill {spill_analyzer.spill_num} (Run {run_number})', fontsize=16)
        fft_magnitudes, fft_frequencies = np.abs(fft_vals)[positive_freq_mask], fft_freq[positive_freq_mask]
        ax.plot(fft_frequencies, fft_magnitudes)
        ax.set_title('Fast Fourier Transform (FFT)'), ax.set_xlabel('Frequency (Hz)'), ax.set_ylabel('Magnitude')
        ax.set_xlim(0, 500) 
        ax.grid(True)
        peaks, properties = find_peaks(fft_magnitudes, height=np.mean(fft_magnitudes), distance=50)
        if len(peaks) > 0:
            top_indices = np.argsort(properties['peak_heights'])[-10:]
            for peak_index in peaks[top_indices]:
                peak_freq, peak_mag = fft_frequencies[peak_index], fft_magnitudes[peak_index]
                ax.axvline(x=peak_freq, color='r', linestyle='--', alpha=0.7)
                ax.text(peak_freq + 7, peak_mag, f'{peak_freq:.2f} Hz', color='r')
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        output_filename = f"{self.plots_dir}/FFT_run_{run_number}_spill_{spill_analyzer.spill_num}.png"
        plt.savefig(output_filename, dpi=150), plt.close(fig)
        #print(f"FFT analysis plot for spill {spill_analyzer.spill_num} saved.")
    """

    def plot_fft(self, run_number, spill_analyzer):
        """Fast Fourier Transform (FFT) for a given spill with labels for the tallest peaks"""
        if spill_analyzer.data.empty: return
        intensity = spill_analyzer.data['intensity'].values
        n_points, time_step = len(intensity), spill_analyzer.data['time_s'].iloc[1] - spill_analyzer.data['time_s'].iloc[0]
        fft_vals, fft_freq = np.fft.fft(intensity), np.fft.fftfreq(n_points, d=time_step)
        positive_freq_mask = fft_freq > 0
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.suptitle(f'Frequency Analysis for Spill {spill_analyzer.spill_num} (Run {run_number})', fontsize=16)
        fft_magnitudes, fft_frequencies = np.abs(fft_vals)[positive_freq_mask], fft_freq[positive_freq_mask]
        ax.plot(fft_frequencies, fft_magnitudes, zorder=1) # Plot the main FFT line
        ax.set_title('Fast Fourier Transform (FFT)'), ax.set_xlabel('Frequency (Hz)'), ax.set_ylabel('Magnitude')
        ax.set_xlim(0, 500)
        ax.grid(True)
        peaks, properties = find_peaks(fft_magnitudes, height=np.mean(fft_magnitudes), distance=50, prominence=1)

        if len(peaks) > 0:
            number_of_peaks_to_highlight = 15
            top_peak_indices = []

            # Check if the 'prominences' key exists and has data
            if 'prominences' in properties and len(properties['prominences']) > 0:
                # If so, sort by prominence (the preferred method)
                # Pair each peak's location with its prominence value. Creates a list of tuples,  [(prominence, location), ...]
                peaks_with_prominence = list(zip(properties['prominences'], peaks))
                # Sort this list. By default, it sorts by the first item in each tuple (the prominence).
                sorted_peaks = sorted(peaks_with_prominence, reverse=True)
                # Get the last N items, which are the top N most prominent peaks.
                top_n_peaks = sorted_peaks[:number_of_peaks_to_highlight]
                # Extract the peak locations from the tuples (index 1 in tuple)
                top_peak_indices = [peak[1] for peak in top_n_peaks]
            else:
                # Otherwise, fall back to sorting by height
                print(f"  - WARNING: No peaks met prominence criteria for spill {spill_analyzer.spill_num}. Falling back to sorting by height.")
                top_indices = np.argsort(properties['peak_heights'])[-number_of_peaks_to_highlight:]
                top_peak_indices = peaks[top_indices]

            # Get the coordinates of the top peaks
            top_freqs = fft_frequencies[top_peak_indices]
            top_mags = fft_magnitudes[top_peak_indices]

            # --- NEW: Draw circles and labels instead of lines ---
            # Plot all circles in a single call for a clean legend
            ax.scatter(top_freqs, top_mags, s=150, facecolors='none', edgecolors='red', linewidth=1.5, zorder=2, label=f'Top {number_of_peaks_to_highlight} peaks by prominence')

            # Loop through just to add the text labels
            for i in range(len(top_freqs)):
                ax.text(top_freqs[i] + 10, top_mags[i], f'{top_freqs[i]:.2f} Hz', color='red', verticalalignment='center')

            ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        output_filename = f"{self.plots_dir}/FFT_run_{run_number}_spill_{spill_analyzer.spill_num}.png"
        plt.savefig(output_filename, dpi=150), plt.close(fig)
        #print(f"FFT analysis plot for spill {spill_analyzer.spill_num} saved.")


    def plot_autocorrelation(self, run_number, spill_analyzer):
        """Autocorrelation for a given spill"""
        if spill_analyzer.data.empty: return
        intensity = spill_analyzer.data['intensity'].values

        autocorr = self._calculate_autocorrelation(intensity)

        time_step = spill_analyzer.data['time_s'].iloc[1] - spill_analyzer.data['time_s'].iloc[0]
        time_lags = np.arange(len(autocorr)) * time_step

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(time_lags, autocorr)
        ax.set_title(f'Autocorrelation for Spill {spill_analyzer.spill_num} (Run {run_number})')
        ax.set_xlabel('Time Lag (s)'), ax.set_ylabel('Correlation')
        ax.set_xlim(0, 0.2), ax.grid(True)
        plt.tight_layout()
        output_filename = f"{self.plots_dir}/Autocorrelation_run_{run_number}_spill_{spill_analyzer.spill_num}.png"
        plt.savefig(output_filename, dpi=150), plt.close(fig)
        #print(f"Autocorrelation plot for spill {spill_analyzer.spill_num} saved.")

    def plot_fft_of_autocorrelation(self, run_number, spill_analyzer):
        """Calculates the autocorrelation and then plots its FFT (the Power Spectrum)."""
        if spill_analyzer.data.empty: return

        autocorr = self._calculate_autocorrelation(spill_analyzer.data['intensity'].values)
        n_points = len(autocorr)
        time_step = spill_analyzer.data['time_s'].iloc[1] - spill_analyzer.data['time_s'].iloc[0]
        fft_vals = np.fft.fft(autocorr)
        fft_freq = np.fft.fftfreq(n_points, d=time_step)

        positive_freq_mask = fft_freq > 0
        fig, ax = plt.subplots(figsize=(12, 7))

        power_spectrum = np.abs(fft_vals)[positive_freq_mask]
        fft_frequencies = fft_freq[positive_freq_mask]

        ax.plot(fft_frequencies, power_spectrum, zorder=1)
        ax.set_title(f'Power Spectral Density for Spill {spill_analyzer.spill_num} (Run {run_number})')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.set_xlim(0, 500)
        ax.set_yscale('log')
        ax.grid(True)

        # --- NEW: Add Peak Finding and Labeling ---
        peaks, properties = find_peaks(power_spectrum, height=np.mean(power_spectrum), distance=50, prominence=1)

        if len(peaks) > 0:
            number_of_peaks_to_highlight = 10
            top_peak_indices = []

            if 'prominences' in properties and len(properties['prominences']) > 0:
                peaks_with_prominence = list(zip(properties['prominences'], peaks))
                sorted_peaks = sorted(peaks_with_prominence, reverse=True)
                top_n_peaks = sorted_peaks[:number_of_peaks_to_highlight]
                top_peak_indices = [peak[1] for peak in top_n_peaks]
            else:
                top_indices = np.argsort(properties['peak_heights'])[-number_of_peaks_to_highlight:]
                top_peak_indices = peaks[top_indices]

            top_freqs = fft_frequencies[top_peak_indices]
            top_mags = power_spectrum[top_peak_indices]

            ax.scatter(top_freqs, top_mags, s=200, facecolors='none', edgecolors='red', linewidth=1.5, zorder=2, label='Top Peaks')
            for i in range(len(top_freqs)):
                ax.text(top_freqs[i] + 10, top_mags[i], f'{top_freqs[i]:.2f} Hz', color='red', verticalalignment='center')

            ax.legend()
        # --- End of New Section ---

        plt.tight_layout()
        output_filename = f"{self.plots_dir}/Power_Spectrum_run_{run_number}_spill_{spill_analyzer.spill_num}.png"
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        #print(f"Power spectrum plot for spill {spill_analyzer.spill_num} saved.")

    def plot_time_of_max_vs_spill(self, run_number, run_data):
        """Scatter plot. Time of peak intensity bin vs spill number"""    
        if run_data.empty or run_data['time_of_max'].isnull().all(): return
        df = run_data.sort_values(by='spill').dropna(subset=['time_of_max'])
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(df['spill'], df['time_of_max'], marker='o', linestyle='', ms=4)
        ax.set_title(f"Time of Peak Intensity vs. Spill Number for Run {run_number}")
        ax.set_xlabel("Spill Number"), ax.set_ylabel("Time of Maximum Intensity (s)")
        ax.grid(True, linestyle='--'), ax.get_xaxis().get_major_formatter().set_useOffset(False)
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        output_filename = f"{self.plots_dir}/Time_of_Max_vs_Spill_run_{run_number}_Summary.png"
        plt.savefig(output_filename, dpi=150), plt.close(fig)
        #print(f"Time of Max plot for Run {run_number} saved.")

    def plot_time_of_max_histogram(self, run_number, run_data, weighted=False):
        """Histogram: For all spills in a run plot the time of the max intensity"""    
        if run_data.empty: return
        df = run_data.dropna(subset=['time_of_max', 'total_intensity'])
        if df.empty: return
        fig, ax = plt.subplots(figsize=(12, 7))
        weights = df['total_intensity'] if weighted else None
        ax.hist(df['time_of_max'], bins=16, range=(0, 4), weights=weights)
        title = 'Intensity-Weighted ' if weighted else ''
        title += f'Distribution of Peak Times for Run {run_number}'
        ax.set_title(title), ax.set_xlabel("Time of Maximum Intensity (s)")
        ax.set_ylabel("Summed Spill Intensity" if weighted else "Number of Spills")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True)), ax.grid(True, linestyle='--', axis='y')
        filename_suffix = "_weighted" if weighted else ""
        output_filename = f"{self.plots_dir}/Time_of_Max_Hist_run_{run_number}{filename_suffix}.png"
        plt.savefig(output_filename, dpi=150), plt.close(fig)
        #print(f"Time of Max histogram (weighted={weighted}) for Run {run_number} saved.")

    def plot_total_intensity_overall(self, all_run_data):
        """SUMMARY PLOT: Plot the total intensity recorded per spill vs spill number (chronological, not sequential) for all runs"""    
        if all_run_data.empty or all_run_data['total_intensity'].isnull().all(): return
        df_sorted = all_run_data.sort_values(by='spill').dropna(subset=['total_intensity']).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(15, 10))
        unique_runs = df_sorted['run'].unique()
        colormap = plt.get_cmap('hsv', len(unique_runs))
        for i, run_number in enumerate(unique_runs):
            run_data = df_sorted[df_sorted['run'] == run_number]
            ax.plot(run_data.index, run_data['total_intensity'], 'o', ms=4, color=colormap(i), label=f'Run {run_number}')
        ax.set_title("Total Integrated Linearized Intensity vs. Spill Number")
        ax.set_xlabel("Spill Number (chronological, not sequential)")
        ax.set_ylabel("Total Integrated Linearized Intensity per Spill"), ax.grid(True, linestyle='--'), ax.set_xlim(left=-1, right=len(df_sorted))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=10, title="Run Number", fontsize='small')
        tick_spacing = 40
        tick_locs = range(0, len(df_sorted), tick_spacing)
        tick_labels = df_sorted['spill'].iloc[tick_locs]
        ax.set_xticks(tick_locs), ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.tick_params(axis='x', which='major', labelsize=8)
        plt.subplots_adjust(bottom=0.3)
        output_filename = f"{self.plots_dir}/Overall_Total_Intensity_vs_Spill_Summary.png"
        plt.savefig(output_filename, dpi=200), plt.close(fig)
        #print(f"\nOverall Total Intensity plot saved to: {output_filename}")
    """    
    def plot_total_intensity_half_spill_overall(self, all_run_data):
        if all_run_data.empty or all_run_data['total_intensity_0_2_seconds'].isnull().all(): return
        if all_run_data.empty or all_run_data['total_intensity_2_4_seconds'].isnull().all(): return
        df_sorted1 = all_run_data.sort_values(by='spill').dropna(subset=['total_intensity_0_2_seconds']).reset_index(drop=True)
        df_sorted2 = all_run_data.sort_values(by='spill').dropna(subset=['total_intensity_2_4_seconds']).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(15, 10))
        unique_runs1 = df_sorted1['run'].unique()
        unique_runs2 = df_sorted2['run'].unique()
        for i, run_number in enumerate(unique_runs1):
            run_data1 = df_sorted1[df_sorted1['run'] == run_number]
            run_data2 = df_sorted2[df_sorted2['run'] == run_number]
            ax.plot(run_data1.index, run_data1['total_intensity_0_2_seconds'], 'o', ms=4, color='tab:red')
            ax.plot(run_data2.index, run_data2['total_intensity_2_4_seconds'], 'o', ms=4, color='tab:blue')
        ax.set_title("Total Integrated Linearized Intensity vs. Spill Number")
        ax.set_xlabel("Spill Number (chronological, not sequential)")
        ax.set_ylabel("Total Integrated Linearized Intensity per Spill"), ax.grid(True, linestyle='--'), ax.set_xlim(left=-1, right=len(df_sorted1))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=10, title="Run Number", fontsize='small')
        tick_spacing = 40
        tick_locs = range(0, len(df_sorted1), tick_spacing)
        tick_labels = df_sorted1['spill'].iloc[tick_locs]
        ax.set_xticks(tick_locs), ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.tick_params(axis='x', which='major', labelsize=8)
        plt.subplots_adjust(bottom=0.3)
        output_filename = f"{self.plots_dir}/Overall_Total_Intensity_HalfSpills_vs_Spill_Summary.png"
        plt.savefig(output_filename, dpi=200), plt.close(fig)
        #print(f"\nOverall Total Intensity plot saved to: {output_filename}")
    """

    def plot_total_intensity_half_spill_overall(self, all_run_data):
        """Plots the integrated intensity of the first and second half of each spill."""
        if all_run_data.empty: return

        # 1. Prepare a single, clean DataFrame, sorted chronologically.
        df_sorted = all_run_data.sort_values(by='spill').dropna(
            subset=['total_intensity_0_2_seconds', 'total_intensity_2_4_seconds']
        ).reset_index(drop=True)

        if df_sorted.empty: return

        fig, ax = plt.subplots(figsize=(15, 8))

        # 2. Plot all points for each half in a single call, providing a label for the legend.
        ax.plot(df_sorted.index, df_sorted['total_intensity_0_2_seconds'], 'o', ms=4,
                color='red', label='First Half of Spill (0-2s)')
        
        ax.plot(df_sorted.index, df_sorted['total_intensity_2_4_seconds'], 'o', ms=4,
                color='blue', label='Second Half of Spill (2-4s)')

        # 3. Set clear titles and labels.
        ax.set_title("Intensity Comparison of Spill Halves")
        ax.set_xlabel("Spill Number (Chronological)")
        ax.set_ylabel("Total Integrated Linearized Intensity")
        ax.grid(True, linestyle='--')
        
        # 4. Add the legend to the plot in the best location.
        ax.legend(loc='best')

        # 5. Set the x-axis tick labels to show the actual spill number at reasonable intervals.
        tick_spacing = max(1, len(df_sorted) // 10) # Aim for ~10 ticks
        tick_locs = np.arange(0, len(df_sorted), tick_spacing)
        tick_labels = df_sorted['spill'].iloc[tick_locs]
        
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.tick_params(axis='x', which='major', labelsize=10)

        # 6. Final adjustments and save.
        plt.tight_layout()
        output_filename = f"{self.plots_dir}/Overall_Total_Intensity_HalfSpills_vs_Spill_Summary.png"
        plt.savefig(output_filename, dpi=200, bbox_inches='tight')
        plt.close(fig)
        #print(f"\nOverall intensity of spill halves plot saved to: {output_filename}")


    def plot_total_intensity_half_spill_overall(self, all_run_data):
        """SUMMARY PLOT: Plots the integrated intensity of the first and second half of each spill."""
        if all_run_data.empty: return

        # 1. Prepare a single, clean DataFrame, sorted chronologically.
        df_sorted = all_run_data.sort_values(by='spill').dropna(
            subset=['total_intensity_0_2_seconds', 'total_intensity_2_4_seconds']
        ).reset_index(drop=True)

        if df_sorted.empty: return

        fig, ax = plt.subplots(figsize=(15, 8))

        # 2. Plot all points for each half in a single call, providing a label for the legend.
        ax.plot(df_sorted.index, df_sorted['total_intensity_0_2_seconds'], 'o', ms=4,
                color='red', label='First Half of Spill (0-2s)')
        
        ax.plot(df_sorted.index, df_sorted['total_intensity_2_4_seconds'], 'o', ms=4,
                color='blue', label='Second Half of Spill (2-4s)')

        # 3. Set clear titles and labels.
        ax.set_title("Intensity Comparison of Spill Halves")
        ax.set_xlabel("Spill Number (Chronological)")
        ax.set_ylabel("Total Integrated Linearized Intensity")
        ax.grid(True, linestyle='--')
        
        # 4. Add the legend to the plot in the best location.
        ax.legend(loc='best')

        # 5. Set the x-axis tick labels to show the actual spill number at reasonable intervals.
        tick_spacing = max(1, len(df_sorted) // 10) # Aim for ~10 ticks
        tick_locs = np.arange(0, len(df_sorted), tick_spacing)
        tick_labels = df_sorted['spill'].iloc[tick_locs]
        
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.tick_params(axis='x', which='major', labelsize=10)

        # 6. Final adjustments and save.
        plt.tight_layout()
        output_filename = f"{self.plots_dir}/Overall_Total_Intensity_HalfSpills_vs_Spill_Summary.png"
        plt.savefig(output_filename, dpi=200, bbox_inches='tight')
        plt.close(fig)
        #print(f"\nOverall intensity of spill halves plot saved to: {output_filename}")

    def plot_max_intensity_overall(self, all_run_data):
        """SUMMARY PLOT: Plot the peak intensity recorded per spill vs spill number (chronological, not sequential) for all runs"""    
        if all_run_data.empty or all_run_data['max_intensity'].isnull().all(): return
        df_sorted = all_run_data.sort_values(by='spill').dropna(subset=['max_intensity']).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(15, 10))
        unique_runs = df_sorted['run'].unique()
        colormap = plt.get_cmap('hsv', len(unique_runs))
        for i, run_number in enumerate(unique_runs):
            run_data = df_sorted[df_sorted['run'] == run_number]
            ax.plot(run_data.index, run_data['max_intensity'], 'o', ms=4, color=colormap(i), label=f'Run {run_number}')
        ax.set_title("Peak 100µs Integrated Linearized Intensity vs. Spill Number")
        ax.set_xlabel("Spill Number (chronological, not sequential)")
        ax.set_ylabel("Peak 100µs Integrated Linearized Intensity"), ax.grid(True, linestyle='--'), ax.set_xlim(left=-1, right=len(df_sorted))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=10, title="Run Number", fontsize='small')
        tick_spacing = 40
        tick_locs = range(0, len(df_sorted), tick_spacing)
        tick_labels = df_sorted['spill'].iloc[tick_locs]
        ax.set_xticks(tick_locs), ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.tick_params(axis='x', which='major', labelsize=8)
        plt.subplots_adjust(bottom=0.3)
        output_filename = f"{self.plots_dir}/Overall_Max_Intensity_vs_Spill_Summary.png"
        plt.savefig(output_filename, dpi=200), plt.close(fig)
        #print(f"\nOverall Peak Intensity plot saved to: {output_filename}")

    def plot_duty_factor_overall(self, all_run_data):
        """SUMMARY PLOT: Plot the duty factor per spill vs spill number (chronological, not sequential) for all runs"""    
        if all_run_data.empty or all_run_data['duty_factor'].isnull().all(): return
        df_sorted = all_run_data.sort_values(by='spill').dropna(subset=['duty_factor']).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(15, 10))
        unique_runs = df_sorted['run'].unique()
        colormap = plt.get_cmap('hsv', len(unique_runs))
        for i, run_number in enumerate(unique_runs):
            run_data = df_sorted[df_sorted['run'] == run_number]
            ax.plot(run_data.index, run_data['duty_factor'], 'o', ms=4, color=colormap(i), label=f'Run {run_number}')
        ax.set_title("Duty Factor vs. Spill Number")
        ax.set_xlabel("Spill Number (chronological, not sequential)"), ax.set_ylabel("Spill Duty Factor")
        ax.grid(True, linestyle='--'), ax.set_xlim(left=-1, right=len(df_sorted)), ax.set_ylim(0, 1.05)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=10, title="Run Number", fontsize='small')
        tick_spacing = 40
        tick_locs = range(0, len(df_sorted), tick_spacing)
        tick_labels = df_sorted['spill'].iloc[tick_locs]
        ax.set_xticks(tick_locs), ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.tick_params(axis='x', which='major', labelsize=8)
        plt.subplots_adjust(bottom=0.3)
        output_filename = f"{self.plots_dir}/Overall_Duty_Factor_vs_Spill_Summary.png"
        plt.savefig(output_filename, dpi=200), plt.close(fig)
        #print(f"\nOverall Duty Factor plot saved to: {output_filename}")
        
    def plot_time_of_max_overall(self, all_run_data):
        """SUMMARY PLOT: Plot the time of max intensity per spill vs spill number (chronological, not sequential) for all runs"""    
        if all_run_data.empty or all_run_data['time_of_max'].isnull().all(): return
        df_sorted = all_run_data.sort_values(by='spill').dropna(subset=['time_of_max']).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(15, 10))
        unique_runs = df_sorted['run'].unique()
        colormap = plt.get_cmap('hsv', len(unique_runs))
        for i, run_number in enumerate(unique_runs):
            run_data = df_sorted[df_sorted['run'] == run_number]
            ax.plot(run_data.index, run_data['time_of_max'], 'o', ms=4, color=colormap(i), label=f'Run {run_number}')
        ax.set_title("Time of Peak Intensity vs. Spill Number")
        ax.set_xlabel("Spill Number (chronological, not sequential)")
        ax.set_ylabel("Time of Maximum Intensity (s)"), ax.grid(True, linestyle='--'), ax.set_xlim(left=-1, right=len(df_sorted))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=10, title="Run Number", fontsize='small')
        tick_spacing = 40
        tick_locs = range(0, len(df_sorted), tick_spacing)
        tick_labels = df_sorted['spill'].iloc[tick_locs]
        ax.set_xticks(tick_locs), ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.tick_params(axis='x', which='major', labelsize=8)
        plt.subplots_adjust(bottom=0.3)
        output_filename = f"{self.plots_dir}/Overall_Time_of_Max_vs_Spill_Summary.png"
        plt.savefig(output_filename, dpi=200), plt.close(fig)
        #print(f"\nOverall Time of Max plot saved to: {output_filename}")

    def plot_overall_time_of_max_histogram(self, all_run_data, weighted=False):
        """SUMMARY PLOT: Histogram of time of max intensity for all spills in all runs
                Weighted = False implies a traditional histogram with count on the y axis (number of spills)
                Weighted = True instead fills with weight equal to the total intensity of the run
        """    
        if all_run_data.empty: return
        df = all_run_data.dropna(subset=['time_of_max', 'total_intensity'])
        if df.empty: return
        fig, ax = plt.subplots(figsize=(12, 7))
        weights = df['total_intensity'] if weighted else None
        ax.hist(df['time_of_max'], bins=16, range=(0, 4), weights=weights, color='skyblue', edgecolor='black')
        title = 'Intensity-Weighted ' if weighted else ''
        title += 'Overall Distribution of Peak Intensity Times'
        ax.set_title(title), ax.set_xlabel("Time of Maximum Intensity (s)")
        ax.set_ylabel("Summed Spill Intensity" if weighted else "Number of Spills")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True)), ax.grid(True, linestyle='--', axis='y')
        filename_suffix = "_weighted" if weighted else ""
        output_filename = f"{self.plots_dir}/Overall_Time_of_Max_Hist{filename_suffix}.png"
        plt.savefig(output_filename, dpi=150), plt.close(fig)
        #print(f"\nOverall Time of Max histogram (weighted={weighted}) saved.")


    def plot_duty_factor_vs_time(self, analyzer):
        """Plots the duty factor as a function of time within a single spill."""
        if analyzer.duty_factor_vs_time.empty:
            return

        fig, ax = plt.subplots(figsize=(12, 7))
        df = analyzer.duty_factor_vs_time

        interval_s = self.config.DUTY_FACTOR_INTERVAL_S
        ax.plot(df['time_center'], df['duty_factor'], marker='o', linestyle='-')

        ax.set_title(f"Duty Factor vs. Time for Run {analyzer.run_num}, Spill {analyzer.spill_num}")
        ax.set_xlabel(f"Time (s) in {interval_s}s Intervals")
        ax.set_ylabel("Duty Factor")
        ax.grid(True, linestyle='--')
        ax.set_ylim(0, 1.05)

        output_filename = f"{self.plots_dir}/Duty_Factor_vs_Time_run{analyzer.run_num}_spill{analyzer.spill_num}.png"
        plt.savefig(output_filename, dpi=150)
        plt.close(fig)
        #print(f"Duty factor vs. time plot for spill {analyzer.spill_num} saved.")


    def plot_intensity_vs_time_barchart(self, analyzer):
        """
        Creates a bar chart of the integrated intensity within time intervals for a single spill.
        """
        if analyzer.intensity_vs_time.empty:
            return

        fig, ax = plt.subplots(figsize=(12, 7))
        df = analyzer.intensity_vs_time
        interval_s = self.config.DUTY_FACTOR_INTERVAL_S

        # Calculate the left edge of each bar for proper alignment
        left_edges = df['time_center'] - (interval_s / 2.0)

        ax.bar(left_edges, df['total_intensity'], width=interval_s, align='edge', edgecolor='black')

        ax.set_title(f"Integrated Linearized Intensity vs. Time for Run {analyzer.run_num}, Spill {analyzer.spill_num}")
        ax.set_xlabel(f"Time (s) in {interval_s}s Intervals")
        ax.set_ylabel("Integrated Linearized Intensity")
        ax.grid(True, axis='y', linestyle='--')

        # Ensure the x-axis limits are tight to the bars
        if not left_edges.empty:
            ax.set_xlim(left_edges.iloc[0], left_edges.iloc[-1] + interval_s)

        output_filename = f"{self.plots_dir}/Intensity_Barchart_vs_Time_run{analyzer.run_num}_spill{analyzer.spill_num}.png"
        plt.savefig(output_filename, dpi=150)
        plt.close(fig)
        #print(f"Intensity bar chart for spill {analyzer.spill_num} saved.")


    def plot_duty_factor_and_intensity_vs_time(self, analyzer):
        """
        Creates a combined plot with two y-axes, now including error bars.
        """
        if analyzer.intensity_vs_time.empty or analyzer.duty_factor_vs_time.empty:
            return

        fig, ax1 = plt.subplots(figsize=(14, 7))
        fig.suptitle(f"Linearized Intensity and Duty Factor vs. Time for Run {analyzer.run_num}, Spill {analyzer.spill_num}", fontsize=16)
        ax2 = ax1.twinx()
        interval_s = self.config.DUTY_FACTOR_INTERVAL_S

        # --- Plot 1: Intensity Bar Chart with Error Bars ---
        df_intensity = analyzer.intensity_vs_time
        left_edges = df_intensity['time_center'] - (interval_s / 2.0)
        bar_color = 'cornflowerblue'

        ax1.bar(left_edges, df_intensity['total_intensity'], width=interval_s, align='edge',
                color=bar_color, alpha=0.7, edgecolor='black', label='Intensity',
                yerr=df_intensity['total_intensity_err'], capsize=4) # Add error bars

        ax1.set_xlabel(f"Time (s) [Bin size {interval_s}s]")
        ax1.set_ylabel("Integrated Linearized Intensity", color=bar_color)
        ax1.tick_params(axis='y', labelcolor=bar_color)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

        # --- Plot 2: Duty Factor Line Plot with Error Bars ---
        df_duty = analyzer.duty_factor_vs_time
        line_color = 'tab:red'

        # Replace .plot() with .errorbar()
        ax2.errorbar(df_duty['time_center'], df_duty['duty_factor'], 
                     yerr=df_duty['duty_factor_uncertainty'],
                     marker='o', linestyle='-', color=line_color, label='Duty Factor', capsize=4)

        ax2.set_ylabel("Duty Factor", color=line_color)
        ax2.tick_params(axis='y', labelcolor=line_color)
        ax2.set_ylim(0, 1.05)

        # --- Final Touches ---
        if not left_edges.empty:
            ax1.set_xlim(left_edges.iloc[0], left_edges.iloc[-1] + interval_s)

        # Use a combined legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        output_filename = f"{self.plots_dir}/Combined_Intensity_DF_vs_Time_run{analyzer.run_num}_spill{analyzer.spill_num}.png"
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        #print(f"Combined intensity and duty factor plot for spill {analyzer.spill_num} saved.")


    def plot_peak_interval_histogram(self, run_number, analyzers_for_run):
        """Plots a histogram of the time between spikes, summed over a whole run."""
        if not analyzers_for_run: return

        all_intervals = []
        # Loop through each spill's analyzer object for this run
        for analyzer in analyzers_for_run:
            # The peaks_df is already sorted by time
            if len(analyzer.peaks_df) > 1:
                # Calculate intervals by taking the difference of consecutive peak times
                intervals = np.diff(analyzer.peaks_df['time'].values)
                all_intervals.extend(intervals)

        if not all_intervals:
            print(f"No peak intervals found to plot for Run {run_number}.")
            return

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.hist(all_intervals, bins=100, range=(0, max(0.1, np.percentile(all_intervals, 99))), color='mediumseagreen', edgecolor='black')
        ax.set_title(f'Distribution of Time Between Spikes for Run {run_number}')
        ax.set_xlabel('Time Between Consecutive Spikes (s)')
        ax.set_ylabel('Number of Occurrences')
        ax.grid(True, axis='y', linestyle='--')

        output_filename = f"{self.plots_dir}/Peak_Interval_Hist_run_{run_number}.png"
        plt.savefig(output_filename, dpi=150)
        plt.close(fig)
        #print(f"Peak interval histogram for Run {run_number} saved.")

    def plot_peak_width_histogram(self, run_number, analyzers_for_run):
        """Plots a histogram of the spike widths, summed over a whole run."""
        if not analyzers_for_run: return

        all_widths = []
        # Loop through each spill's analyzer object for this run
        for analyzer in analyzers_for_run:
            if not analyzer.peaks_df.empty:
                # Get the width data directly from the 'width_s' column
                all_widths.extend(analyzer.peaks_df['width_s'].tolist())

        if not all_widths: 
            print(f"No peak widths found to plot for Run {run_number}.")
            return

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.hist(all_widths, bins=50, range=(0, np.percentile(all_widths, 99.5) if all_widths else 1), color='darkorange', edgecolor='black')
        ax.set_title(f'Distribution of Spike Widths for Run {run_number}')
        ax.set_xlabel('Spike Width (s)')
        ax.set_ylabel('Number of Spikes')
        ax.grid(True, axis='y', linestyle='--')

        output_filename = f"{self.plots_dir}/Peak_Width_Hist_run_{run_number}.png"
        plt.savefig(output_filename, dpi=150)
        plt.close(fig)
        #print(f"Peak width histogram for Run {run_number} saved.")

    def plot_zoomed_spike(self, analyzer, spike_series, rank, sort_key=""):
        """Creates a zoomed-in plot centered on a spike, including its width."""
        if analyzer.data.empty: return

        # Extract all the spike properties from the passed Series
        spike_time = spike_series['time']
        spike_width_s = spike_series['width_s']
        width_height = spike_series['width_height']

        fig, ax = plt.subplots(figsize=(12, 7))
        window = self.config.SPIKE_ZOOM_WINDOW_S

        ax.plot(analyzer.data['time_s'], analyzer.data['intensity'], marker='.', linestyle='-', markersize=4, color='black')
        ax.set_xlim(spike_time - (window / 2), spike_time + (window / 2))
        ax.axvline(x=spike_time, color='r', linestyle='--', alpha=0.8, label=f"Spike Time: {spike_time:.4f}s")

        # Add the horizontal line for the spike width
        width_start_time = spike_time - (spike_width_s / 2)
        width_end_time = spike_time + (spike_width_s / 2)

        ax.hlines(y=width_height, xmin=width_start_time, xmax=width_end_time,
                  color='red', linewidth=2, label=f"Width: {spike_width_s*1000:.2f} ms")

        # Use the sort_key in the title for clarity
        ax.set_title(f"Zoomed View of Top Spike #{rank+1} by {sort_key} (Run: {analyzer.run_num}, Spill: {analyzer.spill_num})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Linearized Intensity")
        ax.legend()
        ax.grid(True)

        # Use the sort_key in the filename to prevent overwriting
        output_filename = f"{self.plots_dir}/Zoomed_Top_Spike_by_{sort_key}_Rank_{rank+1}_run{analyzer.run_num}_spill{analyzer.spill_num}.png"
        plt.savefig(output_filename, dpi=150)
        plt.close(fig)
        ##print(f"Zoomed plot for top spike #{rank+1} by {sort_key} saved.")




    def plot_spike_width_vs_time_scatter(self, all_peaks_df):
        """Creates a scatter plot of every spike's width vs. its time in the spill."""
        if all_peaks_df.empty: return

        fig, ax = plt.subplots(figsize=(12, 7))

        # Use a small marker size and alpha for better visibility with many points
        ax.scatter(all_peaks_df['time'], all_peaks_df['width_s'], s=5, alpha=0.3)

        ax.set_title('Spike Width vs. Time in Spill (All Spills)')
        ax.set_xlabel('Time in Spill (s)')
        ax.set_ylabel('Spike Width (s)')
        ax.grid(True, linestyle='--')
        # Set y-axis to log scale to better see the distribution of widths
        ax.set_yscale('log')

        output_filename = f"{self.plots_dir}/Overall_Width_vs_Time_Scatter.png"
        plt.savefig(output_filename, dpi=150)
        plt.close(fig)
        #print(f"\nOverall spike width vs. time scatter plot saved to: {output_filename}")

    def plot_spike_width_vs_time_hist2d(self, all_peaks_df):
        """Creates a 2D histogram of spike width vs. its time in the spill."""
        if all_peaks_df.empty: return

        fig, ax = plt.subplots(figsize=(12, 7))

        # Create the 2D histogram. Use LogNorm for the color scale to see low-count bins.
        counts, xedges, yedges, im = ax.hist2d(
            all_peaks_df['time'], 
            all_peaks_df['width_s'], 
            bins=(100, 100), # (bins_x, bins_y)
            norm=LogNorm(),
            range=[[0, 4], [0, np.percentile(all_peaks_df['width_s'], 99.5)]] # Set sensible ranges
        )

        fig.colorbar(im, ax=ax, label='Number of Spikes')
        ax.set_title('Spike Width vs. Time in Spill (All Spills)')
        ax.set_xlabel('Time in Spill (s)')
        ax.set_ylabel('Spike Width (s)')

        output_filename = f"{self.plots_dir}/Overall_Width_vs_Time_Hist2D.png"
        plt.savefig(output_filename, dpi=150)
        plt.close(fig)
        #print(f"Overall spike width vs. time 2D histogram saved to: {output_filename}")
        


    def plot_overall_peak_width_histogram(self, analyzers_by_run):
        """Plots a histogram of the spike widths, summed over all spills."""
        if not analyzers_by_run: return

        # Collect all peak widths from all spills into a single list
        all_widths = []
        # Loop through the dictionary of all processed analyzer objects
        for run_num in analyzers_by_run:
            for analyzer in analyzers_by_run[run_num]:
                if not analyzer.peaks_df.empty:
                    # Get the width data directly from the 'width_s' column
                    all_widths.extend(analyzer.peaks_df['width_s'].tolist())

        if not all_widths:
            print("No peak widths found to plot for the overall summary.")
            return

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.hist(all_widths, bins=100, range=(0, np.percentile(all_widths, 99.5) if all_widths else 1), color='darkorange', edgecolor='black')
        ax.set_title('Overall Distribution of Spike Widths (All Spills)')
        ax.set_xlabel('Spike Width (s)')
        ax.set_ylabel('Number of Spikes')
        ax.grid(True, axis='y', linestyle='--')

        output_filename = f"{self.plots_dir}/Overall_Peak_Width_Hist.png"
        plt.savefig(output_filename, dpi=150)
        plt.close(fig)
        #print(f"Overall peak width histogram saved to: {output_filename}")

    def plot_overall_peak_interval_histogram(self, analyzers_by_run):
        """Plots a histogram of the time between spikes, summed over all spills."""
        if not analyzers_by_run: return

        # Collect all peak intervals from all spills into a single list
        all_intervals = []
        # Loop through the dictionary of all processed analyzer objects
        for run_num in analyzers_by_run:
            for analyzer in analyzers_by_run[run_num]:
                # If a spill has more than one peak, calculate the time between them
                if len(analyzer.peaks_df) > 1:
                    # Calculate intervals from the 'time' column of the peaks_df
                    intervals = np.diff(analyzer.peaks_df['time'].values)
                    all_intervals.extend(intervals)

        if not all_intervals:
            print("No peak intervals found to plot for the overall summary.")
            return

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.hist(all_intervals, bins=100, range=(0, max(0.1, np.percentile(all_intervals, 99))), color='mediumseagreen', edgecolor='black')
        ax.set_title('Overall Distribution of Time Between Spikes (All Spills)')
        ax.set_xlabel('Time Between Consecutive Spikes (s)')
        ax.set_ylabel('Number of Occurrences')
        ax.grid(True, axis='y', linestyle='--')

        output_filename = f"{self.plots_dir}/Overall_Peak_Interval_Hist.png"
        plt.savefig(output_filename, dpi=150)
        plt.close(fig)
        #print(f"\nOverall peak interval histogram saved to: {output_filename}")



    def plot_spike_width_by_spill_half(self, all_peaks_df):
        """
        Compares the distribution of spike widths from the first half of spills
        versus the second half.
        """
        if all_peaks_df.empty: return

        # Assuming a 4-second spill, the midpoint is 2.0 seconds.
        # This could be made dynamic if spill durations vary significantly.
        spill_midpoint = 2.0

        # Separate the spikes into two groups based on their time of occurrence
        first_half_widths = all_peaks_df[all_peaks_df['time'] < spill_midpoint]['width_s']
        second_half_widths = all_peaks_df[all_peaks_df['time'] >= spill_midpoint]['width_s']

        if first_half_widths.empty or second_half_widths.empty:
            print("Not enough data to compare spike widths by spill half.")
            return

        fig, ax = plt.subplots(figsize=(12, 7))

        # Define shared histogram properties
        # Focus the range on the 0.44 ms spike by setting a tight x-limit
        bins = 100
        hist_range = (0, 0.005) # Range up to 2 ms to see the 0.44 ms spike clearly

        # Plot the tw   o histograms on top of each other with transparency
        ax.hist(first_half_widths, bins=bins, range=hist_range,
                label='First Half of Spill (0-2s)', color='blue', histtype='step', linewidth=2)

        ax.hist(second_half_widths, bins=bins, range=hist_range,
                label='Second Half of Spill (2-4s)', color='red', histtype='step', linewidth=2)


        # Add a vertical line to highlight the 0.44 ms hypothesis
        #ax.axvline(x=0.00044, color='black', linestyle='--', label='0.44 ms')

        ax.set_title('Comparison of Spike Widths by Time in Spill (All Spills)')
        ax.set_xlabel('Spike Width (s)')
        ax.set_ylabel('Number of Spikes')
        ax.legend()
        ax.grid(True, axis='y', linestyle='--')

        output_filename = f"{self.plots_dir}/Overall_Width_vs_Spill_Half_Comparison.png"
        plt.savefig(output_filename, dpi=150)
        plt.close(fig)
        #print(f"\nSpike width by spill half comparison plot saved to: {output_filename}")



    def plot_width_vs_intensity_hist2d(self, all_peaks_df):
        """Creates a 2D histogram of spike width vs. spike intensity."""
        if all_peaks_df.empty: return

        fig, ax = plt.subplots(figsize=(12, 7))
        counts, xedges, yedges, im = ax.hist2d(
            all_peaks_df['intensity'],
            all_peaks_df['width_s'],
            bins=(100, 100),
            norm=LogNorm()#,
            #range=[[0, np.percentile(all_peaks_df['intensity'], 99.5)],
                   #[0, np.percentile(all_peaks_df['width_s'], 99.5)]]
        )
        fig.colorbar(im, ax=ax, label='Number of Spikes')
        ax.set_title('Spike Width vs. Spike Linearized Intensity (All Spills)')
        ax.set_xlabel('Spike Linearized Intensity')
        ax.set_ylabel('Spike Width (s)')

        output_filename = f"{self.plots_dir}/Overall_Width_vs_Intensity_Hist2D.png"
        plt.savefig(output_filename, dpi=150)
        plt.close(fig)
        #print(f"Overall spike width vs. intensity 2D histogram saved to: {output_filename}")

    def plot_width_vs_prominence_hist2d(self, all_peaks_df):
        """Creates a 2D histogram of spike width vs. spike prominence."""
        if all_peaks_df.empty: return

        fig, ax = plt.subplots(figsize=(12, 7))
        counts, xedges, yedges, im = ax.hist2d(
            all_peaks_df['prominence'],
            all_peaks_df['width_s'],
            bins=(100, 100),
            norm=LogNorm()#,
            #range=[[0, np.percentile(all_peaks_df['prominence'], 99.5)],
                   #[0, np.percentile(all_peaks_df['width_s'], 99.5)]]
        )
        fig.colorbar(im, ax=ax, label='Number of Spikes')
        ax.set_title('Spike Width vs. Spike Prominence (All Spills)')
        ax.set_xlabel('Spike Prominence')
        ax.set_ylabel('Spike Width (s)')

        output_filename = f"{self.plots_dir}/Overall_Width_vs_Prominence_Hist2D.png"
        plt.savefig(output_filename, dpi=150)
        plt.close(fig)
        #print(f"Overall spike width vs. prominence 2D histogram saved to: {output_filename}")

    def plot_intensity_vs_time_hist2d(self, all_peaks_df):
        """Creates a 2D histogram of spike intensity vs. time in spill."""
        if all_peaks_df.empty: return

        fig, ax = plt.subplots(figsize=(12, 7))
        counts, xedges, yedges, im = ax.hist2d(
            all_peaks_df['time'],
            all_peaks_df['intensity'],
            bins=(100, 100),
            norm=LogNorm()#,
            #range=[[0, 4], [0, np.percentile(all_peaks_df['intensity'], 99.5)]]
        )
        fig.colorbar(im, ax=ax, label='Number of Spikes')
        ax.set_title('Spike Linearized Intensity vs. Time in Spill (All Spills)')
        ax.set_xlabel('Time in Spill (s)')
        ax.set_ylabel('Spike Linearized Intensity')

        output_filename = f"{self.plots_dir}/Overall_Intensity_vs_Time_Hist2D.png"
        plt.savefig(output_filename, dpi=150)
        plt.close(fig)
        #print(f"Overall spike intensity vs. time 2D histogram saved to: {output_filename}")

    def plot_spike_intensity_by_spill_half(self, all_peaks_df):
        """
        Compares the distribution of spike intensities from the first half of spills 
        vs. the second half. Saves both linear and log scale versions.
        """
        if all_peaks_df.empty: return

        spill_midpoint = 2.0
        first_half_intensities = all_peaks_df[all_peaks_df['time'] < spill_midpoint]['intensity']
        second_half_intensities = all_peaks_df[all_peaks_df['time'] >= spill_midpoint]['intensity']

        if first_half_intensities.empty or second_half_intensities.empty:
            print("Not enough data to compare spike intensities by spill half.")
            return

        # --- Create the plot once ---
        fig, ax = plt.subplots(figsize=(12, 7))
        bins = 100
        # Use a shared range for both histograms for a direct comparison
        hist_range = (0, np.percentile(all_peaks_df['intensity'], 99.5))

        # --- Draw histograms as lines using histtype='step' ---
        ax.hist(first_half_intensities, bins=bins, range=hist_range,
                label='First Half of Spill (0-2s)', color='blue', histtype='step', linewidth=2)
        ax.hist(second_half_intensities, bins=bins, range=hist_range,
                label='Second Half of Spill (2-4s)', color='red', histtype='step', linewidth=2)

        # --- Set shared labels and titles ---
        ax.set_title('Comparison of Spike Intensities by Time in Spill (All Spills)')
        ax.set_xlabel('Spike Linearized Intensity')
        ax.set_ylabel('Number of Spikes')
        ax.legend()
        ax.grid(True, axis='y', linestyle='--')

        # --- 1. Save the Linear Scale Version ---
        linear_filename = f"{self.plots_dir}/Overall_Intensity_vs_Spill_Half_Comparison_Linear.png"
        plt.savefig(linear_filename, dpi=150, bbox_inches='tight')
        #print(f"Spike intensity comparison plot saved to: {linear_filename}")

        # --- 2. Save the Log Scale Version ---
        ax.set_yscale('log') # Now, set the y-axis to log scale
        log_filename = f"{self.plots_dir}/Overall_Intensity_vs_Spill_Half_Comparison_Logy.png"
        plt.savefig(log_filename, dpi=150, bbox_inches='tight')
        #print(f"Log scale spike intensity comparison plot saved to: {log_filename}")

        # Close the figure once at the very end
        plt.close(fig)

    def plot_overall_spike_time_histogram(self, all_peaks_df):
        """Creates a histogram of the occurrence time of all spikes found by find_peaks."""
        if all_peaks_df.empty: return

        # --- Create the plot once ---
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.hist(all_peaks_df['time'], bins=100, range=(0, 4), color='purple', edgecolor='black')

        # --- Set shared labels and titles ---
        ax.set_title('Overall Distribution of Spike Times (from find_peaks)')
        ax.set_xlabel('Time in Spill (s)')
        ax.set_ylabel('Number of Spikes Found')
        ax.grid(True, axis='y', linestyle='--')

        # --- 1. Save the Linear Scale Version ---
        linear_filename = f"{self.plots_dir}/Overall_Spike_Time_Hist_Linear.png"
        plt.savefig(linear_filename, dpi=150, bbox_inches='tight')
        #print(f"\nOverall spike time histogram saved to: {linear_filename}")

        # --- 2. Save the Log Scale Version ---
        ax.set_yscale('log') # Now, set the y-axis to log scale
        log_filename = f"{self.plots_dir}/Overall_Spike_Time_Hist_Logy.png"
        plt.savefig(log_filename, dpi=150, bbox_inches='tight')
        #print(f"Log scale spike time histogram saved to: {log_filename}")

        # Close the figure once at the very end
        plt.close(fig)


    def plot_overall_top_fft_frequency_histogram(self, all_run_data):
        """
        Plots a histogram of the most prominent FFT frequency from each spill.
        This shows which frequencies appear most often as the dominant instability.
        """
        if all_run_data.empty or 'top_fft_freq' not in all_run_data.columns:
            return

        # Drop any spills where FFT calculation might have failed
        top_freqs = all_run_data['top_fft_freq'].dropna()

        if top_freqs.empty:
            print("No top FFT frequency data found to plot.")
            return

        fig, ax = plt.subplots(figsize=(18, 9)) # Increased figure size for readability

        # 1. Use 1 Hz bins from 0 to 750 Hz
        bins = 150
        hist_range = (0, 150)

        # Create the histogram and get the patches (the bars)
        counts, bin_edges, patches = ax.hist(top_freqs, bins=bins, range=hist_range, color='darkcyan', edgecolor='black', zorder=2)

        ax.set_title('Frequency of Dominant FFT Peaks (All Spills)')
        ax.set_xlabel('Dominant Frequency (Hz)')
        ax.set_ylabel('Number of Spills')

        # 2. Increase the number of x-axis tick marks
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
        ax.grid(which='major', linestyle='-', zorder=1)
        ax.grid(which='minor', linestyle=':', alpha=0.6, zorder=1)
        
        # Ensure y-axis ticks are integers
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # 3. Label the lower bound frequency on top of each filled bar
        for patch in patches:
            height = patch.get_height()
            if height > 0:
                # Get the frequency (x-position) and format the label
                freq_label = str(int(patch.get_x()))+" - "+str(int(patch.get_x()+patch.get_width() ) )
                # Place the text label
                ax.text(patch.get_x() + patch.get_width() / 2., height + 0.5, freq_label,
                        ha='center', va='bottom', rotation=90, fontsize=7)

        # Adjust y-axis limit to make space for the labels
        if counts.any():
             ax.set_ylim(top=ax.get_ylim()[1] * 1.2)

        plt.tight_layout()
        output_filename = f"{self.plots_dir}/Overall_Top_FFT_Frequency_Hist.png"
        plt.savefig(output_filename, dpi=150)
        plt.close(fig)


    def plot_weighted_fft_peak_histogram(self, all_fft_peaks):
        """
        Plots a magnitude-weighted histogram of all significant FFT peaks from all spills.
        This shows the total 'power' at different frequencies. It now also identifies
        and labels the 8 tallest peaks in the resulting distribution.
        """
        if not all_fft_peaks:
            print("No FFT peak data found to generate weighted histogram.")
            return

        # Unzip the list of (frequency, magnitude) tuples into two separate lists
        frequencies, magnitudes = zip(*all_fft_peaks)

        fig, ax = plt.subplots(figsize=(18, 9)) # Made the plot wider for labels

        # A 10kHz sampling rate gives a theoretical max frequency (Nyquist) of 5000 Hz.
        # We will look up to 2500 Hz.
        bins = 1000  # Increased bins for better resolution over the wider range
        hist_range = (0, 1000)

        # Create the histogram, weighted by the magnitude of each peak
        counts, bin_edges, patches = ax.hist(frequencies, bins=bins, range=hist_range,
                                             weights=magnitudes,
                                             color='firebrick', zorder=2)

        # --- NEW: Find and Label the 8 Tallest Peaks ---
        # Use find_peaks on the histogram counts to find the indices of the peaks
        peak_indices, properties = find_peaks(counts, height=np.mean(counts), prominence=1)

        if len(peak_indices) > 0:
            peak_heights = properties['peak_heights']
            sorted_indices = np.argsort(peak_heights)[::-1] # Sort descending

            # Loop through the top 8 peaks
            for i in sorted_indices[:15]:
                peak_index = peak_indices[i]
                peak_height = counts[peak_index]
                bin_center = (bin_edges[peak_index] + bin_edges[peak_index+1]) / 2.0

                # Add an annotation with an arrow
                ax.annotate(f'{bin_center:.1f} Hz',
                            xy=(bin_center, peak_height),
                            xytext=(bin_center, peak_height + 0.05 * ax.get_ylim()[1]),
                            ha='center',
                            va='bottom',
                            fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4))
        # --- End of New Section ---

        ax.set_title('Magnitude-Weighted Distribution of All FFT Peaks (All Spills)')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Sum of FFT Peak Magnitudes')
        ax.grid(True, linestyle='--', zorder=1)

        ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
        ax.grid(which='major', linestyle='-', zorder=1)
        ax.grid(which='minor', linestyle=':', alpha=0.6, zorder=1)


        # Adjust y-axis limit to make space for annotations
        if len(peak_indices) > 0:
            ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

        # You may want to re-enable the log scale, as it's very useful for this type of plot
        # ax.set_yscale('log')
        # if len(peak_indices) > 0:
        #     ax.set_ylim(bottom=1) # Reset bottom limit if using log scale

        plt.tight_layout()
        output_filename = f"{self.plots_dir}/Overall_Weighted_FFT_Peak_Hist.png"
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

    """def plot_ranked_choice_fft_histogram(self, all_ranked_peaks):

        if not all_ranked_peaks:
            print("No ranked FFT peak data found to generate histogram.")
            return

        # Unzip the list of (frequency, score) tuples
        frequencies, scores = zip(*all_ranked_peaks)

        fig, ax = plt.subplots(figsize=(15, 8))

        # Use a wider range and more bins for detail
        bins = 400
        hist_range = (0, 400)

        # Create the histogram, weighted by the rank score of each peak
        ax.hist(frequencies, bins=bins, range=hist_range,
                weights=scores,
                color='darkslateblue')

        # 2. Increase the number of x-axis tick marks
        ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
        ax.grid(which='major', linestyle='-', zorder=1)
        ax.grid(which='minor', linestyle=':', alpha=0.6, zorder=1)
        
        # Ensure y-axis ticks are integers
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))


        ax.set_title(f'Ranked-Choice FFT Peak Score (Top {self.config.N_RANKED_PEAKS} Peaks per Spill)')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Total Ranked-Choice Score')
        ax.grid(True, linestyle='--', alpha=0.7)
        #ax.set_yscale('log') # Log scale is often useful for these plots
        #ax.set_ylim(bottom=1)

        plt.tight_layout()
        output_filename = f"{self.plots_dir}/Overall_Ranked_Choice_FFT_Hist.png"
        plt.savefig(output_filename, dpi=150)
        plt.close(fig)
    """

    def plot_ranked_choice_fft_histogram(self, all_ranked_peaks):
        """
        Plots a histogram of FFT peak frequencies weighted by a ranked-choice
        (Borda count) scoring system. It now also identifies and labels the 8
        tallest peaks in the resulting distribution.
        """
        if not all_ranked_peaks:
            print("No ranked FFT peak data found to generate histogram.")
            return

        # Unzip the list of (frequency, score) tuples
        frequencies, scores = zip(*all_ranked_peaks)

        fig, ax = plt.subplots(figsize=(18, 9)) # Made the plot wider for labels

        # Use a wider range and more bins for detail
        bins = 400
        hist_range = (0, 400)

        # Create the histogram, weighted by the rank score of each peak
        # This now returns the counts (bar heights) and bin_edges
        counts, bin_edges, patches = ax.hist(frequencies, bins=bins, range=hist_range,
                                             weights=scores,
                                             color='darkslateblue', zorder=2)

        # --- NEW: Find and Label the 12 Tallest Peaks ---
        # Use find_peaks on the histogram counts to find the indices of the peaks
        # We set a minimum height to avoid labeling tiny bumps
        peak_indices, properties = find_peaks(counts, height=np.mean(counts), prominence=1)

        if len(peak_indices) > 0:
            # Get the heights of the found peaks
            peak_heights = properties['peak_heights']

            # Get the indices that would sort the peak heights in descending order
            sorted_indices = np.argsort(peak_heights)[::-1]

            # Loop through the top 12 peaks (or fewer if less than 8 were found)
            for i in sorted_indices[:12]:
                peak_index = peak_indices[i]
                peak_height = counts[peak_index]

                # Calculate the center of the bin for the x-coordinate
                bin_center = (bin_edges[peak_index] + bin_edges[peak_index+1]) / 2.0

                # Add an annotation with an arrow
                ax.annotate(f'{bin_center:.1f} Hz',
                            xy=(bin_center, peak_height),
                            xytext=(bin_center, peak_height + 0.05 * ax.get_ylim()[1]), # Position text above the peak
                            ha='center',
                            va='bottom',
                            fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4))


        # Increase the number of x-axis tick marks
        ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
        ax.grid(which='major', linestyle='-', zorder=1)
        ax.grid(which='minor', linestyle=':', alpha=0.6, zorder=1)

        # Ensure y-axis ticks are integers
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_title(f'Ranked-Choice FFT Peak Score (Top {self.config.N_RANKED_PEAKS} Peaks per Spill)')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Total Ranked-Choice Score')
        ax.grid(True, linestyle='--', alpha=0.7)

        # Adjust y-axis limit to make space for annotations
        if len(peak_indices) > 0:
            ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

        plt.tight_layout()
        output_filename = f"{self.plots_dir}/Overall_Ranked_Choice_FFT_Hist.png"
        plt.savefig(output_filename, dpi=150)
        plt.close(fig)


    def plot_stability_profile(self, mean_profile, std_profile):
        """
        Plots the mean intensity profile across all spills, with a shaded
        band representing +/- one standard deviation.
        """
        if mean_profile.empty:
            return

        fig, ax = plt.subplots(figsize=(15, 7))
        time_axis = mean_profile.index

        # Plot the mean intensity as a solid line
        ax.plot(time_axis, mean_profile, color='blue', label='Mean Intensity', zorder=2)

        # Create the shaded standard deviation band
        ax.fill_between(time_axis,
                        mean_profile - std_profile,
                        mean_profile + std_profile,
                        color='blue', alpha=0.2, label='±1 Std. Deviation', zorder=1)

        ax.set_title('Spill Stability Profile (All Spills)')
        ax.set_xlabel('Time in Spill (s)')
        ax.set_ylabel('Linearized Intensity')
        ax.set_xlim(time_axis.min(), time_axis.max())
        ax.grid(True, linestyle='--')
        ax.legend()

        output_filename = f"{self.plots_dir}/Overall_Stability_Profile.png"
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def plot_fft_spectrogram(self, all_spectra, frequencies):
        """
        Creates a 2D spectrogram (waterfall plot) of the FFT power spectrum
        for every spill in the dataset.
        """
        if not all_spectra:
            return

        # Stack the 1D spectra into a 2D numpy array
        # We transpose here so that each column is a spill, which is more natural for imshow
        spectrogram_data = np.vstack(all_spectra).T

        fig, ax = plt.subplots(figsize=(15, 10))

        # Use imshow to display the 2D array.
        # LogNorm is crucial for seeing both faint and strong signals.
        # extent sets the correct axis labels.
        # aspect='auto' allows the plot to be non-square.
        im = ax.imshow(spectrogram_data,
                       aspect='auto',
                       origin='lower',
                       norm=LogNorm(),
                       extent=[0, len(all_spectra), frequencies.min(), frequencies.max()])

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('FFT Power Spectrum Magnitude')


        ax.set_title('FFT Spectrogram of All Spills')
        ax.set_xlabel('Spill Count (Chronological not sequential)')
        ax.set_ylabel('Frequency (Hz)')

        ax.yaxis.set_major_locator(mticker.MultipleLocator(500))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(100))

        ax.tick_params(axis='both', which='major', direction='out', length=10, width=1.5)
        ax.tick_params(axis='both', which='minor', direction='out', length=5, width=0.75)

        output_filename = f"{self.plots_dir}/Overall_FFT_Spectrogram.png"
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"FFT Spectrogram plot saved to: {output_filename}")

    def plot_fft_spectrogram_zoomed(self, all_spectra, frequencies, freq_max=800, major_marker=100):
        """
        Creates a zoomed-in 2D spectrogram of the FFT power spectrum, focusing
        on a specific frequency range.
        """
        if not all_spectra:
            print("No spectra available for zoomed spectrogram.")
            return

        # Stack the 1D spectra into a 2D numpy array and transpose it
        spectrogram_data = np.vstack(all_spectra).T

        # --- Select the data for the zoomed frequency range ---
        # Create a boolean mask for the desired frequency range
        zoom_mask = (frequencies >= 0) & (frequencies <= freq_max)

        # Apply the mask to the data and the frequency axis
        zoomed_spectrogram_data = spectrogram_data[zoom_mask, :]
        zoomed_frequencies = frequencies[zoom_mask]

        if zoomed_frequencies.size == 0:
            print(f"No frequency data found in the 0-{freq_max} Hz range.")
            return

        fig, ax = plt.subplots(figsize=(15, 10))

        im = ax.imshow(zoomed_spectrogram_data,
                       aspect='auto',
                       origin='lower',
                       norm=LogNorm(),
                       extent=[0, len(all_spectra), zoomed_frequencies.min(), zoomed_frequencies.max()])

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('FFT Power Spectrum Magnitude')

        ax.yaxis.set_major_locator(mticker.MultipleLocator(major_marker)) # Major tick every 100 Hz
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(major_marker/10))  # Minor tick every 25 Hz

        ax.tick_params(axis='both', which='major', direction='out', length=10, width=1.5)
        ax.tick_params(axis='both', which='minor', direction='out', length=5, width=0.75)
      
        ax.set_title(f'Zoomed FFT Spectrogram of All Spills (0-{freq_max} Hz)')
        ax.set_xlabel('Spill Count (Chronological not sequential)')
        ax.set_ylabel('Frequency (Hz)')

        output_filename = f"{self.plots_dir}/Overall_FFT_Spectrogram_Zoomed_{freq_max}Hz.png"
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Zoomed FFT Spectrogram plot saved to: {output_filename}")

    def plot_duty_factor_comparison(self, summary_df):
        """Plots a 2D histogram comparing the calculated DF and the ACNET DF."""

        # Define the columns that this plot requires
        required_cols = ['duty_factor', 'I:FTSDF']

        # Check if the required columns exist before proceeding
        if not all(col in summary_df.columns for col in required_cols):
            print("  - WARNING: Skipping Duty Factor comparison plot because required columns are missing from the data.")
            return

        # Explicitly convert the ACNET data to a numeric type.
        # The 'coerce' option will turn any non-numeric text values into NaN (Not a Number).
        summary_df['I:FTSDF'] = pd.to_numeric(summary_df['I:FTSDF'], errors='coerce')

        # Now, drop rows where any of the required columns have missing or non-numeric values.
        df_filtered = summary_df.dropna(subset=required_cols)

        if df_filtered.empty:
            print("Not enough data to create duty factor comparison plot after cleaning.")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # The ACNET DF is in percent, so this division will now work correctly
        acnet_df = df_filtered['I:FTSDF'] / 100.0

        # Create the 2D histogram
        hist = ax.hist2d(df_filtered['duty_factor'], acnet_df, bins=50, cmin=1, norm=LogNorm())

        # Add a y=x line for reference
        ax.plot([0, 1], [0, 1], 'r--', label='y=x (Perfect Agreement)')

        fig.colorbar(hist[3], ax=ax, label='Number of Spills')
        ax.set_title('Comparison of Calculated Duty Factor vs. ACNET Duty Factor (I:FTSDF)')
        ax.set_xlabel('Calculated Duty Factor (from 10kHz FreqHist)')
        ax.set_ylabel('ACNET Duty Factor (from 53kHz I:FTSDF)')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--')
        ax.legend()

        output_filename = f"{self.plots_dir}/Overall_DutyFactor_Comparison.png"
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    """
    def plot_acnet_variable_vs_spill(self, summary_df, variable_name, plot_title, yaxis_label):

        # --- 1. Prepare and Clean Data ---
        df_sorted = summary_df.copy() # Work on a copy
        
        # Ensure the variable column is numeric, coercing errors
        df_sorted[variable_name] = pd.to_numeric(df_sorted[variable_name], errors='coerce')

        # --- START: NEW FILTER FOR BAD VALUES ---
        # Check if the variable is one known to have bad readings
        if variable_name.startswith('E:M3T'):
            bad_value_threshold = -9000
            # Replace values below the threshold with NaN (Not a Number)
            df_sorted.loc[df_sorted[variable_name] < bad_value_threshold, variable_name] = np.nan
        # --- END: NEW FILTER FOR BAD VALUES ---

        # Now, drop all rows with any NaN values in the target column and sort
        #df_sorted.dropna(subset=[variable_name], inplace=True)
        # df_sorted.sort_values(by='spill', inplace=True)

        df_sorted = df.sort_values(by='spill').dropna(subset=[variable_name]).reset_index(drop=True)


        if df_sorted.empty:
            print(f"  - WARNING: No valid data to plot for {variable_name} vs. spill.")
            return

        # --- 2. Create Plot ---
        fig, ax = plt.subplots(figsize=(15, 8))

        unique_runs = df_sorted['run'].unique()
        colormap = plt.get_cmap('hsv', len(unique_runs))
        for i, run_number in enumerate(unique_runs):
            run_data = df_sorted[df_sorted['run'] == run_number]
            
            # Plot against the actual 'spill' column for an accurate x-axis
            ax.plot(run_data['spill'], run_data[variable_name], 'o', ms=4, color=colormap(i), label=f'Run {run_number}')

        # --- 3. Format Plot ---
        ax.set_title(plot_title)
        ax.set_xlabel("Spill Number")
        ax.set_ylabel(yaxis_label)
        ax.grid(True, linestyle='--')
        
        ax.legend(loc='best', title="Run Number")
        ax.ticklabel_format(style='plain', axis='x')
        
        plt.tight_layout()

        # --- 4. Save Plot ---
        safe_filename = variable_name.replace(':', '_')
        output_filename = f"{self.plots_dir}/Overall_{safe_filename}_vs_Spill_Summary.png"
        plt.savefig(output_filename, dpi=200)
        plt.close(fig)
    """
    
    def plot_acnet_over_time(self, summary_df, variable_name):

        # --- 1. Validate Data ---
        # Check if the required columns exist in the DataFrame
        if variable_name not in summary_df.columns or 'acnet_timestamp' not in summary_df.columns:
            print(f"  - WARNING: Cannot plot '{variable_name}'. Required columns are missing.")
            return

        # --- 2. Prepare Data ---
        # Create a clean copy with only the necessary columns
        df = summary_df[['acnet_timestamp', variable_name]].copy()
        
        # Ensure the variable column is numeric, converting any non-numbers to NaN
        df[variable_name] = pd.to_numeric(df[variable_name], errors='coerce')
        
        # Check if the variable is one of the beam position monitors
        if variable_name.startswith('E:M3T'):
            # Define a threshold for what is considered a "bad" reading
            bad_value_threshold = -9000
            # Replace values below the threshold with NaN (Not a Number)
            df.loc[df[variable_name] < bad_value_threshold, variable_name] = np.nan

        # Drop any rows with missing data and sort chronologically
        df.dropna(inplace=True)
        df.sort_values(by='acnet_timestamp', inplace=True)

        if df.empty:
            print(f"  - WARNING: No valid, plottable data found for '{variable_name}'.")
            return

        # --- 3. Create Plot ---
        fig, ax = plt.subplots(figsize=(15, 7))
        
        # Create a scatter plot of the variable vs. time
        ax.plot(df['acnet_timestamp'], df[variable_name], marker='o', linestyle='', ms=3, alpha=0.7)

        # --- 4. Format Plot ---
        # Get metadata for prettier labels and titles
        metadata = ACNET_METADATA.get(variable_name, {'title': variable_name, 'unit': 'Unknown'})
        plot_title = metadata['title'].replace(' vs. Spill', ' over Time')
        yaxis_label = f"{metadata['title'].split(' vs.')[0]} ({metadata['unit']})"
        
        ax.set_title(plot_title)
        ax.set_xlabel("Date and Time")
        ax.set_ylabel(yaxis_label)
        ax.grid(True, linestyle='--')
        
        # Automatically format the date labels on the x-axis to prevent overlap
        fig.autofmt_xdate()

        # --- 5. Save Plot ---
        safe_filename = variable_name.replace(':', '_')
        output_filename = f"{self.plots_dir}/ACNET_{safe_filename}_over_Time.png"
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  - Successfully created time-series plot for {variable_name}")


    """


    def plot_acnet_grid_vs_spill(self, summary_df, variables_to_plot, plot_title, output_filename):

        n_vars = len(variables_to_plot)
        if n_vars == 0:
            return

        # Create a figure with n_vars subplots stacked vertically, sharing the x-axis
        fig, axes = plt.subplots(n_vars, 1, figsize=(15, 4 * n_vars), sharex=True)
        fig.suptitle(plot_title, fontsize=18)

        # If there's only one variable, axes is not a list, so we make it one
        if n_vars == 1:
            axes = [axes]

        # Get a consistent color map for all runs across all subplots
        all_runs = summary_df['run'].unique()
        colormap = plt.get_cmap('hsv', len(all_runs))
        run_color_map = {run: colormap(i) for i, run in enumerate(all_runs)}

        # Loop through each variable to create its subplot
        for i, var_name in enumerate(variables_to_plot):
            ax = axes[i]
            
            # --- Data Preparation ---
            if var_name not in summary_df.columns:
                ax.text(0.5, 0.5, f"Data for '{var_name}' not found", ha='center', va='center', style='italic')
                continue

            df = summary_df.copy()
            df[var_name] = pd.to_numeric(df[var_name], errors='coerce')

            if var_name.startswith('E:M3T'):
                df.loc[df[var_name] < -9000, var_name] = np.nan
            
            df.dropna(subset=[var_name], inplace=True)
            
            if df.empty:
                ax.text(0.5, 0.5, f"No valid data for '{var_name}'", ha='center', va='center', style='italic')
                continue

            # --- Plotting ---
            # Plot data for each run with a consistent color
            for run_number, run_data in df.groupby('run'):
                ax.plot(run_data['spill'], run_data[var_name], 'o', ms=2, 
                        color=run_color_map[run_number], label=f'Run {run_number}')

            # --- Formatting ---
            metadata = ACNET_METADATA.get(var_name, {'title': var_name, 'unit': 'Unknown'})
            yaxis_label = f"{metadata['title'].split(' vs.')[0]} ({metadata['unit']})"
            ax.set_ylabel(yaxis_label)
            ax.grid(True, linestyle='--')
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # Use scientific notation for y-axis

        # Add a single shared x-label and a master legend
        axes[-1].set_xlabel("Spill Number")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', title="Run Number", bbox_to_anchor=(0.98, 0.95))

        plt.tight_layout(rect=[0, 0, 0.9, 0.96]) # Adjust layout to make room for suptitle and legend
        plt.savefig(f"{self.plots_dir}/{output_filename}", dpi=150)
        plt.close(fig)
        print(f"  - Successfully created grid plot: {output_filename}")
    """


    def plot_acnet_variable_vs_spill(self, summary_df, variable_name, plot_title, yaxis_label):
        """
        Generic function to plot an ACNET variable vs. spill number for all runs.
        Plots chronologically, not sequentially, to remove gaps between runs.
        """
        df = summary_df.copy()
        df[variable_name] = pd.to_numeric(df[variable_name], errors='coerce')
        if variable_name.startswith('E:M3T'):
            df.loc[df[variable_name] < -9000, variable_name] = np.nan
        
        # REVISED: Sort and then RESET THE INDEX to create a chronological index
        df_sorted = df.sort_values(by='spill').dropna(subset=[variable_name]).reset_index(drop=True)

        if df_sorted.empty:
            return

        fig, ax = plt.subplots(figsize=(15, 8))

        unique_runs = df_sorted['run'].unique()
        colormap = plt.get_cmap('hsv', len(unique_runs))
        for i, run_number in enumerate(unique_runs):
            run_data = df_sorted[df_sorted['run'] == run_number]
            
            # REVISED: Plot against the new chronological index
            ax.plot(run_data.index, run_data[variable_name], 'o', ms=4, color=colormap(i), label=f'Run {run_number}')

        ax.set_title(plot_title)
        ax.set_ylabel(yaxis_label)
        ax.grid(True, linestyle='--')
        
        # REVISED: Set the x-label to indicate chronological order
        ax.set_xlabel("Spill Event (Chronological)")

        # REVISED: Add custom tick labels to show the actual spill number for context
        tick_spacing = max(1, len(df_sorted) // 15) # Aim for ~15 ticks
        tick_locs = np.arange(0, len(df_sorted), tick_spacing)
        tick_labels = df_sorted['spill'].iloc[tick_locs]
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.tick_params(axis='x', which='major', labelsize=10)
        
        ax.legend(loc='best', title="Run Number")
        plt.tight_layout()

        safe_filename = variable_name.replace(':', '_')
        output_filename = f"{self.plots_dir}/Overall_{safe_filename}_vs_Spill_Summary.png"
        plt.savefig(output_filename, dpi=200)
        plt.close(fig)




    def plot_acnet_grid_vs_spill(self, summary_df, variables_to_plot, plot_title, output_filename):
        """
        Creates a grid of subplots for a list of ACNET variables vs. spill number.
        Plots chronologically, not sequentially, to remove gaps between runs.
        """
        n_vars = len(variables_to_plot)
        if n_vars == 0: return

        fig, axes = plt.subplots(n_vars, 1, figsize=(15, 4 * n_vars), sharex=True)
        fig.suptitle(plot_title, fontsize=18)
        if n_vars == 1: axes = [axes]

        # REVISED: Sort by spill and reset the index once for all variables
        df_sorted = summary_df.sort_values(by='spill').reset_index(drop=True)

        all_runs = df_sorted['run'].unique()
        colormap = plt.get_cmap('hsv', len(all_runs))
        run_color_map = {run: colormap(i) for i, run in enumerate(all_runs)}

        for i, var_name in enumerate(variables_to_plot):
            ax = axes[i]
            
            df_var = df_sorted.copy()
            if var_name not in df_var.columns:
                ax.text(0.5, 0.5, f"Data for '{var_name}' not found", ha='center', va='center')
                continue
            
            df_var[var_name] = pd.to_numeric(df_var[var_name], errors='coerce')
            if var_name.startswith('E:M3T'):
                df_var.loc[df_var[var_name] < -9000, var_name] = np.nan
            df_var.dropna(subset=[var_name], inplace=True)

            if df_var.empty:
                ax.text(0.5, 0.5, f"No valid data for '{var_name}'", ha='center', va='center')
                continue

            for run_number, run_data in df_var.groupby('run'):
                # REVISED: Plot against the chronological index
                ax.plot(run_data.index, run_data[var_name], 'o', ms=2, 
                        color=run_color_map[run_number], label=f'Run {run_number}')

            metadata = ACNET_METADATA.get(var_name, {'title': var_name, 'unit': 'Unknown'})
            yaxis_label = f"{metadata['title'].split(' vs.')[0]} ({metadata['unit']})"
            ax.set_ylabel(yaxis_label)
            ax.grid(True, linestyle='--')
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        # REVISED: Add custom ticks and label to the LAST subplot
        axes[-1].set_xlabel("Spill Event (Chronological)")
        tick_spacing = max(1, len(df_sorted) // 15)
        tick_locs = np.arange(0, len(df_sorted), tick_spacing)
        tick_labels = df_sorted['spill'].iloc[tick_locs]
        axes[-1].set_xticks(tick_locs)
        axes[-1].set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=10)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', title="Run Number", bbox_to_anchor=(0.98, 0.95))

        plt.tight_layout(rect=[0, 0, 0.9, 0.96])
        plt.savefig(f"{self.plots_dir}/{output_filename}", dpi=150)
        plt.close(fig)
        print(f"  - Successfully created grid plot: {output_filename}")

    def plot_acnet_ratio_vs_spill(self, summary_df, numerator_var, denominator_var, yaxis_label):
        """
        Calculates and plots the ratio of two ACNET variables vs. spill number.
        Plots chronologically, not sequentially, to remove gaps between runs.
        """
        required_cols = [numerator_var, denominator_var, 'spill', 'run']
        if not all(col in summary_df.columns for col in required_cols):
            return

        df = summary_df[required_cols].copy()
        df[numerator_var] = pd.to_numeric(df[numerator_var], errors='coerce')
        df[denominator_var] = pd.to_numeric(df[denominator_var], errors='coerce')
        df[denominator_var] = df[denominator_var].replace(0, np.nan)
        df['ratio'] = df[numerator_var] / df[denominator_var]
        
        # REVISED: Sort by spill and then RESET THE INDEX
        df_sorted = df.sort_values(by='spill').dropna(subset=['ratio']).reset_index(drop=True)

        if df_sorted.empty:
            return

        fig, ax = plt.subplots(figsize=(15, 8))
        
        unique_runs = df_sorted['run'].unique()
        colormap = plt.get_cmap('hsv', len(unique_runs))
        for i, run_number in enumerate(unique_runs):
            run_data = df_sorted[df_sorted['run'] == run_number]
            # REVISED: Plot against the chronological index
            ax.plot(run_data.index, run_data['ratio'], 'o', ms=3, 
                    alpha=0.7, color=colormap(i), label=f'Run {run_number}')

        ax.set_title(f"Ratio of {numerator_var} / {denominator_var} vs. Spill Number")
        ax.set_ylabel(yaxis_label)
        ax.grid(True, linestyle='--')
        ax.legend(loc='best', title="Run Number")

        # REVISED: Add chronological label and custom ticks
        ax.set_xlabel("Spill Event (Chronological)")
        tick_spacing = max(1, len(df_sorted) // 15)
        tick_locs = np.arange(0, len(df_sorted), tick_spacing)
        tick_labels = df_sorted['spill'].iloc[tick_locs]
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=10)
        
        plt.tight_layout()
        
        safe_num = numerator_var.replace(':', '_')
        safe_den = denominator_var.replace(':', '_')
        output_filename = f"{self.plots_dir}/Ratio_{safe_num}_vs_{safe_den}_vs_Spill.png"
        plt.savefig(output_filename, dpi=150)
        plt.close(fig)
        print(f"  - Successfully created ratio vs. spill plot: {output_filename}")

    def plot_cross_correlation(self, summary_df, var1, var2, max_lag=100):
        """
        Calculates and plots the cross-correlation between two variables.
        The input data is ordered chronologically by spill number.
        """
        # --- 1. Validate and Prepare Data ---
        required_cols = [var1, var2, 'spill']
        if not all(col in summary_df.columns for col in required_cols):
            print(f"  - WARNING: Skipping cross-correlation. Missing one or more columns: {required_cols}")
            return

        # Sort chronologically and prepare a clean DataFrame
        df = summary_df.sort_values(by='spill').reset_index(drop=True)
        df[var1] = pd.to_numeric(df[var1], errors='coerce')
        df[var2] = pd.to_numeric(df[var2], errors='coerce')
        df.dropna(subset=[var1, var2], inplace=True)

        if df.empty or len(df) < 2:
            print(f"  - WARNING: Not enough valid data to correlate {var1} and {var2}.")
            return

        # --- 2. Calculate Cross-Correlation ---
        # Normalize the series by subtracting the mean
        series1 = (df[var1] - df[var1].mean()).to_numpy()
        series2 = (df[var2] - df[var2].mean()).to_numpy()
        
        # Calculate the correlation
        correlation = np.correlate(series1, series2, mode='full')
        
        # Normalize the result to be between -1 and 1
        norm_factor = np.sqrt(np.sum(series1**2) * np.sum(series2**2))
        normalized_correlation = correlation / norm_factor

        # Create the lag axis
        n = len(series1)
        lags = np.arange(-n + 1, n)

        # --- 3. Create and Format Plot ---
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(lags, normalized_correlation)
        
        # --- 4. Annotate the Plot ---
        # Find the lag with the highest correlation
        peak_corr_idx = np.argmax(normalized_correlation)
        peak_lag = lags[peak_corr_idx]
        peak_corr_val = normalized_correlation[peak_corr_idx]

        # Add a vertical line at the peak
        ax.axvline(peak_lag, color='red', linestyle='--', label=f'Peak Correlation at Lag {peak_lag}')
        
        # Add text to show the peak value
        ax.text(peak_lag + (max_lag * 0.05), peak_corr_val, 
                f'{peak_corr_val:.2f} at lag={peak_lag}', 
                ha='left', va='center', color='red',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # --- 5. Formatting and Saving ---
        ax.set_title(f"Cross-Correlation of {var1} and {var2}")
        ax.set_xlabel("Lag (in number of spills)")
        ax.set_ylabel("Normalized Cross-Correlation")
        ax.grid(True, linestyle='--')
        ax.set_xlim(-max_lag, max_lag) # Limit view to a reasonable lag
        ax.axhline(0, color='black', linewidth=0.5) # Add a zero line
        ax.legend()
        
        plt.tight_layout()
        safe_var1 = var1.replace(':', '_')
        safe_var2 = var2.replace(':', '_')
        output_filename = f"{self.plots_dir}/XCorr_{safe_var1}_vs_{safe_var2}.png"
        plt.savefig(output_filename, dpi=150)
        plt.close(fig)
        print(f"  - Successfully created cross-correlation plot: {output_filename}")



