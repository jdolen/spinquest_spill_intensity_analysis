# plotting.py
"""
spill_timing_analyzer plotting 
  Contains the SpillPlotter class for creating all visualizations
  related to the spill timing analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.signal import find_peaks

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

    def _add_run_spill_text(self, fig, run_num, spill_num, total_intensity=None):
        """Adds a standardized text box with run and spill info to a figure."""
        text = f"Run: {run_num}\nSpill: {spill_num}"
        if total_intensity is not None:
            text += f"\nTotal Intensity: {total_intensity:,.0f}"
        
        fig.text(0.99, 0.99, text, transform=fig.transFigure,
                 horizontalalignment='right', verticalalignment='top',
                 fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

    def _plot_intensity_profile(self, ax, analyzer, is_representative_spill, is_zoomed, highlight_peaks):
        """Helper function to perform spike analysis and plot the intensity profile."""
        intensity, times = analyzer.data['intensity'], analyzer.data['time_s']
        peak_indices, _ = find_peaks(intensity, height=intensity.mean() + 1.2 * intensity.std(), distance=2)

        if is_representative_spill:
            print(f"  - Diagnostic for Spill {analyzer.spill_num}: Found {len(peak_indices)} significant peaks.")

        if highlight_peaks:

            ax.plot(times, intensity, marker='.', markersize=1, linestyle='', color='black', label='All Data')

            freq_to_highlight = 73.75
            period = 1.0 / freq_to_highlight
            tolerance = 0.10
            
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
                print(f"    - Found {len(connected_pairs)} pairs matching ~{freq_to_highlight:.1f} Hz pattern.")

            
            if connected_pairs:
                for idx1, idx2 in connected_pairs:
                    ax.plot([times.iloc[idx1], times.iloc[idx2]], [intensity.iloc[idx1], intensity.iloc[idx2]],
                            color='orange', linestyle='-', linewidth=0.8, alpha=0.6)

            if highlight_peaks and highlighted_indices:
                ax.plot(times.iloc[list(highlighted_indices)], intensity.iloc[list(highlighted_indices)], 
                        'o', ms=6, mec='orangered', mfc='none', label=f'~{freq_to_highlight:.1f} Hz Spikes')
                ax.legend()
        else:
            ax.plot(times, intensity, linestyle='-', linewidth=0.5, color='black', label='All Data')


        title_suffix = " (Zoomed 1.0-1.4s)" if is_zoomed else ""
        ax.set_title(f"Full Spill Intensity (10kHz) for Run {analyzer.run_num}, Spill {analyzer.spill_num}{title_suffix}")
        ax.set_xlabel("Time (s)"), ax.set_ylabel("Intensity")
        ax.grid(True, linestyle='--')
        if is_zoomed:
            ax.set_xlim(1.0, 1.4)
        else:
            ax.set_xlim(left=0)

    def plot_single_spill(self, analyzer, is_representative_spill=False):
        """Intensity vs time for a single spill"""    
        if analyzer.data.empty: return
        fig, ax = plt.subplots(figsize=(15, 7))
        self._plot_intensity_profile(ax, analyzer, is_representative_spill, is_zoomed=False, highlight_peaks=False)
        self._add_run_spill_text(fig, analyzer.run_num, analyzer.spill_num, analyzer.total_intensity)
        output_filename = f"{self.plots_dir}/Intensity_vs_Time_run{analyzer.run_num}_spill{analyzer.spill_num}.png"
        plt.savefig(output_filename, dpi=300), plt.close(fig)
        print(f"Intensity plot for most intense spill of run {analyzer.run_num} saved.")

    def plot_single_spill_zoomed(self, analyzer):
        """Intensity vs time for a single spill but zoomed in to a abitrary 0.4 second window from 1-1.4s"""    
        if analyzer.data.empty: return
        fig, ax = plt.subplots(figsize=(15, 7))
        self._plot_intensity_profile(ax, analyzer, is_representative_spill=False, is_zoomed=True,highlight_peaks=True)
        self._add_run_spill_text(fig, analyzer.run_num, analyzer.spill_num, analyzer.total_intensity)
        output_filename = f"{self.plots_dir}/Intensity_vs_Time_Zoomed_run{analyzer.run_num}_spill{analyzer.spill_num}.png"
        plt.savefig(output_filename, dpi=200), plt.close(fig)
        print(f"Zoomed intensity plot for most intense spill of run {analyzer.run_num} saved.")

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
        print(f"\nSpill count per run plot saved to: {output_filename}")

    def plot_max_intensity_vs_spill(self, run_number, run_data):
        """maximum intensity bin in each spill vs spill number in a given run"""    
        if run_data.empty or run_data['max_intensity'].isnull().all(): return
        df = run_data.sort_values(by='spill').dropna(subset=['max_intensity'])
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(df['spill'], df['max_intensity'], marker='o', linestyle='--')
        ax.set_title(f"Peak Intensity vs. Spill Number for Run {run_number}")
        ax.set_xlabel("Spill Number"), ax.set_ylabel("Peak 100µs Integrated Intensity")
        ax.grid(True, linestyle='--')
        ax.get_xaxis().get_major_formatter().set_useOffset(False), ax.get_xaxis().get_major_formatter().set_scientific(False)
        output_filename = f"{self.plots_dir}/Max_Intensity_vs_Spill_run_{run_number}_Summary.png"
        plt.savefig(output_filename, dpi=150), plt.close(fig)
        print(f"\nPeak Intensity plot for Run {run_number} saved.")

    def plot_total_intensity_vs_spill(self, run_number, run_data):
        """total intensity in each spill vs spill number in a given run"""    
        if run_data.empty or run_data['total_intensity'].isnull().all(): return
        df = run_data.sort_values(by='spill').dropna(subset=['total_intensity'])
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(df['spill'], df['total_intensity'], marker='o', linestyle='--')
        ax.set_title(f"Total Integrated Intensity vs. Spill Number for Run {run_number}")
        ax.set_xlabel("Spill Number"), ax.set_ylabel("Total Integrated Intensity per Spill")
        ax.grid(True, linestyle='--')
        ax.get_xaxis().get_major_formatter().set_useOffset(False), ax.get_xaxis().get_major_formatter().set_scientific(False)
        output_filename = f"{self.plots_dir}/Total_Intensity_vs_Spill_run_{run_number}_Summary.png"
        plt.savefig(output_filename, dpi=150), plt.close(fig)
        print(f"Total Intensity plot for Run {run_number} saved.")
        
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
        print(f"Duty Factor plot for Run {run_number} saved.")

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
        print(f"Uniformity metrics plot for Run {run_number} saved.")
        
    def plot_peak_interval_histogram(self, run_number, run_data, weighted=False):
        """Plot time between consecutive spikes"""    
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
        print(f"Peak interval histogram (weighted={weighted}) for Run {run_number} saved.")

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
        ax.plot(fft_frequencies, fft_magnitudes)
        ax.set_title('Fast Fourier Transform (FFT)'), ax.set_xlabel('Frequency (Hz)'), ax.set_ylabel('Magnitude')
        ax.set_xlim(0, 500), ax.grid(True)
        peaks, properties = find_peaks(fft_magnitudes, height=np.mean(fft_magnitudes), distance=50)
        if len(peaks) > 0:
            top_indices = np.argsort(properties['peak_heights'])[-5:]
            for peak_index in peaks[top_indices]:
                peak_freq, peak_mag = fft_frequencies[peak_index], fft_magnitudes[peak_index]
                ax.axvline(x=peak_freq, color='r', linestyle='--', alpha=0.7)
                ax.text(peak_freq + 10, peak_mag, f'{peak_freq:.2f} Hz', color='r')
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        output_filename = f"{self.plots_dir}/FFT_run_{run_number}_spill_{spill_analyzer.spill_num}.png"
        plt.savefig(output_filename, dpi=150), plt.close(fig)
        print(f"FFT analysis plot for spill {spill_analyzer.spill_num} saved.")

    def plot_autocorrelation(self, run_number, spill_analyzer):
        """Autocorrelation for a given spill"""    
        if spill_analyzer.data.empty: return
        intensity = spill_analyzer.data['intensity'].values
        autocorr = np.correlate(intensity - intensity.mean(), intensity - intensity.mean(), mode='full')
        autocorr = autocorr[autocorr.size // 2:]
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
        print(f"Autocorrelation plot for spill {spill_analyzer.spill_num} saved.")

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
        print(f"Time of Max plot for Run {run_number} saved.")

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
        print(f"Time of Max histogram (weighted={weighted}) for Run {run_number} saved.")

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
        ax.set_title("Total Integrated Intensity vs. Spill Number")
        ax.set_xlabel("Spill Number (chronological, not sequential)")
        ax.set_ylabel("Total Integrated Intensity per Spill"), ax.grid(True, linestyle='--'), ax.set_xlim(left=-1, right=len(df_sorted))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=10, title="Run Number", fontsize='small')
        tick_spacing = 40
        tick_locs = range(0, len(df_sorted), tick_spacing)
        tick_labels = df_sorted['spill'].iloc[tick_locs]
        ax.set_xticks(tick_locs), ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.tick_params(axis='x', which='major', labelsize=8)
        plt.subplots_adjust(bottom=0.3)
        output_filename = f"{self.plots_dir}/Overall_Total_Intensity_vs_Spill_Summary.png"
        plt.savefig(output_filename, dpi=200), plt.close(fig)
        print(f"\nOverall Total Intensity plot saved to: {output_filename}")

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
        ax.set_title("Peak 100µs Integrated Intensity vs. Spill Number")
        ax.set_xlabel("Spill Number (chronological, not sequential)")
        ax.set_ylabel("Peak 100µs Integrated Intensity"), ax.grid(True, linestyle='--'), ax.set_xlim(left=-1, right=len(df_sorted))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=10, title="Run Number", fontsize='small')
        tick_spacing = 40
        tick_locs = range(0, len(df_sorted), tick_spacing)
        tick_labels = df_sorted['spill'].iloc[tick_locs]
        ax.set_xticks(tick_locs), ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.tick_params(axis='x', which='major', labelsize=8)
        plt.subplots_adjust(bottom=0.3)
        output_filename = f"{self.plots_dir}/Overall_Max_Intensity_vs_Spill_Summary.png"
        plt.savefig(output_filename, dpi=200), plt.close(fig)
        print(f"\nOverall Peak Intensity plot saved to: {output_filename}")

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
        print(f"\nOverall Duty Factor plot saved to: {output_filename}")
        
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
        print(f"\nOverall Time of Max plot saved to: {output_filename}")

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
        print(f"\nOverall Time of Max histogram (weighted={weighted}) saved.")