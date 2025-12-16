# plotting.py
"""
turn_structure_analyzer plotting script

Contains the TurnStructurePlotter class for creating all visualizations
from the results of a SpillAnalyzer object.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, ListedColormap

class TurnStructurePlotter:
    """Handles all plotting for an analyzed spill."""

    def __init__(self, analyzer, config):
        """
        Initializes the plotter with analysis results and configuration.

        Args:
            analyzer (SpillAnalyzer): The completed analyzer object.
            config (dict): A dictionary of configuration parameters.
        """
        self.analyzer = analyzer
        self.config = config
        self.results = analyzer.results
        self.run_num = analyzer.run_num
        self.spill_num = analyzer.spill_num
        self.plots_dir = self.config['PLOTS_DIR']
        self.colors = self.config['PLOT_COLORS']

    def plot_all(self):
        """A convenience method to generate and save all standard plots."""
        print("  Generating all plots...")
        self.plot_start_finder_debug()
        self.plot_start_finder_refinement()
        self.plot_start_finder_refinement_fom() 
        self.plot_first_two_turns_detailed()
        self.plot_initial_buckets(num_buckets=1000)
        self.plot_initial_buckets(num_buckets=2000)
        self.plot_selected_turns()
        self.plot_duty_factor_trends()
        self._plot_batch_df_vs_turn(self.config['NUM_BATCHES_PER_TURN'])
        self._plot_batch_df_vs_turn(4)
        self._plot_batch_df_vs_turn(2)
        self.plot_turn_intensity_stats()
        self.plot_turn_summary_subplots()
        self._plot_turn_trends_dual_axis(plot_df=True, plot_fom=False)
        self._plot_turn_trends_dual_axis(plot_df=True, plot_fom=True)
        self.plot_max_intensity_by_batch_color()
        self.plot_distribution_histograms()
        self._plot_distribution_boxplot(show_outliers=False)
        self._plot_distribution_boxplot(show_outliers=True)
        self.plot_distribution_violinplots()
        self.plot_most_intense_batch_by_sum_stats()
        self.plot_which_batch_had_max_bucket_stats()
        self.plot_batch_integrated_intensity_vs_turn()
        self.plot_turn_integrated_intensity_vs_turn()
        self.plot_turn_and_batch_integrated_intensity()
        self.plot_batch_integrated_intensity_stacked()
        self.plot_intensity_heatmaps()
        self.plot_df_heatmap()
        self.plot_intensity_vs_bucket_2d_hist()
        self.plot_distribution_stacked_histogram()
        self.plot_integrated_vs_peak_intensity()
        self.plot_zoomed_intensity_peaks()


        print("  Plotting complete.")

    # --------------------------------------------------------------------------
    # Plotting Helper Methods
    # --------------------------------------------------------------------------

    def _generate_filename(self, base_name):
        """Generates a standard filename for a plot."""
        return f"{self.plots_dir}/{base_name}_run{self.run_num}_spill{self.spill_num}.png"

    def _add_run_spill_text(self, fig):
        """Adds a standardized text box with run and spill info to a figure."""
        text = f"Run: {self.run_num}\nSpill: {self.spill_num}"
        fig.text(0.99, 0.99, text, transform=fig.transFigure,
                 horizontalalignment='right', verticalalignment='top',
                 fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    # --------------------------------------------------------------------------
    # Individual Plot Generation Methods
    # --------------------------------------------------------------------------
    
    def plot_first_two_turns_detailed(self):
        """Generates a detailed plot of the first two turns with batch annotations."""
        print("  Generating detailed plot for first two turns...")
        
        # Get config params
        start_bucket = self.config['start_bucket']
        buckets_per_batch = self.config['BUCKETS_PER_BATCH']
        num_batches = self.config['NUM_BATCHES_PER_TURN']
        turn_len = buckets_per_batch * num_batches
        
        # Check if we have enough data
        if len(self.analyzer.results['raw_intensity_series']) < start_bucket + 2 * turn_len:
            print("  - Not enough data to plot first two turns. Skipping.")
            return

        # Slice the data for the first two turns
        data_slice = self.analyzer.results['raw_intensity_series'].iloc[start_bucket : start_bucket + 2 * turn_len]
        
        fig, ax = plt.subplots(figsize=(20, 10))

        # Plot the intensity data
        ax.plot(data_slice.index, data_slice.values, '.', markersize=3, color='gray', alpha=0.7)

        # Find a good height for the annotations, above the max data point
        max_intensity = data_slice.max()
        if max_intensity <= 0: max_intensity = 1.0 # Avoid issues with empty data
        
        y_level_turn_label = max_intensity * 1.45
        y_level_batch_1 = max_intensity * 1.28
        y_level_batch_2 = max_intensity * 1.1

        # --- Annotate Turn 1 ---
        turn1_start = start_bucket
        turn1_end = start_bucket + turn_len
        ax.text((turn1_start + turn1_end) / 2, y_level_turn_label, 'Turn 1', ha='center', fontsize=14, fontweight='bold')
        
        for i in range(num_batches):
            batch_start = turn1_start + i * buckets_per_batch
            batch_end = batch_start + buckets_per_batch
            color = self.colors[i % len(self.colors)]
            
            ax.annotate(f'Batch {i+1}', 
                        xy=(batch_start, y_level_batch_1), 
                        xytext=(batch_end, y_level_batch_1),
                        arrowprops=dict(arrowstyle='<->', color=color, lw=2),
                        ha='center', va='bottom', color=color, fontsize=10, fontweight='bold')

        # --- Annotate Turn 2 ---
        turn2_start = start_bucket + turn_len
        turn2_end = start_bucket + 2 * turn_len
        ax.text((turn2_start + turn2_end) / 2, y_level_turn_label, 'Turn 2', ha='center', fontsize=14, fontweight='bold')
        
        for i in range(num_batches):
            batch_start = turn2_start + i * buckets_per_batch
            batch_end = batch_start + buckets_per_batch
            color = self.colors[i % len(self.colors)]
            
            ax.annotate(f'Batch {i+1}', 
                        xy=(batch_start, y_level_batch_2), 
                        xytext=(batch_end, y_level_batch_2),
                        arrowprops=dict(arrowstyle='<->', color=color, lw=2),
                        ha='center', va='bottom', color=color, fontsize=10, fontweight='bold')

        # Final plot styling
        ax.set_title("Intensity Profile of First Two Turns")
        ax.set_xlabel("RF Bucket Index")
        ax.set_ylabel("Per-Bucket Intensity")
        ax.grid(True, linestyle='--')
        ax.set_xlim(start_bucket, start_bucket + 2 * turn_len)
        ax.set_ylim(bottom=0, top=max_intensity * 1.6)
        
        # Add run/spill text box
        self._add_run_spill_text(fig)
        
        filename = self._generate_filename("First_Two_Turns_Detail")
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close(fig)
        print(f"  - Detailed first two turns plot saved.")

    def plot_initial_buckets(self, num_buckets=2000):
        """Plots the intensity of the first N buckets to find the start."""
        series = self.results['raw_intensity_series']
        if series.empty or len(series) < num_buckets:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        subset = series.iloc[:num_buckets]
        ax.plot(subset.index, subset.values, marker='.', markersize=2, linestyle='')
        ax.set_title(f"Intensity vs. Bucket Index (First {num_buckets} Buckets)")
        ax.set_xlabel("Relative Time Index (ns)")
        ax.set_ylabel("Intensity")
        ax.grid(True, linestyle='--', alpha=0.6)
        self._add_run_spill_text(fig)
        
        filename = self._generate_filename(f"Initial_{num_buckets}_Buckets")
        plt.savefig(filename, dpi=150)
        plt.close(fig)

    def plot_selected_turns(self):
        """Plots individual turns that meet certain criteria."""
        turn_df = self.results['turn_df']
        overall_max = turn_df['max'].max()
        
        for turn_idx, row in turn_df.iterrows():
            is_first_n = turn_idx < self.config['PLOT_FIRST_N_TURNS']
            is_intense = row['max'] > overall_max * self.config['PLOT_INTENSITY_THRESHOLD_FRAC']
            is_max_bucket_in_batch7 = row['batch_with_max_bucket'] == 7

            if is_first_n or is_intense or is_max_bucket_in_batch7:
                self._plot_single_turn(turn_idx, show_df=False)
                self._plot_single_turn(turn_idx, show_df=True)
    
    def _plot_single_turn(self, turn_idx, show_df=False):
        """
        CONSOLIDATED function to plot a single turn.
        Can plot a simple colored-batch view or a detailed view with Duty Factor.
        """
        num_batches = self.config['NUM_BATCHES_PER_TURN']
        buckets_per_batch = self.config['BUCKETS_PER_BATCH']
        turn_len = num_batches * buckets_per_batch
        
        start = self.config['start_bucket'] + turn_idx * turn_len
        end = start + turn_len
        
        turn_series = self.analyzer.intensity_series.iloc[start:end].copy()
        turn_series.index = np.arange(turn_len)
        
        fig, ax1 = plt.subplots(figsize=(14, 8) if show_df else (12, 6))

        if show_df:
            # --- Detailed View with Duty Factor ---
            ax2 = ax1.twinx()
            turn_metrics = self.results['turn_df'].loc[turn_idx]
            
            # Plot intensity
            ax1.plot(turn_series.index, turn_series.values, ".", color='black', markersize=3, alpha=0.8, label='Intensity/Bucket')
            
            # Plot duty factors
            handles = []
            for i in range(num_batches):
                df = turn_metrics[f'df_batch_{i+1}']
                if pd.notna(df):
                    xmin, xmax = i * buckets_per_batch, (i + 1) * buckets_per_batch
                    line = ax2.hlines(y=df, xmin=xmin, xmax=xmax, color=self.colors[i % len(self.colors)],
                                      linestyle='-', linewidth=3, label=f"Batch {i+1} DF = {df:.3f}")
                    handles.append(line)
            
            df_b14 = turn_metrics['df_b14']
            if pd.notna(df_b14):
                line, = ax2.plot([0, 4 * buckets_per_batch], [df_b14]*2, color='lime', linestyle=':', linewidth=3, label=f'DF B1-4 = {df_b14:.3f}')
                handles.append(line)

            ax2.set_ylabel("Duty Factor (<I>² / <I²>)", color='tab:blue')
            ax2.set_ylim(bottom=-0.05, top=1.1)
            ax1.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1.0), title="Legend")
            fig.tight_layout(rect=[0.1, 0.1, 0.8, 0.9])
            title = f"Intensity per Bucket and Duty Factor per Batch and Turn - Run: {self.run_num}, Spill: {self.spill_num}, Turn: {turn_idx}"
            #filename_base = f"IntensityAndDF_Turn_{turn_idx}"
            filename = f"{self.plots_dir}/IntensityAndDF_run{self.run_num}_spill{self.spill_num}_turn{turn_idx}.png"

        else:
            # --- Simple Colored Batch View ---
            for i in range(num_batches):
                batch_subset = turn_series.iloc[i*buckets_per_batch:(i+1)*buckets_per_batch]
                ax1.plot(batch_subset.index, batch_subset.values, ".", markersize=3, 
                        label=f"Batch {i+1}", color=self.colors[i % len(self.colors)])
            ax1.legend(fontsize='small')
            title = f"Intensity per Bucket - Run: {self.run_num}, Spill: {self.spill_num}, Turn: {turn_idx}"
            #filename_base = f"ColoredBatches_Turn_{turn_idx}"
            filename = f"{self.plots_dir}/ColoredBatches_run{self.run_num}_spill{self.spill_num}_turn{turn_idx}.png"

        # Common plot elements
        ax1.set_xlabel(f"RF Bucket within Turn (0 to {turn_len-1})")
        ax1.set_ylabel("Intensity")
        ax1.set_title(title)
        #ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

        # Draw vertical lines separating the batches
        for i in range(1, num_batches):
            line_pos = i * buckets_per_batch - 0.5
            ax1.axvline(line_pos, color='k', linestyle=':', linewidth=0.8, alpha=0.7)
            
        #self._add_run_spill_text(fig)
        #filename = self._generate_filename(filename_base)
        plt.savefig(filename, dpi=150)
        plt.close(fig)

    def plot_duty_factor_trends(self):
        """Plots various duty factors vs. turn number."""
        turn_df = self.results['turn_df']
        fig, ax = plt.subplots(figsize=(12, 7))

        ax.plot(turn_df.index, turn_df['df_overall'], '.', ms=4, color='black', label="Overall Turn DF")
        ax.plot(turn_df.index, turn_df['df_b12'], '.', ms=4, color='cyan', label="DF Batches 1-2")
        ax.plot(turn_df.index, turn_df['df_b14'], '.', ms=4, color='lime', label="DF Batches 1-4")
        
        ax.set_xlabel("Turn Number")
        ax.set_ylabel("Duty Factor (<I>² / <I²>)")
        ax.set_title("Duty Factors Integrated Over Turns")
        ax.grid(True, linestyle=':')
        ax.set_ylim(bottom=-0.05, top=1.05)
        ax.legend(fontsize='small', markerscale=2)
        fig.tight_layout()
        self._add_run_spill_text(fig)

        filename = self._generate_filename("DutyFactor_vs_Turn")
        plt.savefig(filename, dpi=150)
        plt.close(fig)

    def _plot_batch_df_vs_turn(self, num_batches_to_plot):
        """Generic helper to plot individual batch DFs vs. turn."""
        turn_df = self.results['turn_df']
        fig, ax = plt.subplots(figsize=(12, 7))

        for i in range(num_batches_to_plot):
            ax.plot(turn_df.index, turn_df[f'df_batch_{i+1}'], '.', ms=4,
                    color=self.colors[i % len(self.colors)], label=f"Batch {i+1} DF")
        
        ax.set_xlabel("Turn Number")
        ax.set_ylabel("Batch Duty Factor (<I>² / <I²>)")
        ax.set_title(f"Individual Batch Duty Factors vs. Turn (Batches 1-{num_batches_to_plot})")
        ax.grid(True, linestyle=':')
        ax.set_ylim(bottom=-0.05, top=1.05)
        ax.legend(fontsize='small', markerscale=2)
        fig.tight_layout()
        self._add_run_spill_text(fig)

        filename = self._generate_filename(f"BatchDF_vs_Turn_1_to_{num_batches_to_plot}")
        plt.savefig(filename, dpi=150)
        plt.close(fig)

    def plot_turn_intensity_stats(self):
        """Plots the mean, median, and max intensity for each turn."""
        turn_df = self.results['turn_df']
        fig, ax = plt.subplots(figsize=(12, 7))

        yerr = [turn_df['median'] - turn_df['q1'], turn_df['q3'] - turn_df['median']]
        ax.errorbar(turn_df.index, turn_df['median'], yerr=yerr, fmt='o', capsize=3, ms=3, color='red', ecolor='salmon', elinewidth=1, label='Median Intensity (Quartiles)')
        ax.plot(turn_df.index, turn_df['mean'], 'o-', ms=3, color='black', label='Mean Intensity')
        ax.plot(turn_df.index, turn_df['max'], '.', ms=3, color='purple', label='Peak Intensity')
        
        ax.set_xlabel("Turn Number")
        ax.set_ylabel("Intensity")
        ax.set_title("Per-Turn Intensity Statistics")
        ax.legend()
        ax.grid(True, linestyle=':')
        self._add_run_spill_text(fig)
        
        filename = self._generate_filename("Turn_Intensity_Stats")
        plt.savefig(filename, dpi=150)
        plt.close(fig)

    def _plot_turn_trends_dual_axis(self, plot_df=True, plot_fom=False):
        """CONSOLIDATED function to create dual-axis plots of intensity, DF, and FoM."""
        turn_df = self.results['turn_df'].dropna()
        if turn_df.empty:
            return
            
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax2 = ax1.twinx()

        # Left Axis: Intensity
        ax1.plot(turn_df.index, turn_df['max'], color='purple', ls=':', marker='.', ms=3, label='Peak Intensity')
        ax1.set_xlabel("Turn Number")
        ax1.set_ylabel("Intensity")

        # Right Axis: DF and/or FoM
        handles, labels = ax1.get_legend_handles_labels()
        title_parts = ["Peak Intensity"]
        filename_parts = ["Intensity"]

        if plot_df:
            line, = ax2.plot(turn_df.index, turn_df['df_b12'], color='cyan', ls='-', marker='.', ms=3, label='DF Batches 1-2')
            handles.append(line)
            labels.append(line.get_label())
            title_parts.append("Duty Factor")
            filename_parts.append("DF")

        if plot_fom:
            line, = ax2.plot(turn_df.index, turn_df['fom_b12'], color='red', ls='-', marker='.', ms=3, label='FoM Batches 1-2')
            handles.append(line)
            labels.append(line.get_label())
            title_parts.append("FoM")
            filename_parts.append("FoM")

        ax2.set_ylabel(" / ".join(title_parts[1:]), color='tab:blue') # Dynamic label
        ax2.set_ylim(bottom=0)
        
        # Combined Legend
        ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        
        ax1.grid(True, linestyle='--')
        plt.title(f"Per-Turn {', '.join(title_parts)}")
        fig.tight_layout(rect=[0, 0.1, 1, 0.95])
        self._add_run_spill_text(fig)
        
        filename = self._generate_filename(f"{'_'.join(filename_parts)}_vs_Turn")
        plt.savefig(filename, dpi=150)
        plt.close(fig)
        
    def plot_max_intensity_by_batch_color(self):
        """Plots max turn intensity vs. turn, with points colored by the batch containing the peak bucket."""
        turn_df = self.results['turn_df']
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot points for each batch that was the max
        for i in range(1, self.config['NUM_BATCHES_PER_TURN'] + 1):
            subset = turn_df[turn_df['batch_with_max_bucket'] == i]
            if not subset.empty:
                ax.plot(subset.index, subset['max'], ".", markersize=4, 
                        color=self.colors[(i-1) % len(self.colors)], label=f"Max from Batch {i}")

        # Overlay Mean and Median
        yerr = [turn_df['median'] - turn_df['q1'], turn_df['q3'] - turn_df['median']]
        ax.errorbar(turn_df.index, turn_df['median'], yerr=yerr, fmt='o', capsize=3, ms=2, color='darkred', ecolor='lightcoral', elinewidth=1, label='Median Intensity (Quartiles)', zorder=10)
        ax.plot(turn_df.index, turn_df['mean'], 'o', ms=2, color='black', label='Mean Intensity', zorder=11)

        ax.set_xlabel("Turn Number")
        ax.set_ylabel("Intensity")
        ax.set_title("Peak Turn Intensity (Colored by Batch of Origin) with Mean/Median")
        ax.legend(loc='best', fontsize='small')
        ax.grid(True, linestyle=':')
        self._add_run_spill_text(fig)
        
        filename = self._generate_filename("Max_Intensity_by_Batch_Color")
        plt.savefig(filename, dpi=150)
        plt.close(fig)


    def plot_distribution_histograms(self):
        """Plots histograms of intensity distributions for each batch."""
        all_batch_data = self.results['aggregate_stats']['all_batch_data']
        max_intensity = self.results['raw_intensity_series'].max()
        
        fig, ax = plt.subplots(figsize=(12, 7))

        for i, (batch_label, data) in enumerate(all_batch_data.items()):
            if data:
                ax.hist(data, bins=100, range=(0, max_intensity),
                        label=batch_label, color=self.colors[i % len(self.colors)],
                        histtype='step', linewidth=2)
        
        ax.set_title("Intensity Distribution per Batch (Log Scale)")
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Count")
        ax.set_yscale('log')
        ax.legend()
        self._add_run_spill_text(fig)
        
        filename = self._generate_filename("Batch_Intensity_Hist_Log")
        plt.savefig(filename, dpi=150)
        plt.close(fig)

    def _plot_distribution_boxplot(self, show_outliers=False):
        """CONSOLIDATED function to plot boxplots with or without outliers."""
        all_batch_data = self.results['aggregate_stats']['all_batch_data']
        labels = list(all_batch_data.keys())
        data = list(all_batch_data.values())

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.boxplot(data, tick_labels=labels, showfliers=show_outliers)
        
        title_suffix = "(With Outliers)" if show_outliers else "(Outliers Excluded)"
        filename_suffix = "WithOutliers" if show_outliers else "NoFliers"
        
        ax.set_title(f"Intensity Distributions per Batch {title_suffix}")
        ax.set_ylabel("Intensity")
        self._add_run_spill_text(fig)
        
        filename = self._generate_filename(f"Batch_Boxplot_{filename_suffix}")
        plt.savefig(filename, dpi=150)
        plt.close(fig)

    def plot_distribution_violinplots(self):
        """Plots violin plots of intensity distributions for each batch."""
        all_batch_data = self.results['aggregate_stats']['all_batch_data']
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in all_batch_data.items()]))
        
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.violinplot(data=df, palette=self.colors, cut=0, inner="quartile", ax=ax)
        ax.set_title("Violin Plot of Intensity Distributions")
        ax.set_ylabel("Intensity")
        self._add_run_spill_text(fig)
        
        filename = self._generate_filename("Batch_Violinplot")
        plt.savefig(filename, dpi=150)
        plt.close(fig)

    def plot_which_batch_had_max_bucket_stats(self):
        """Plots frequency of each batch containing the turn's single most intense bucket."""
        counts = self.results['turn_df']['batch_with_max_bucket'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 6))

        batch_numbers = counts.index.astype(int)
        bar_colors = [self.colors[(i-1) % len(self.colors)] for i in batch_numbers]
        bars = ax.bar(batch_numbers, counts.values, color=bar_colors)
        
        ax.set_xlabel("Batch Number")
        ax.set_ylabel("Number of Turns")
        ax.set_title("Frequency of Each Batch Containing the Peak Intensity Bucket")
        ax.set_xticks(range(1, self.config['NUM_BATCHES_PER_TURN'] + 1))
        ax.bar_label(bars)
        self._add_run_spill_text(fig)
        
        filename = self._generate_filename("Peak_Bucket_Location_by_Batch_Counts")
        plt.savefig(filename, dpi=150)
        plt.close(fig)

    def plot_intensity_heatmaps(self):
        """Plots a 2D heatmap of max intensity per batch vs. turn."""
        data = self.results['batch_max_intensity_heatmap_data']
        fig, ax = plt.subplots(figsize=(12, 7))
        
        cmap = ListedColormap(sns.color_palette("viridis", n_colors=256))
        cmap.set_bad(color='grey')
        
        im = ax.imshow(data, cmap=cmap, aspect='auto', origin='lower', interpolation='nearest',
                       norm=LogNorm(vmin=1, vmax=np.nanmax(data)))
        
        fig.colorbar(im, label="Max Intensity in Batch (Log Scale)")
        ax.set_xlabel("Turn Number")
        ax.set_ylabel("Batch Number")
        ax.set_yticks(range(self.config['NUM_BATCHES_PER_TURN']))
        ax.set_yticklabels([f"Batch {i+1}" for i in range(self.config['NUM_BATCHES_PER_TURN'])])
        ax.set_title("Max Intensity per Batch per Turn")
        self._add_run_spill_text(fig)
        
        filename = self._generate_filename("Max_Intensity_Heatmap")
        plt.savefig(filename, dpi=150)
        plt.close(fig)

    def plot_df_heatmap(self):
        """Plots a 2D heatmap of duty factor per batch vs. turn."""
        data = self.results['batch_df_heatmap_data']
        fig, ax = plt.subplots(figsize=(12, 7))

        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='grey')
        
        im = ax.imshow(data, cmap=cmap, aspect='auto', origin='lower', interpolation='nearest', vmin=0, vmax=1)
        
        fig.colorbar(im, label="Batch Duty Factor")
        ax.set_xlabel("Turn Number")
        ax.set_ylabel("Batch Number")
        ax.set_yticks(range(self.config['NUM_BATCHES_PER_TURN']))
        ax.set_yticklabels([f"Batch {i+1}" for i in range(self.config['NUM_BATCHES_PER_TURN'])])
        ax.set_title("Duty Factor per Batch per Turn")
        self._add_run_spill_text(fig)
        
        filename = self._generate_filename("Duty_Factor_Heatmap")
        plt.savefig(filename, dpi=150)
        plt.close(fig)

    def plot_intensity_vs_bucket_2d_hist(self):
        """Plots a 2D histogram of intensity vs. bucket index over all turns."""
        data = self.results['intensity_vs_bucket_data']
        if not data['indices']: return

        num_batches = self.config['NUM_BATCHES_PER_TURN']
        buckets_per_batch = self.config['BUCKETS_PER_BATCH']
        expected_turn_len = num_batches * buckets_per_batch
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        x_bins = np.arange(expected_turn_len + 1)
        y_bins = np.linspace(0, np.max(data['intensities']), 101)

        h, xedges, yedges = np.histogram2d(data['indices'], data['intensities'], bins=[x_bins, y_bins])
        
        pcm = ax.pcolormesh(xedges, yedges, h.T, cmap='viridis', norm=LogNorm(vmin=1), shading='auto')
        fig.colorbar(pcm, label='Number of Turns (Log Scale)')

        # Draw vertical lines separating the batches
        for i in range(1, num_batches):
            line_pos = i * buckets_per_batch - 0.5
            ax.axvline(line_pos, color='k', linestyle=':', linewidth=0.8, alpha=0.7)
            
        ax.set_xlabel("RF Bucket Number within Turn")
        ax.set_ylabel("Intensity")
        ax.set_title("Turn Count vs. Intensity and Bucket Index (All Turns)")
        self._add_run_spill_text(fig)
        
        filename = self._generate_filename("Intensity_vs_Bucket_2DHist")
        plt.savefig(filename, dpi=150)
        plt.close(fig)

    def plot_most_intense_batch_by_sum_stats(self):
        """Plots a bar chart showing how often each batch had the most integrated intensity."""
        if 'most_intense_batch_by_sum' not in self.results['turn_df'].columns:
            print("  - 'most_intense_batch_by_sum' data not found. Skipping plot.")
            return

        counts = self.results['turn_df']['most_intense_batch_by_sum'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 6))

        batch_numbers = counts.index.astype(int)
        bar_colors = [self.colors[(i-1) % len(self.colors)] for i in batch_numbers]
        bars = ax.bar(batch_numbers, counts.values, color=bar_colors)
        
        ax.set_xlabel("Batch Number")
        ax.set_ylabel("Number of Turns")
        ax.set_title("Frequency of Each Batch Having the Highest Integrated Intensity")
        ax.set_xticks(range(1, self.config['NUM_BATCHES_PER_TURN'] + 1))
        ax.bar_label(bars)
        self._add_run_spill_text(fig)
        
        filename = self._generate_filename("Most_Intense_Batch_by_Sum_Counts")
        plt.savefig(filename, dpi=150)
        plt.close(fig)

    def plot_batch_integrated_intensity_vs_turn(self):
        """Plots the integrated intensity for each batch vs. turn number."""
        turn_df = self.results['turn_df']
        fig, ax = plt.subplots(figsize=(12, 7))

        # Verify the necessary data columns exist before trying to plot
        if 'integrated_intensity_batch_1' not in turn_df.columns:
            print("  - Batch integrated intensity data not found. Skipping plot.")
            plt.close(fig)
            return

        num_batches = self.config['NUM_BATCHES_PER_TURN']
        for i in range(num_batches):
            batch_col = f'integrated_intensity_batch_{i+1}'
            ax.plot(turn_df.index, turn_df[batch_col], '.', ms=4,
                    color=self.colors[i % len(self.colors)], 
                    label=f"Batch {i+1} Integral")

        ax.set_xlabel("Turn Number")
        ax.set_ylabel("Batch Integrated Intensity")
        ax.set_title("Individual Batch Integrated Intensity vs. Turn")
        ax.grid(True, linestyle=':')
        ax.legend(fontsize='small', markerscale=2)
        fig.tight_layout()
        self._add_run_spill_text(fig)

        filename = self._generate_filename("Batch_Integrated_Intensity_vs_Turn")
        plt.savefig(filename, dpi=150)
        plt.close(fig)   



    def plot_turn_integrated_intensity_vs_turn(self):
        """Plots the total integrated intensity for each turn vs. turn number."""
        turn_df = self.results['turn_df']
        peaks_df = self.results.get('peaks_df')
        fig, ax = plt.subplots(figsize=(12, 7))

        # Check if the integrated_intensity column exists
        if 'integrated_intensity' not in turn_df.columns:
            print("  - Turn integrated intensity data not found. Skipping plot.")
            plt.close(fig)
            return

        ax.plot(turn_df.index, turn_df['integrated_intensity'], 'o', ms=2, color='black',
                label='Total Integrated Intensity')

        if peaks_df is not None and not peaks_df.empty:
            # Plot top 3 peaks
            for i, peak in peaks_df.head(3).iterrows():
                # Vertical line at peak center
                ax.axvline(x=peak['turn'], color='r', linestyle='--', alpha=0.9)
                # Horizontal line for peak width
                ax.hlines(y=peak['width_height'], xmin=peak['left_turn'], xmax=peak['right_turn'],
                          color='red', linewidth=2, linestyle='--')
                # Add a text label for the width
                ax.text(peak['turn'], peak['width_height'] * 1.05, f"  Width: {peak['width_turns']:.1f} turns",
                        color='r', ha='center')

        ax.set_xlabel("Turn Number")
        ax.set_ylabel("Total Integrated Intensity per Turn")
        ax.set_title("Total Integrated Intensity vs. Turn")
        ax.grid(True, linestyle='--')
        ax.legend()
        fig.tight_layout()
        self._add_run_spill_text(fig)

        filename = self._generate_filename("Turn_Integrated_Intensity_vs_Turn")
        plt.savefig(filename, dpi=150)
        plt.close(fig)  

    def plot_zoomed_intensity_peaks(self):
        """Creates a zoomed-in plot for each of the top 3 intensity peaks."""
        turn_df = self.results.get('turn_df')
        peaks_df = self.results.get('peaks_df')

        if turn_df is None or peaks_df is None or peaks_df.empty:
            return # Nothing to plot

        # Loop through the top 3 peaks and create a plot for each
        for rank, peak in peaks_df.head(3).iterrows():
            fig, ax = plt.subplots(figsize=(12, 7))

            # Plot the intensity data
            ax.plot(turn_df.index, turn_df['integrated_intensity'], marker='.', linestyle='-', markersize=4, color='black')

            # Zoom in on the spike
            window_size = peak['width_turns'] * 10 # Zoom window is 10x the peak width
            ax.set_xlim(peak['turn'] - (window_size / 2), peak['turn'] + (window_size / 2))

            # Add vertical line for peak center
            ax.axvline(x=peak['turn'], color='r', linestyle='--', alpha=0.8, label=f"Peak Turn: {peak['turn']}")

            # Add horizontal line for the peak width
            ax.hlines(y=peak['width_height'], xmin=peak['left_turn'], xmax=peak['right_turn'],
                      color='red', linewidth=2, label=f"Width: {peak['width_turns']:.2f} turns")

            ax.set_title(f"Zoomed View of Top Intensity Peak #{rank+1}")
            ax.set_xlabel("Turn Number")
            ax.set_ylabel("Total Integrated Intensity")
            ax.legend()
            ax.grid(True)
            self._add_run_spill_text(fig)

            output_filename = self._generate_filename(f"Zoomed_Intensity_Peak_Rank_{rank+1}")
            plt.savefig(output_filename, dpi=150)
            plt.close(fig)


    def plot_turn_and_batch_integrated_intensity(self):
        """Plots total turn and individual batch integrated intensity vs. turn number."""
        turn_df = self.results['turn_df']
        fig, ax = plt.subplots(figsize=(12, 8))

        # Check if all necessary data columns exist
        required_cols = ['integrated_intensity'] + [f'integrated_intensity_batch_{i+1}' for i in range(self.config['NUM_BATCHES_PER_TURN'])]
        if not all(col in turn_df.columns for col in required_cols):
            print("  - Missing integrated intensity data for combined plot. Skipping.")
            plt.close(fig)
            return

        # Plot the total integrated intensity for the whole turn first
        ax.plot(turn_df.index, turn_df['integrated_intensity'], '-', color='black', lw=2,
                label='Total Turn Integral', zorder=10) # zorder brings it to the front

        # Plot the integrated intensity for each batch
        num_batches = self.config['NUM_BATCHES_PER_TURN']
        for i in range(num_batches):
            batch_col = f'integrated_intensity_batch_{i+1}'
            ax.plot(turn_df.index, turn_df[batch_col], '.', ms=4,
                    color=self.colors[i % len(self.colors)],
                    label=f"Batch {i+1} Integral")

        ax.set_xlabel("Turn Number")
        ax.set_ylabel("Integrated Intensity")
        ax.set_title("Total and Per-Batch Integrated Intensity vs. Turn")
        ax.grid(True, linestyle=':')
        ax.legend(fontsize='small', markerscale=2)
        fig.tight_layout()
        self._add_run_spill_text(fig)

        filename = self._generate_filename("Turn_and_Batch_Integrated_Intensity_vs_Turn")
        plt.savefig(filename, dpi=150)
        plt.close(fig)



    def plot_distribution_stacked_histogram(self):
        """Plots a stacked histogram of intensity distributions for each batch."""
        all_batch_data = self.results['aggregate_stats']['all_batch_data']
        
        # Prepare data and labels for the hist function
        # The order matters for stacking, so we use a loop
        data_to_plot = []
        labels = []
        plot_colors = []
        num_batches = self.config['NUM_BATCHES_PER_TURN']

        for i in range(num_batches):
            batch_label = f"Batch {i+1}"
            batch_data = all_batch_data.get(batch_label, [])
            if batch_data:
                data_to_plot.append(batch_data)
                labels.append(batch_label)
                plot_colors.append(self.colors[i % len(self.colors)])

        if not data_to_plot:
            print("  - No data found for stacked histogram. Skipping plot.")
            return

        fig, ax = plt.subplots(figsize=(12, 7))

        # The 'stacked=True' parameter does all the work
        ax.hist(data_to_plot,
                bins=100,
                stacked=True,
                label=labels,
                color=plot_colors,
                histtype='bar') # 'bar' is the default fill type

        ax.set_title("Stacked Intensity Distribution per Batch (Log Scale)")
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Count")
        ax.set_yscale('log')
        ax.legend()
        self._add_run_spill_text(fig)
        fig.tight_layout()

        filename = self._generate_filename("Batch_Intensity_Hist_Stacked_Log")
        plt.savefig(filename, dpi=150)
        plt.close(fig)




    def plot_batch_integrated_intensity_stacked(self):
        """Plots a stacked area chart of integrated intensity for each batch vs. turn."""
        turn_df = self.results['turn_df']
        fig, ax = plt.subplots(figsize=(12, 7))

        num_batches = self.config['NUM_BATCHES_PER_TURN']
        batch_integral_cols = [f'integrated_intensity_batch_{i+1}' for i in range(num_batches)]

        if not all(col in turn_df.columns for col in batch_integral_cols):
            print("  - Missing batch integrated intensity data for stacked plot. Skipping.")
            plt.close(fig)
            return

        x = turn_df.index
        y = turn_df[batch_integral_cols].T.values

        # This now correctly generates labels "Batch 1" through "Batch 7"
        labels = [f"Batch {i+1}" for i in range(num_batches)]

        ax.stackplot(x, y, labels=labels, colors=self.colors[:num_batches], alpha=0.8)

        ax.set_xlabel("Turn Number")
        ax.set_ylabel("Total Integrated Intensity (Stacked)")
        ax.set_title("Stacked Integrated Intensity per Batch vs. Turn")
        ax.grid(True, linestyle='--')
        ax.legend(loc='upper left')
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        self._add_run_spill_text(fig)

        filename = self._generate_filename("Batch_Integrated_Intensity_Stacked_vs_Turn")
        plt.savefig(filename, dpi=150)
        plt.close(fig)



    def plot_integrated_vs_peak_intensity(self):
        """Plots total integrated intensity and peak bucket intensity vs. turn on dual axes."""
        turn_df = self.results['turn_df']
        if turn_df.empty:
            return

        fig, ax1 = plt.subplots(figsize=(12, 7))

        # Left Y-Axis: Integrated Intensity
        color1 = 'tab:blue'
        ax1.set_xlabel('Turn Number')
        ax1.set_ylabel('Total Integrated Intensity per Turn', color=color1)
        ax1.plot(turn_df.index, turn_df['integrated_intensity'], 'o', color=color1, ms=3, label='Integrated Intensity')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, linestyle='--')

        # Right Y-Axis: Peak Bucket Intensity
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Peak Bucket Intensity in Turn', color=color2)
        ax2.plot(turn_df.index, turn_df['max'], '.', color=color2, ms=3, label='Peak Bucket Intensity')
        ax2.tick_params(axis='y', labelcolor=color2)

        # Final Touches
        fig.tight_layout()
        ax1.set_title('Integrated vs. Peak Intensity per Turn')
        self._add_run_spill_text(fig)
        
        filename = self._generate_filename("Integrated_vs_Peak_Intensity_per_Turn")
        plt.savefig(filename, dpi=150)
        plt.close(fig)



    def plot_start_finder_debug(self):
        """Plots the first 800 buckets with noise/threshold lines for debugging."""
        start_finder_stats = self.results.get('start_finder_stats', {})
        noise_mean = start_finder_stats.get('noise_mean')
        threshold = start_finder_stats.get('threshold')

        if noise_mean is None or threshold is None:
            return

        fig, ax = plt.subplots(figsize=(14, 7))

        data_slice = self.analyzer.results['raw_intensity_series'].iloc[:800]
        x_axis_indices = np.arange(len(data_slice))
        
        ax.plot(x_axis_indices, data_slice.values, '.', ms=3, color='black', label='Intensity', zorder=10)

        ax.axhline(y=noise_mean, color='cyan', linestyle='--', label=f'Noise Level: {noise_mean:.2f}')
        ax.axhline(y=threshold, color='green', linestyle=':', lw=2, label=f'Threshold: {threshold:.2f}')

        start_bucket = self.config.get('start_bucket')
        if start_bucket is not None:
            ax.axvline(x=start_bucket, color='red', linestyle='-', lw=2, label=f'Found Start Bucket: {start_bucket}')

            buckets_per_batch = self.config.get('BUCKETS_PER_BATCH', 84)
            num_batches = self.config.get('NUM_BATCHES_PER_TURN', 7)
            
            ax.axvline(x=-1, color='blue', linestyle='--', lw=1, label='Batch Starts')
            
            for i in range(num_batches+3):
                batch_start_pos = start_bucket + (i * buckets_per_batch)
                if batch_start_pos < 800:
                    ax.axvline(x=batch_start_pos, color='blue', linestyle='--', lw=1)

        ax.set_title("Start Bucket Finder Debug Plot (First 800 Buckets)")
        ax.set_xlabel("Bucket Index (from start of file)")
        ax.set_ylabel("Intensity")
        ax.legend()
        
        # Only draw gridlines for the y-axis (horizontal lines)
        ax.grid(True, axis='y', linestyle=':')

        ax.set_xlim(0, 800)
        
        self._add_run_spill_text(fig)
        filename = self._generate_filename("Start_Finder_Debug")
        plt.savefig(filename, dpi=150)
        plt.close(fig)



    def plot_turn_summary_subplots(self):
        """
        Creates a 3-panel plot showing integrated intensity, peak intensity,
        and duty factor per turn, all sharing a common turn ID axis.
        """
        turn_df = self.results['turn_df']
        if turn_df.empty:
            return

        # Create 3 subplots that share the same x-axis
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # --- Top Plot: Integrated Intensity ---
        color1 = 'tab:blue'
        ax1.plot(turn_df.index, turn_df['integrated_intensity'], 'o', color=color1, ms=2)
        ax1.set_ylabel('Integrated Intensity')
        ax1.grid(True, linestyle='--')
        ax1.set_title("Per-Turn Beam Characteristics")

        # --- Middle Plot: Peak Bucket Intensity ---
        color2 = 'tab:red'
        ax2.plot(turn_df.index, turn_df['max'], '.', color=color2, ms=2)
        ax2.set_ylabel('Peak Bucket Intensity')
        ax2.grid(True, linestyle='--')

        # --- Bottom Plot: Overall Duty Factor ---
        ax3.plot(turn_df.index, turn_df['df_overall'], '.', color='green', ms=2)
        ax3.set_ylabel('Overall Duty Factor')
        ax3.set_xlabel('Turn Number')
        ax3.grid(True, linestyle='--')
        ax3.set_ylim(bottom=-0.05, top=1.05)

        # Add run/spill info and save
        self._add_run_spill_text(fig)
        fig.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout for main title
        
        filename = self._generate_filename("Turn_Summary_Subplots")
        plt.savefig(filename, dpi=150)
        plt.close(fig)

    def plot_start_finder_refinement(self):
        """
        Plots the results of the iterative start bucket refinement, showing
        Batch 7 intensity vs. the candidate start bucket.
        """
        debug_data = self.analyzer.results.get('refinement_debug_data')
        if not debug_data:
            return

        fig, ax = plt.subplots(figsize=(12, 7))

        search_range = debug_data['search_range']
        intensities = debug_data['batch7_intensities']
        best_start = debug_data['best_start']
        initial_guess = debug_data['initial_guess']

        ax.plot(search_range, intensities, '.-', label='Batch 7 Integrated Intensity')
        
        # Mark the initial guess and the final chosen start bucket
        ax.axvline(x=initial_guess, color='orange', linestyle='--', label=f'Initial Guess: {initial_guess}')
        ax.axvline(x=best_start, color='red', linestyle='-', label=f'Refined Start: {best_start}')
        
        ax.set_xlabel("Candidate Start Bucket Index")
        ax.set_ylabel(f"Integrated Intensity in Batch 7 (over {self.config.get('REFINEMENT_SAMPLE_TURNS')} turns)")
        ax.set_title("Start Bucket Refinement by Minimizing Batch 7 Intensity")
        ax.legend()
        ax.grid(True, linestyle=':')
        
        self._add_run_spill_text(fig)
        filename = self._generate_filename("Start_Finder_Refinement_Debug")
        plt.savefig(filename, dpi=150)
        plt.close(fig)

    def plot_start_finder_refinement_fom(self):
        """
        Plots the results of the iterative start bucket refinement, showing
        the Figure of Merit (Batch 1 / Batch 7 intensity) vs. the candidate start bucket.
        """
        debug_data = self.analyzer.results.get('refinement_debug_data')
        # Check if the new FOM data is available
        if not debug_data or 'fom_values' not in debug_data:
            return

        fig, ax = plt.subplots(figsize=(12, 7))

        search_range = debug_data['search_range']
        fom_values = debug_data['fom_values'] # Use FOM values
        best_start = debug_data['best_start']
        initial_guess = debug_data['initial_guess']

        ax.plot(search_range, fom_values, '.-', label='Refinement Figure of Merit (FOM)')
        
        # Mark the initial guess and the final chosen start bucket
        ax.axvline(x=initial_guess, color='orange', linestyle='--', label=f'Initial Guess: {initial_guess}')
        ax.axvline(x=best_start, color='red', linestyle='-', label=f'Refined Start (Max FOM): {best_start}')
        
        ax.set_xlabel("Candidate Start Bucket Index")
        ax.set_ylabel("Figure of Merit (Intensity B1 / Intensity B7)")
        ax.set_title("Start Bucket Refinement by Maximizing Figure of Merit")
        ax.legend()
        ax.grid(True, linestyle=':')
        
        self._add_run_spill_text(fig)
        filename = self._generate_filename("Start_Finder_Refinement_FOM_Debug") # New filename
        plt.savefig(filename, dpi=150)
        plt.close(fig)

    def plot_filled_batches_summary(df, plots_dir):
        """
        SUMMARY PLOT: Plots the number of filled batches per spill vs. spill number 
        (chronological, not sequential) for all runs, styled with one color per run.
        """
        # --- 1. Data Preparation ---
        if df.empty or df['filled_batches'].isnull().all():
            return # Nothing to plot

        # Ensure data is numeric and sorted chronologically by spill number
        df['run'] = pd.to_numeric(df['run'])
        df['spill'] = pd.to_numeric(df['spill'])
        df_sorted = df.sort_values(by='spill').dropna(subset=['filled_batches']).reset_index(drop=True)

        # --- 2. Plot Setup ---
        fig, ax = plt.subplots(figsize=(15, 10))
        unique_runs = sorted(df_sorted['run'].unique()) # Sort runs for a clean legend
        colormap = plt.get_cmap('hsv', len(unique_runs))

        # --- 3. Plotting Logic (Loop over each run) ---
        for i, run_number in enumerate(unique_runs):
            run_data = df_sorted[df_sorted['run'] == run_number]
            ax.plot(
                run_data.index, 
                run_data['filled_batches'], 
                'o', 
                ms=4, 
                color=colormap(i), 
                label=f'Run {run_number}'
            )

        # --- 4. Styling and Labels ---
        ax.set_title("Number of Filled Batches vs. Spill Number")
        ax.set_xlabel("Spill Number (chronological, not sequential)")
        ax.set_ylabel("Number of Filled Batches")
        ax.grid(True, linestyle='--')
        ax.set_xlim(left=-1, right=len(df_sorted))
        
        # Set Y-axis for discrete integer values (0-7 batches)
        ax.set_yticks(range(8))
        ax.set_ylim(bottom=-0.5, top=7.5)

        # --- 5. Custom X-Axis Ticks (to prevent overcrowding) ---
        tick_spacing = 40
        if len(df_sorted) < tick_spacing: # Adjust for smaller datasets
            tick_spacing = len(df_sorted) // 4 if len(df_sorted) > 4 else 1
        if tick_spacing == 0: tick_spacing = 1
        
        tick_locs = range(0, len(df_sorted), tick_spacing)
        tick_labels = df_sorted['spill'].iloc[tick_locs]
        
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.tick_params(axis='x', which='major', labelsize=8)

        # --- 6. Legend and Layout ---
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=10, title="Run Number", fontsize='small')
        plt.subplots_adjust(bottom=0.3) # Make room for legend and labels

        # --- 7. Save Figure ---
        output_filename = f"{plots_dir}/Overall_Filled_Batches_vs_Spill_Summary.png"
        plt.savefig(output_filename, dpi=200)
        plt.close(fig)
        print(f"\nOverall Filled Batches plot saved to: {output_filename}")    
