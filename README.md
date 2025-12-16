# spinquest_spill_intensity_analysis
Analysis of data recorded by the SpinQuest/DarkQuest Beam Intensity Monitor (BIM)

## Project 1: spill_timing_analyzer

Loop over all spills. For each spill access any intensity vs time histogram and analyze the spill timing. 

Instructions:

1. Edit config.py to select a given FreqHist. Options:
- FreqHist_53MHz  (0.03 s duration)
- FreqHist_1MHz   (0.1 s duration)
- FreqHist_100kHz (1 s duration)
- FreqHist_10kHz  (full 4 s spill)
- FreqHist_7_5kHz (full 4 s spill)
- FreqHist_1kHz   (full 4 s spill)

2. Edit config.py to select the path to the rootfile directory.

3. Example setup: 

```
python3 -m venv my_env
source my_env/bin/activate
python3.13 -m pip install --upgrade pip
pip3  install pandas numpy matplotlib seaborn scipy
source $(brew --prefix root)/bin/thisroot.sh
```

4. Run:

```
python3 spinquest_spill_intensity_analysis/spill_timing_analyzer/main.py
```

## Project 2: turn_structure_analyzer

Specfic to FreqHist_53MHz where we have full RF bucket, batch, and turn information for 0.03 s of a spill.


## Performance Recommendations: Data Reading

### Why uproot is Faster than PyROOT for Histogram Data

The current PyROOT implementation reads histogram data bin-by-bin using Python loops:

```python
# Current PyROOT approach (slow for large histograms)
times = [hist.GetBinCenter(i) for i in range(1, n_bins + 1)]
intensities = [hist.GetBinContent(i) for i in range(1, n_bins + 1)]
```

This approach is slow because:
- **Python loop overhead**: Each bin requires a separate Python-to-C++ call
- **For FreqHist_53MHz** (1,593,118 bins): This means ~3.2 million function calls
- **Scales linearly**: Processing time increases directly with bin count

### Recommended: Use uproot for Faster I/O

The uproot library uses vectorized numpy operations that are significantly faster:

```python
# Faster uproot approach
import uproot
with uproot.open(filename) as root_file:
    hist = root_file[hist_name]
    intensities, bin_edges = hist.to_numpy()  # Single vectorized call
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Vectorized numpy
```

**Performance benefits:**
- **Vectorized operations**: Data is extracted in bulk, not bin-by-bin
- **Native numpy arrays**: Direct conversion without Python loop overhead
- **Lightweight**: No need for full ROOT installation (just `pip install uproot`)
- **Optimized for Python**: Designed specifically for efficient Python I/O

### Expected Speedup

| Histogram | Bins | PyROOT (approx) | uproot (approx) | Speedup |
|-----------|------|-----------------|-----------------|---------|
| FreqHist_1kHz | 4,000 | ~0.1s | ~0.01s | ~10x |
| FreqHist_10kHz | 40,000 | ~1s | ~0.05s | ~20x |
| FreqHist_53MHz | 1,593,118 | ~30-60s | ~0.5-1s | ~50-100x |

*Note: Actual performance depends on system and disk speed.*

### When to Use Each Library

| Use Case | Recommended Library |
|----------|---------------------|
| Reading histogram data to numpy/pandas | **uproot** |
| Rebinning histograms before extraction | PyROOT |
| Creating/modifying ROOT files | PyROOT |
| Plotting with ROOT's TCanvas | PyROOT |
| High-performance batch processing | **uproot** |

### Installation

To use uproot, add it to your environment:

```bash
pip install uproot
```

See `analyze_root_file.py` for a working example comparing both approaches.


