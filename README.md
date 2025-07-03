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

python3 -m venv my_env
source my_env/bin/activate
python3.13 -m pip install --upgrade pip
pip3  install pandas numpy matplotlib seaborn scipy
source $(brew --prefix root)/bin/thisroot.sh

4. Run:

python3 spinquest_spill_intensity_analysis/spill_timing_analyzer/main.py

## Project 2: turn_structure_analyzer

Specfic to FreqHist_53MHz where we have full RF bucket, batch, and turn information for 0.03 s of a spill.



