import unittest
import os
import sys
import glob

# Ensure the project root is in sys.path so we can import the analyzer modules
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

try:
    from spill_timing_analyzer.analysis import SpillAnalyzer
    import spill_timing_analyzer.config as spill_config
    from turn_structure_analyzer.analysis import TurnStructureAnalyzer
    import turn_structure_analyzer.config as turn_config
except ImportError as e:
    print(f"Error importing analysis modules: {e}")
    print("Make sure you are running this script from the project root.")
    sys.exit(1)

class TestAnalysisPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Setup paths and find test data."""
        cls.data_dir = os.path.join(PROJECT_ROOT, 'data')
        
        # Check if data directory exists
        if not os.path.exists(cls.data_dir):
            print(f"\n[WARNING] Data directory not found at: {cls.data_dir}")
            print("Please create a 'data' folder and add a sample .root file and .tsv file.")
            raise unittest.SkipTest("Data directory missing")

        # Find a sample ROOT file
        root_files = glob.glob(os.path.join(cls.data_dir, "*.root"))
        if not root_files:
            print(f"\n[WARNING] No .root files found in: {cls.data_dir}")
            raise unittest.SkipTest("No ROOT files found")
            
        cls.sample_root_file = root_files[0]
        print(f"\nUsing sample ROOT file: {os.path.basename(cls.sample_root_file)}")

    def test_01_spill_timing_analyzer(self):
        """Smoke test for SpillAnalyzer."""
        print("\n--- Testing SpillAnalyzer ---")
        
        # Force config to use the local data directory for this test
        spill_config.ACNET_TSV_PATH = self.data_dir
        spill_config.ROOT_FILE_PATH = self.data_dir
        
        # Initialize the analyzer (this loads data and runs basic stats)
        try:
            analyzer = SpillAnalyzer(self.sample_root_file, spill_config)
        except Exception as e:
            self.fail(f"SpillAnalyzer initialization failed: {e}")

        # Assertions to ensure data was loaded
        self.assertIsNotNone(analyzer.data, "Data DataFrame should not be None")
        self.assertFalse(analyzer.data.empty, "Data DataFrame should not be empty")
        self.assertNotEqual(analyzer.run_num, -1, "Run number should be extracted from filename")
        
        print(f"Successfully analyzed spill {analyzer.spill_num} from run {analyzer.run_num}")

    def test_02_turn_structure_analyzer(self):
        """Smoke test for TurnStructureAnalyzer."""
        print("\n--- Testing TurnStructureAnalyzer ---")
        
        # Construct configuration dictionary
        # We use values from the config module but override paths to use local data
        analyzer_config = {
            'BUCKETS_PER_BATCH': turn_config.BUCKETS_PER_BATCH,
            'NUM_BATCHES_PER_TURN': turn_config.NUM_BATCHES_PER_TURN,
            'HISTOGRAM_NAME': turn_config.HISTOGRAM_NAME,
            'ACNET_TSV_PATH': self.data_dir,
            'AUTO_FIND_START_BUCKET': True,
            'START_FINDER_NOISE_WINDOW': turn_config.START_FINDER_NOISE_WINDOW,
            'START_FINDER_THRESHOLD_STD': turn_config.START_FINDER_THRESHOLD_STD,
            'REFINE_START_BUCKET_ITERATIVELY': False, # Disable for speed during testing
            'REFINEMENT_SEARCH_WINDOW': turn_config.REFINEMENT_SEARCH_WINDOW,
            'REFINEMENT_SAMPLE_TURNS': turn_config.REFINEMENT_SAMPLE_TURNS,
            'PLOTS_DIR': os.path.join(PROJECT_ROOT, 'test_plots'), # Dummy output dir
            'PLOT_COLORS': turn_config.PLOT_COLORS
        }

        # Initialize
        try:
            analyzer = TurnStructureAnalyzer(self.sample_root_file, analyzer_config)
        except Exception as e:
            self.fail(f"TurnStructureAnalyzer initialization failed: {e}")

        # Run analysis
        try:
            analyzer.run_analysis()
        except Exception as e:
            self.fail(f"TurnStructureAnalyzer execution failed: {e}")

        # Assertions to ensure results were generated
        self.assertIn('turn_df', analyzer.results)
        self.assertFalse(analyzer.results['turn_df'].empty, "Turn analysis results should not be empty")
        
        print(f"Successfully processed {len(analyzer.results['turn_df'])} turns.")

if __name__ == '__main__':
    unittest.main(verbosity=2)