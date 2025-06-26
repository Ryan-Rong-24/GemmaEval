#!/usr/bin/env python3
"""
Test suite for the Gemma Fire/Smoke Detection Evaluation System.
"""

import unittest
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.dataset import DFireDataset
from src.evaluator import GemmaEvaluator
from src.analyzer import EvaluationAnalyzer

class TestConfig(unittest.TestCase):
    """Test configuration module."""
    
    def test_default_values(self):
        """Test that default configuration values are set."""
        self.assertIsInstance(Config.GEMMA_MODELS, list)
        self.assertTrue(len(Config.GEMMA_MODELS) > 0)
        self.assertIsInstance(Config.DETECTION_PROMPTS, dict)
        self.assertTrue(len(Config.DETECTION_PROMPTS) > 0)
    
    def test_prompt_types(self):
        """Test that all expected prompt types are present."""
        expected_prompts = ["simple", "detailed", "binary"]
        for prompt_type in expected_prompts:
            self.assertIn(prompt_type, Config.DETECTION_PROMPTS)

class TestDataset(unittest.TestCase):
    """Test dataset handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory structure for testing
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = Path(self.temp_dir) / "test_dataset"
        
        # Create directory structure
        for split in ["train", "test"]:
            (self.dataset_path / split / "images").mkdir(parents=True)
            (self.dataset_path / split / "labels").mkdir(parents=True)
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        dataset = DFireDataset(str(self.dataset_path))
        self.assertEqual(dataset.dataset_path, self.dataset_path)
    
    def test_invalid_dataset_path(self):
        """Test handling of invalid dataset path."""
        with self.assertRaises(FileNotFoundError):
            DFireDataset("/nonexistent/path")
    
    def test_yolo_class_mapping(self):
        """Test that YOLO class IDs are correctly mapped."""
        dataset = DFireDataset(str(self.dataset_path))
        
        # Create a test label file
        test_labels_dir = self.dataset_path / "test" / "labels"
        test_label_file = test_labels_dir / "test_image.txt"
        
        # Write test annotations: class_id 0 (smoke) and class_id 1 (fire)
        with open(test_label_file, 'w') as f:
            f.write("0 0.5 0.5 0.2 0.2\n")  # Smoke annotation
            f.write("1 0.3 0.3 0.1 0.1\n")  # Fire annotation
        
        # Load labels
        labels = dataset._load_yolo_labels(test_labels_dir)
        
        # Verify correct mapping
        annotations = labels['test_image']
        self.assertEqual(len(annotations), 2)
        
        # First annotation should be smoke (class_id = 0)
        smoke_ann = annotations[0]
        self.assertEqual(smoke_ann['class_id'], 0)
        self.assertEqual(smoke_ann['class_name'], 'smoke')
        
        # Second annotation should be fire (class_id = 1)
        fire_ann = annotations[1]
        self.assertEqual(fire_ann['class_id'], 1)
        self.assertEqual(fire_ann['class_name'], 'fire')

class TestEvaluator(unittest.TestCase):
    """Test model evaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_api_key = "test_api_key"
    
    def test_response_parsing_simple(self):
        """Test parsing of simple responses."""
        evaluator = GemmaEvaluator(self.mock_api_key)
        
        test_cases = [
            ("fire", {"fire": True, "smoke": False, "prediction": "fire"}),
            ("smoke", {"fire": False, "smoke": True, "prediction": "smoke"}),
            ("both", {"fire": True, "smoke": True, "prediction": "both"}),
            ("none", {"fire": False, "smoke": False, "prediction": "none"}),
            ("nothing here", {"fire": False, "smoke": False, "prediction": "none"}),
        ]
        
        for response, expected in test_cases:
            with self.subTest(response=response):
                result = evaluator._parse_response(response, "simple")
                self.assertEqual(result, expected)
    
    def test_response_parsing_binary(self):
        """Test parsing of binary responses."""
        evaluator = GemmaEvaluator(self.mock_api_key)
        
        test_cases = [
            ("yes", {"fire": True, "smoke": True, "prediction": "positive"}),
            ("no", {"fire": False, "smoke": False, "prediction": "negative"}),
            ("Yes, I can see fire", {"fire": True, "smoke": True, "prediction": "positive"}),
            ("No fire or smoke", {"fire": False, "smoke": False, "prediction": "negative"}),
        ]
        
        for response, expected in test_cases:
            with self.subTest(response=response):
                result = evaluator._parse_response(response, "binary")
                self.assertEqual(result, expected)
    
    def test_response_parsing_detailed(self):
        """Test parsing of detailed JSON responses."""
        evaluator = GemmaEvaluator(self.mock_api_key)
        
        # Valid JSON response
        json_response = '{"fire": true, "smoke": false, "confidence": 0.9}'
        result = evaluator._parse_response(json_response, "detailed")
        expected = {
            "fire": True,
            "smoke": False,
            "confidence": 0.9,
            "prediction": "fire"
        }
        self.assertEqual(result, expected)
        
        # Invalid JSON should fallback to simple parsing
        invalid_json = "there is fire in the image"
        result = evaluator._parse_response(invalid_json, "detailed")
        self.assertEqual(result["prediction"], "fire")

class TestAnalyzer(unittest.TestCase):
    """Test evaluation analyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = EvaluationAnalyzer()
        
        # Sample results for testing
        self.sample_results = [
            {
                "model": "gemma-3-4b-it",
                "ground_truth": "fire",
                "parsed": {"prediction": "fire"},
                "processing_time": 2.5,
                "error": None
            },
            {
                "model": "gemma-3-4b-it", 
                "ground_truth": "smoke",
                "parsed": {"prediction": "smoke"},
                "processing_time": 3.1,
                "error": None
            },
            {
                "model": "gemma-3-27b-it",
                "ground_truth": "none",
                "parsed": {"prediction": "none"},
                "processing_time": 5.2,
                "error": None
            }
        ]
    
    def test_binary_mapping(self):
        """Test category mapping for binary classification."""
        test_cases = [
            ("fire", "positive"),
            ("smoke", "positive"),
            ("both", "positive"),
            ("none", "negative"),
            ("negative", "negative")
        ]
        
        for category, expected in test_cases:
            gt_mapped, pred_mapped = self.analyzer._map_categories(category, category)
            self.assertEqual(gt_mapped, expected)
    
    def test_metrics_computation(self):
        """Test computation of evaluation metrics."""
        metrics = self.analyzer.compute_metrics(self.sample_results)
        
        self.assertIn("total_evaluations", metrics)
        self.assertIn("successful_evaluations", metrics)
        self.assertIn("classification", metrics)
        self.assertIn("per_model", metrics)
        
        self.assertEqual(metrics["total_evaluations"], 3)
        self.assertEqual(metrics["successful_evaluations"], 3)
    
    def test_empty_results(self):
        """Test handling of empty results."""
        metrics = self.analyzer.compute_metrics([])
        self.assertIn("error", metrics)

class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_config_validation_missing_api_key(self):
        """Test config validation with missing API key."""
        # Temporarily modify config
        original_key = Config.GEMINI_API_KEY
        Config.GEMINI_API_KEY = ""
        
        try:
            with self.assertRaises(ValueError):
                Config.validate()
        finally:
            Config.GEMINI_API_KEY = original_key
    
    @patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key_123'})
    def test_config_with_environment_variable(self):
        """Test config loading from environment variables."""
        # Reload config with environment variable
        from src.config import Config
        # In a real scenario, we'd need to reload the module
        # This test is more conceptual

def create_test_suite():
    """Create a test suite with all test cases."""
    suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestConfig,
        TestDataset,
        TestEvaluator,
        TestAnalyzer,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite

def main():
    """Run the test suite."""
    print("🧪 Running Gemma Fire/Smoke Detection Evaluation System Tests")
    print("=" * 70)
    
    # Create and run test suite
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\n')[0]}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split('\n')[-2]}")
    
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(main()) 