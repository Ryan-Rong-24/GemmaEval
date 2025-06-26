#!/usr/bin/env python3
"""
Demonstration script for the Gemma Fire/Smoke Detection Evaluation System.
This script shows the key functionality without requiring the full dataset.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.evaluator import GemmaEvaluator
from src.analyzer import EvaluationAnalyzer

def create_sample_results() -> List[Dict]:
    """Create sample evaluation results for demonstration."""
    return [
        {
            "model": "gemma-3-27b-it", 
            "image_path": "sample_fire_1.jpg",
            "ground_truth": "fire",
            "response": "I can see fire in this image.",
            "parsed": {"fire": True, "smoke": False, "prediction": "fire"},
            "processing_time": 3.2,
            "error": None,
            "image_name": "sample_fire_1"
        },
        {
            "model": "gemma-3-4b-it",
            "image_path": "sample_fire_1.jpg", 
            "ground_truth": "fire",
            "response": "There is fire visible",
            "parsed": {"fire": True, "smoke": False, "prediction": "fire"},
            "processing_time": 1.8,
            "error": None,
            "image_name": "sample_fire_1"
        },
        {
            "model": "gemma-3-27b-it",
            "image_path": "sample_smoke_1.jpg",
            "ground_truth": "smoke", 
            "response": "I can see smoke in the image",
            "parsed": {"fire": False, "smoke": True, "prediction": "smoke"},
            "processing_time": 2.9,
            "error": None,
            "image_name": "sample_smoke_1"
        },
        {
            "model": "gemma-3-4b-it",
            "image_path": "sample_smoke_1.jpg",
            "ground_truth": "smoke",
            "response": "There is smoke present",
            "parsed": {"fire": False, "smoke": True, "prediction": "smoke"}, 
            "processing_time": 1.5,
            "error": None,
            "image_name": "sample_smoke_1"
        },
        {
            "model": "gemma-3-27b-it",
            "image_path": "sample_none_1.jpg",
            "ground_truth": "none",
            "response": "I don't see any fire or smoke",
            "parsed": {"fire": False, "smoke": False, "prediction": "none"},
            "processing_time": 2.1,
            "error": None,
            "image_name": "sample_none_1"
        },
        {
            "model": "gemma-3-4b-it",
            "image_path": "sample_none_1.jpg",
            "ground_truth": "none",
            "response": "No fire or smoke visible",
            "parsed": {"fire": False, "smoke": False, "prediction": "none"},
            "processing_time": 1.3,
            "error": None,
            "image_name": "sample_none_1"
        },
        {
            "model": "gemma-3-27b-it",
            "image_path": "sample_both_1.jpg",
            "ground_truth": "both",
            "response": "I can see both fire and smoke in this image",
            "parsed": {"fire": True, "smoke": True, "prediction": "both"},
            "processing_time": 3.5,
            "error": None,
            "image_name": "sample_both_1"
        },
        {
            "model": "gemma-3-4b-it",
            "image_path": "sample_both_1.jpg",
            "ground_truth": "both",
            "response": "Both fire and smoke are present", 
            "parsed": {"fire": True, "smoke": True, "prediction": "both"},
            "processing_time": 2.0,
            "error": None,
            "image_name": "sample_both_1"
        }
    ]

def demo_response_parsing():
    """Demonstrate response parsing functionality."""
    print("🔍 Response Parsing Demo")
    print("=" * 50)
    
    # Mock API key for demo
    evaluator = GemmaEvaluator("demo_key")
    
    # Test different response types
    test_responses = [
        ("fire", "simple"),
        ("I can see smoke in the image", "simple"),
        ("both fire and smoke are visible", "simple"),
        ("none", "simple"),
        ('{"fire": true, "smoke": false, "confidence": 0.95}', "detailed")
    ]
    
    for response, prompt_type in test_responses:
        parsed = evaluator._parse_response(response, prompt_type)
        print(f"Response: '{response}' ({prompt_type})")
        print(f"Parsed: {parsed}")
        print()

def demo_metrics_computation():
    """Demonstrate metrics computation."""
    print("📊 Metrics Computation Demo")
    print("=" * 50)
    
    analyzer = EvaluationAnalyzer()
    sample_results = create_sample_results()
    
    # Compute metrics
    metrics = analyzer.compute_metrics(sample_results)
    
    print(f"Total evaluations: {metrics['total_evaluations']}")
    print(f"Successful evaluations: {metrics['successful_evaluations']}")
    print(f"Error rate: {metrics['error_rate']:.1%}")
    print(f"Average processing time: {metrics['average_processing_time']:.2f}s")
    
    print("\nClassification Metrics:")
    classification = metrics['classification']
    for metric, value in classification.items():
        print(f"  {metric}: {value:.3f}")
    
    print("\nPer-Model Performance:")
    for model, model_metrics in metrics['per_model'].items():
        print(f"\n{model}:")
        print(f"  Samples: {model_metrics['count']}")
        print(f"  Avg time: {model_metrics['average_processing_time']:.2f}s")
        print(f"  F1-Score: {model_metrics['classification']['f1_score']:.3f}")
        print(f"  Accuracy: {model_metrics['classification']['accuracy']:.3f}")

def demo_configuration():
    """Demonstrate configuration system."""
    print("⚙️  Configuration Demo")
    print("=" * 50)
    
    print("Available Gemma Models:")
    for model in Config.GEMMA_MODELS:
        print(f"  - {model}")
    
    print("\nPrompt Types:")
    for prompt_type, prompt_text in Config.DETECTION_PROMPTS.items():
        print(f"  {prompt_type}: {prompt_text[:50]}...")
    
    print(f"\nDataset path: {Config.DATASET_PATH}")
    print(f"Results path: {Config.RESULTS_PATH}")
    print(f"Max images per category: {Config.MAX_IMAGES_PER_CATEGORY}")
    print(f"Batch size: {Config.BATCH_SIZE}")

def demo_report_generation():
    """Demonstrate report generation."""
    print("📋 Report Generation Demo")
    print("=" * 50)
    
    analyzer = EvaluationAnalyzer()
    sample_results = create_sample_results()
    
    # Sample dataset stats
    dataset_stats = {
        "total_images": 100,
        "categories": {
            "fire": 25,
            "smoke": 30,
            "both": 20,
            "none": 25
        },
        "annotations": {
            "fire": 45,
            "smoke": 50,
            "total": 95
        }
    }
    
    # Generate report
    report_path = analyzer.generate_report(sample_results, dataset_stats, "demo_report.json")
    
    print(f"Generated report: {report_path}")
    
    # Show summary
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    print("\nReport Summary:")
    summary = report['evaluation_summary'] 
    for key, value in summary.items():
        if isinstance(value, list):
            print(f"  {key}: {', '.join(value)}")
        else:
            print(f"  {key}: {value}")

def main():
    """Run the demonstration."""
    print("🔥 Gemma Fire/Smoke Detection Evaluation System Demo")
    print("=" * 60)
    print("This demo shows the key functionality without requiring the full dataset or API key.\n")
    
    try:
        # Configuration demo
        demo_configuration()
        print()
        
        # Response parsing demo  
        demo_response_parsing()
        
        # Metrics computation demo
        demo_metrics_computation()
        print()
        
        # Report generation demo
        demo_report_generation()
        
        print("\n" + "=" * 60)
        print("✅ Demo completed successfully!")
        print("\nTo run the actual evaluation system:")
        print("1. Set up your GEMINI_API_KEY in .env file")
        print("2. Download the D-Fire dataset") 
        print("3. Run: python main.py --quick-test")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 