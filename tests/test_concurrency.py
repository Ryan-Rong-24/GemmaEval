#!/usr/bin/env python3
"""
Test script to find optimal concurrency limits for Gemini API.
"""

import asyncio
import time
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.evaluator import GemmaEvaluator
from src.dataset import DFireDataset

async def test_concurrency(max_concurrent: int, num_requests: int = 20):
    """Test a specific concurrency level."""
    print(f"\n🧪 Testing max_concurrent = {max_concurrent}")
    
    # Initialize components
    evaluator = GemmaEvaluator(Config.GEMINI_API_KEY)
    dataset = DFireDataset(Config.DATASET_PATH)
    
    # Get sample images
    sample_images = dataset.get_sample_images(
        split="test",
        max_per_category=num_requests // 4,  # 5 per category
        seed=42
    )[:num_requests]
    
    models = ["gemma-3-4b-it"]  # Use lightest model
    
    start_time = time.time()
    
    try:
        results = await evaluator.evaluate_images_async(
            sample_images,
            models,
            "simple",
            max_concurrent,
            use_tqdm=True
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        successful = len([r for r in results if not r.get('error')])
        error_rate = (len(results) - successful) / len(results) * 100
        throughput = successful / total_time
        avg_time = sum(r['processing_time'] for r in results if not r.get('error')) / successful
        
        print(f"✅ Results for max_concurrent = {max_concurrent}:")
        print(f"   • Total time: {total_time:.1f}s")
        print(f"   • Successful: {successful}/{len(results)}")
        print(f"   • Error rate: {error_rate:.1f}%")
        print(f"   • Throughput: {throughput:.2f} req/s")
        print(f"   • Avg processing time: {avg_time:.2f}s")
        
        return {
            'max_concurrent': max_concurrent,
            'total_time': total_time,
            'successful': successful,
            'total': len(results),
            'error_rate': error_rate,
            'throughput': throughput,
            'avg_processing_time': avg_time
        }
        
    except Exception as e:
        print(f"❌ Failed with max_concurrent = {max_concurrent}: {e}")
        return {
            'max_concurrent': max_concurrent,
            'error': str(e),
            'total_time': float('inf'),
            'successful': 0,
            'error_rate': 100.0,
            'throughput': 0.0
        }

def main():
    parser = argparse.ArgumentParser(description="Test optimal concurrency for Gemini API")
    parser.add_argument("--test-levels", nargs="+", type=int, 
                       default=[1, 3, 5, 8, 10, 15, 20],
                       help="Concurrency levels to test")
    parser.add_argument("--num-requests", type=int, default=20,
                       help="Number of requests per test")
    
    args = parser.parse_args()
    
    print("🚀 Starting Gemini API Concurrency Test")
    print(f"Testing levels: {args.test_levels}")
    print(f"Requests per test: {args.num_requests}")
    
    results = []
    
    for level in args.test_levels:
        result = asyncio.run(test_concurrency(level, args.num_requests))
        results.append(result)
        
        # Wait between tests to avoid rate limiting
        if level != args.test_levels[-1]:
            print("⏳ Waiting 30s between tests...")
            time.sleep(30)
    
    # Find optimal settings
    print("\n📊 SUMMARY")
    print("=" * 60)
    
    # Filter out failed tests
    valid_results = [r for r in results if r.get('successful', 0) > 0]
    
    if not valid_results:
        print("❌ All tests failed! You may need to check your API key or quotas.")
        return
    
    # Best throughput
    best_throughput = max(valid_results, key=lambda x: x['throughput'])
    print(f"🏆 Best throughput: {best_throughput['throughput']:.2f} req/s at max_concurrent={best_throughput['max_concurrent']}")
    
    # Lowest error rate
    lowest_error = min(valid_results, key=lambda x: x['error_rate'])
    print(f"🎯 Lowest error rate: {lowest_error['error_rate']:.1f}% at max_concurrent={lowest_error['max_concurrent']}")
    
    # Best balance (good throughput, low errors)
    balanced = min(valid_results, key=lambda x: x['total_time'] if x['error_rate'] < 10 else float('inf'))
    if balanced['error_rate'] < 10:
        print(f"⚖️  Best balance: max_concurrent={balanced['max_concurrent']} ({balanced['throughput']:.2f} req/s, {balanced['error_rate']:.1f}% errors)")
    
    print("\n📋 All Results:")
    for r in results:
        if 'error' in r:
            print(f"   max_concurrent={r['max_concurrent']:2d}: FAILED - {r['error']}")
        else:
            print(f"   max_concurrent={r['max_concurrent']:2d}: {r['throughput']:5.2f} req/s, {r['error_rate']:4.1f}% errors, {r['total_time']:5.1f}s total")

if __name__ == "__main__":
    main() 