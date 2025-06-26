#!/usr/bin/env python3
"""
Main evaluation script for Gemma fire/smoke detection evaluation.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import json
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.dataset import DFireDataset
from src.evaluator import GemmaEvaluator
from src.analyzer import EvaluationAnalyzer

console = Console()

def setup_logging():
    """Setup logging configuration using the session-specific log file."""
    # Create log directory if it doesn't exist
    if Config.LOG_FILE:
        log_path = Path(Config.LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Config.LOG_FILE),
                logging.StreamHandler()  # Also log to console
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured. Log file: {Config.LOG_FILE}")
        return logger
    else:
        # Fallback to console logging only
        logging.basicConfig(
            level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        logger.warning("No log file configured, using console logging only")
        return logger

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Gemma models on fire/smoke detection using D-Fire dataset"
    )
    
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=None,
        help="List of Gemma models to evaluate (default: all configured models)"
    )
    
    parser.add_argument(
        "--prompt-types",
        nargs="+",
        default=None,
        choices=list(Config.DETECTION_PROMPTS.keys()),
        help="Types of prompts to use (default: all types)"
    )
    
    parser.add_argument(
        "--max-images",
        type=int,
        default=Config.MAX_IMAGES_PER_CATEGORY,
        help="Maximum images per category to evaluate (use 0 or negative for unlimited)"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum concurrent evaluations"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=Config.RESULTS_PATH,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--dataset-split",
        choices=["train", "test"],
        default="test",
        help="Dataset split to use for evaluation (ignored if --dataset-splits is used)"
    )
    
    parser.add_argument(
        "--dataset-splits",
        nargs="+",
        choices=["train", "test"],
        default=None,
        help="Multiple dataset splits to use for evaluation (overrides --dataset-split)"
    )
    
    parser.add_argument(
        "--use-all-images",
        action="store_true",
        help="Use all available images (ignores --max-images)"
    )
    
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a quick test with minimal images and models"
    )
    
    parser.add_argument(
        "--analyze-only",
        type=str,
        help="Path to existing results file to analyze (skip evaluation)"
    )
    
    parser.add_argument(
        "--use-tqdm",
        action="store_true",
        help="Use tqdm progress bars instead of rich progress display"
    )
    
    parser.add_argument(
        "--use-async",
        action="store_true",
        help="Use async evaluator for better parallel performance"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent async evaluations (only used with --use-async)"
    )
    
    return parser.parse_args()

def display_dataset_info(dataset_stats: dict) -> None:
    """Display dataset information in a nice table."""
    table = Table(title="D-Fire Dataset Statistics")
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Count", style="magenta")
    table.add_column("Percentage", style="green")
    
    total = dataset_stats['total_images']
    for category, count in dataset_stats['categories'].items():
        percentage = (count / total * 100) if total > 0 else 0
        table.add_row(
            category.replace('_', ' ').title(),
            str(count),
            f"{percentage:.1f}%"
        )
    
    table.add_row("Total", str(total), "100.0%", style="bold")
    console.print(table)
    
    # Annotation statistics
    if 'annotations' in dataset_stats:
        ann_stats = dataset_stats['annotations']
        ann_table = Table(title="Annotation Statistics")
        ann_table.add_column("Type", style="cyan")
        ann_table.add_column("Count", style="magenta")
        
        ann_table.add_row("Fire Bounding Boxes", str(ann_stats['fire']))
        ann_table.add_row("Smoke Bounding Boxes", str(ann_stats['smoke']))
        ann_table.add_row("Total Annotations", str(ann_stats['total']))
        
        console.print(ann_table)

def display_evaluation_summary(results: List[dict]) -> None:
    """Display evaluation summary."""
    if not results:
        console.print("No results to display", style="red")
        return
    
    # Count by model
    model_counts = {}
    successful_evals = 0
    total_time = 0
    
    for result in results:
        model = result['model']
        model_counts[model] = model_counts.get(model, 0) + 1
        
        if not result.get('error'):
            successful_evals += 1
            total_time += result.get('processing_time', 0)
    
    # Summary table
    summary_table = Table(title="Evaluation Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="magenta")
    
    summary_table.add_row("Total Evaluations", str(len(results)))
    summary_table.add_row("Successful Evaluations", str(successful_evals))
    summary_table.add_row("Error Rate", f"{((len(results) - successful_evals) / len(results) * 100):.1f}%")
    summary_table.add_row("Average Processing Time", f"{(total_time / successful_evals):.2f}s" if successful_evals > 0 else "N/A")
    
    console.print(summary_table)
    
    # Model counts
    model_table = Table(title="Evaluations per Model")
    model_table.add_column("Model", style="cyan")
    model_table.add_column("Evaluations", style="magenta")
    
    for model, count in model_counts.items():
        model_table.add_row(model.replace('gemma-', ''), str(count))
    
    console.print(model_table)

def main():
    """Main evaluation function."""
    args = parse_args()
    
    console.print(Panel.fit(
        "[bold blue]Gemma Fire/Smoke Detection Evaluation System[/bold blue]\n"
        "Evaluating Gemma models on the D-Fire dataset",
        border_style="blue"
    ))
    
    # Configure models and prompts early to setup session paths
    models = args.models if args.models else Config.GEMMA_MODELS
    prompt_types = args.prompt_types if args.prompt_types else list(Config.DETECTION_PROMPTS.keys())
    
    # Determine dataset splits to use
    if args.dataset_splits:
        dataset_splits = args.dataset_splits
    else:
        dataset_splits = [args.dataset_split]
    
    # Determine max images per category
    if args.use_all_images:
        max_images_per_category = None
        max_images_label = "all"
    elif args.max_images <= 0:
        max_images_per_category = None
        max_images_label = "all"
    else:
        max_images_per_category = args.max_images
        max_images_label = str(args.max_images)
    
    # Quick test configuration
    if args.quick_test:
        console.print("🚀 Running quick test mode", style="yellow")
        models = ["gemma-3-4b-it"]  # Use smallest model
        prompt_types = ["simple"]   # Use simplest prompt
        max_images_per_category = 5 # Minimum images
        max_images_label = "5"
        args.max_workers = 2            # Fewer workers
        dataset_splits = ["test"]       # Single split for quick test
    
    # Create splits string for session path
    splits_str = "_".join(sorted(dataset_splits))
    
    # Setup session-specific paths with timestamp and configuration
    Config.setup_session_paths(
        models=models,
        prompt_types=prompt_types,
        max_images=max_images_label,
        dataset_split=splits_str
    )
    
    # Create directories
    Config.create_directories()
    
    # Setup logging after paths are configured
    logger = setup_logging()
    
    console.print(f"📁 Session folder: {Config.RESULTS_PATH}")
    console.print(f"📊 Evaluations will be saved to: {Config.EVALS_PATH}")
    console.print(f"📈 Plots will be saved to: {Config.PLOTS_PATH}")
    console.print(f"📝 Logs will be saved to: {Config.LOG_FILE}")
    
    logger.info("=" * 80)
    logger.info("GEMMA FIRE/SMOKE DETECTION EVALUATION SESSION STARTED")
    logger.info("=" * 80)
    logger.info(f"Session folder: {Config.RESULTS_PATH}")
    logger.info(f"Models to evaluate: {models}")
    logger.info(f"Prompt types: {prompt_types}")
    logger.info(f"Max images per category: {max_images_label}")
    logger.info(f"Dataset split: {splits_str}")
    
    # Validate configuration
    try:
        Config.validate()
        console.print("✅ Configuration validated", style="green")
        logger.info("Configuration validation successful")
    except ValueError as e:
        console.print(f"❌ Configuration error: {e}", style="red")
        logger.error(f"Configuration validation failed: {e}")
        return 1
    
    # Handle analyze-only mode
    if args.analyze_only:
        console.print(f"📊 Analyzing existing results from: {args.analyze_only}")
        logger.info(f"Running analysis-only mode on: {args.analyze_only}")
        try:
            with open(args.analyze_only, 'r') as f:
                results = json.load(f)
            
            analyzer = EvaluationAnalyzer()
            
            # Generate report if results is a list (raw results)
            if isinstance(results, list):
                # Load dataset for stats - use the specified splits
                dataset = DFireDataset(Config.DATASET_PATH)
                
                # Get combined dataset stats for all splits
                combined_stats = {'total_images': 0, 'categories': {}, 'annotations': {'total': 0, 'fire': 0, 'smoke': 0}}
                
                for split in dataset_splits:
                    split_stats = dataset.get_dataset_stats(split)
                    combined_stats['total_images'] += split_stats['total_images']
                    
                    # Combine category counts
                    for category, count in split_stats['categories'].items():
                        if category not in combined_stats['categories']:
                            combined_stats['categories'][category] = 0
                        combined_stats['categories'][category] += count
                    
                    # Combine annotation counts
                    if 'annotations' in split_stats:
                        combined_stats['annotations']['total'] += split_stats['annotations']['total']
                        combined_stats['annotations']['fire'] += split_stats['annotations']['fire']
                        combined_stats['annotations']['smoke'] += split_stats['annotations']['smoke']
                
                report_path = analyzer.generate_report(results, combined_stats)
                console.print(f"📋 Analysis complete! Report saved to: {report_path}")
                logger.info(f"Analysis complete. Report saved to: {report_path}")
            else:
                console.print("Results file appears to be a report, not raw results")
                logger.warning("Provided file appears to be a report, not raw results")
            
            return 0
            
        except Exception as e:
            console.print(f"❌ Error analyzing results: {e}", style="red")
            logger.error(f"Error in analysis-only mode: {e}")
            return 1
    
    # Initialize components
    console.print("🔧 Initializing components...")
    logger.info("Initializing evaluation components...")
    
    try:
        dataset = DFireDataset(Config.DATASET_PATH)
        evaluator = GemmaEvaluator(Config.GEMINI_API_KEY)
        analyzer = EvaluationAnalyzer()
        
        console.print("✅ Components initialized", style="green")
        logger.info("All components initialized successfully")
    except Exception as e:
        console.print(f"❌ Initialization error: {e}", style="red")
        logger.error(f"Component initialization failed: {e}")
        return 1
    
    # Load and display dataset statistics
    console.print("📊 Loading dataset...")
    logger.info(f"Loading dataset from: {Config.DATASET_PATH}")
    
    # Get combined dataset stats for all splits
    combined_stats = {'total_images': 0, 'categories': {}, 'annotations': {'total': 0, 'fire': 0, 'smoke': 0}}
    
    for split in dataset_splits:
        split_stats = dataset.get_dataset_stats(split)
        combined_stats['total_images'] += split_stats['total_images']
        
        # Combine category counts
        for category, count in split_stats['categories'].items():
            if category not in combined_stats['categories']:
                combined_stats['categories'][category] = 0
            combined_stats['categories'][category] += count
        
        # Combine annotation counts
        if 'annotations' in split_stats:
            combined_stats['annotations']['total'] += split_stats['annotations']['total']
            combined_stats['annotations']['fire'] += split_stats['annotations']['fire']
            combined_stats['annotations']['smoke'] += split_stats['annotations']['smoke']
    
    display_dataset_info(combined_stats)
    logger.info(f"Dataset loaded from {len(dataset_splits)} split(s): {dataset_splits}. Total images: {combined_stats.get('total_images', 0)}")
    
    # Get sample images
    console.print(f"🖼️  Sampling images (max {max_images_label} per category)...")
    logger.info(f"Sampling {max_images_label} images per category from {len(dataset_splits)} split(s): {dataset_splits}")
    
    if len(dataset_splits) == 1:
        # Single split - use existing method
        sample_images = dataset.get_sample_images(
            split=dataset_splits[0],
            max_per_category=max_images_per_category,
            seed=42
        )
    else:
        # Multiple splits - use new method
        sample_images = dataset.get_all_images(
            splits=dataset_splits,
            max_per_category=max_images_per_category,
            seed=42
        )
    
    console.print(f"Selected {len(sample_images)} images for evaluation")
    logger.info(f"Selected {len(sample_images)} images for evaluation")
    
    console.print(f"Models to evaluate: {', '.join(models)}")
    console.print(f"Prompt types: {', '.join(prompt_types)}")
    console.print(f"Dataset splits: {', '.join(dataset_splits)}")
    
    # Run evaluation
    all_results = []
    logger.info("Starting evaluation process...")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        
        main_task = progress.add_task("Overall Progress", total=len(prompt_types))
        
        for prompt_type in prompt_types:
            progress.update(main_task, description=f"Evaluating with {prompt_type} prompts...")
            logger.info(f"Starting evaluation with {prompt_type} prompts")
            
            try:
                if args.use_async:
                    # Use async evaluator
                    import asyncio
                    logger.info(f"Using async evaluator with {args.max_concurrent} max concurrent requests")
                    results = asyncio.run(
                        evaluator.evaluate_images_async(
                            sample_images,
                            models,
                            prompt_type,
                            args.max_concurrent,
                            use_tqdm=args.use_tqdm
                        )
                    )
                else:
                    # Use threaded evaluator
                    logger.info(f"Using threaded evaluator with {args.max_workers} max workers")
                    results = evaluator.evaluate_images(
                        sample_images,
                        models,
                        prompt_type,
                        args.max_workers,
                        use_tqdm=args.use_tqdm
                    )
                
                all_results.extend(results)
                progress.advance(main_task)
                logger.info(f"Completed evaluation with {prompt_type} prompts. Got {len(results)} results.")
                
            except Exception as e:
                console.print(f"❌ Error during evaluation with {prompt_type}: {e}", style="red")
                logger.error(f"Error during evaluation with {prompt_type}: {e}")
                continue
    
    # Display evaluation summary
    console.print("\n📈 Evaluation Complete!")
    logger.info("Evaluation process completed")
    display_evaluation_summary(all_results)
    
    # Save raw results to evals folder
    results_file = Path(Config.EVALS_PATH) / "raw_results.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    console.print(f"💾 Raw results saved to: {results_file}")
    logger.info(f"Raw results saved to: {results_file}")
    
    # Generate comprehensive report
    console.print("📋 Generating analysis report...")
    logger.info("Starting report generation...")
    try:
        report_path = analyzer.generate_report(all_results, combined_stats)
        console.print(f"✅ Report generated: {report_path}")
        logger.info(f"Analysis report generated: {report_path}")
        
        # Display key metrics
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        if 'performance_metrics' in report:
            metrics = report['performance_metrics']
            
            console.print("\n🎯 Key Performance Metrics:")
            
            if 'classification' in metrics:
                classification = metrics['classification']
                console.print(f"Classification F1-Score: {classification['f1_score']:.3f}")
                console.print(f"Classification Accuracy: {classification['accuracy']:.3f}")
                logger.info(f"F1-Score: {classification['f1_score']:.3f}, Accuracy: {classification['accuracy']:.3f}")
            
            console.print(f"Average Processing Time: {metrics.get('average_processing_time', 0):.2f}s")
            logger.info(f"Average Processing Time: {metrics.get('average_processing_time', 0):.2f}s")
        
    except Exception as e:
        console.print(f"❌ Error generating report: {e}", style="red")
        logger.error(f"Error generating report: {e}")
        return 1
    
    console.print("\n🎉 Evaluation pipeline completed successfully!", style="bold green")
    logger.info("=" * 80)
    logger.info("EVALUATION SESSION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    return 0

if __name__ == "__main__":
    sys.exit(main())