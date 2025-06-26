"""
Analysis module for computing evaluation metrics and generating reports.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from .config import Config

class EvaluationAnalyzer:
    """Analyzer for computing metrics and generating evaluation reports."""
    
    def __init__(self):
        """Initialize the evaluation analyzer."""
        # Use session-specific paths if available, otherwise use defaults
        if Config.RESULTS_PATH:
            self.results_path = Path(Config.RESULTS_PATH)
            self.plots_path = Path(Config.PLOTS_PATH)
            self.evals_path = Path(Config.EVALS_PATH)
        else:
            # Fallback for legacy usage
            self.results_path = Path(Config.BASE_RESULTS_PATH)
            self.plots_path = self.results_path / "plots"
            self.evals_path = self.results_path / "evals"
        
        # Ensure directories exist
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.plots_path.mkdir(parents=True, exist_ok=True)
        self.evals_path.mkdir(parents=True, exist_ok=True)
    
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute evaluation metrics from results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary containing computed metrics
        """
        # Extract ground truth and predictions
        ground_truths = []
        predictions = []
        models = []
        processing_times = []
        errors = []
        
        for result in results:
            if result.get('parsed'):
                ground_truths.append(result['ground_truth'])
                predictions.append(result['parsed']['prediction'])
                models.append(result['model'])
                processing_times.append(result.get('processing_time', 0))
                errors.append(1 if result.get('error') else 0)
        
        if not ground_truths:
            return {'error': 'No valid results to analyze'}
        
        # Compute metrics
        metrics = {
            'total_evaluations': len(results),
            'successful_evaluations': len(ground_truths),
            'error_rate': sum(errors) / len(results) if results else 0,
            'average_processing_time': np.mean(processing_times) if processing_times else 0,
            'classification': {
                'accuracy': accuracy_score(ground_truths, predictions),
                'precision': precision_score(ground_truths, predictions, average='weighted', zero_division=0),
                'recall': recall_score(ground_truths, predictions, average='weighted', zero_division=0),
                'f1_score': f1_score(ground_truths, predictions, average='weighted', zero_division=0)
            }
        }
        
        # Compute per-model metrics
        unique_models = list(set(models))
        metrics['per_model'] = {}
        
        for model in unique_models:
            model_indices = [i for i, m in enumerate(models) if m == model]
            model_gt = [ground_truths[i] for i in model_indices]
            model_pred = [predictions[i] for i in model_indices]
            model_times = [processing_times[i] for i in model_indices]
            
            if model_gt:
                metrics['per_model'][model] = {
                    'count': len(model_gt),
                    'average_processing_time': np.mean(model_times),
                    'classification': {
                        'accuracy': accuracy_score(model_gt, model_pred),
                        'precision': precision_score(model_gt, model_pred, average='weighted', zero_division=0),
                        'recall': recall_score(model_gt, model_pred, average='weighted', zero_division=0),
                        'f1_score': f1_score(model_gt, model_pred, average='weighted', zero_division=0)
                    }
                }
        
        # Compute confusion matrices
        unique_categories = list(set(ground_truths + predictions))
        metrics['confusion_matrix'] = confusion_matrix(
            ground_truths, predictions, labels=unique_categories
        ).tolist()
        metrics['confusion_matrix_labels'] = unique_categories
        
        return metrics
    
    def generate_plots(self, 
                      results: List[Dict[str, Any]], 
                      metrics: Dict[str, Any],
                      output_dir: str = None) -> Dict[str, str]:
        """
        Generate visualization plots for the evaluation results.
        
        Args:
            results: List of evaluation results
            metrics: Computed metrics
            output_dir: Output directory for plots
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        # Enhanced data validation
        if not results or not isinstance(results, list):
            print("Warning: No valid results provided for plotting")
            return {}
        
        if not metrics or not isinstance(metrics, dict):
            print("Warning: No valid metrics provided for plotting")
            return {}
        
        if output_dir is None:
            output_dir = self.plots_path
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_files = {}
        
        # Set style with fallback for compatibility
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')
        
        try:
            sns.set_palette("husl")
        except Exception:
            pass  # Use default palette if husl is not available
        
        # 1. Model Performance Comparison
        if 'per_model' in metrics and metrics['per_model']:
            try:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle('Model Performance Comparison', fontsize=16)
                
                models = list(metrics['per_model'].keys())
                
                # Enhanced data validation for models
                if not models or len(models) == 0:
                    print("Warning: No valid models found for performance comparison")
                    plt.close(fig)
                else:
                    # Classification metrics
                    classification_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
                    classification_values = {}
                    
                    for metric in classification_metrics:
                        values = []
                        for model in models:
                            try:
                                value = metrics['per_model'][model]['classification'][metric]
                                # Enhanced validation for metric values
                                if isinstance(value, (int, float)) and not np.isnan(value) and 0 <= value <= 1:
                                    values.append(value)
                                else:
                                    values.append(0)
                            except (KeyError, TypeError):
                                values.append(0)
                        classification_values[metric] = values
                    
                    # Plot classification metrics
                    x = np.arange(len(models))
                    width = 0.2
                    
                    for i, metric in enumerate(classification_metrics):
                        axes[0, 0].bar(x + i * width, classification_values[metric], width, label=metric)
                    
                    axes[0, 0].set_title('Classification Metrics')
                    axes[0, 0].set_xlabel('Models')
                    axes[0, 0].set_ylabel('Score')
                    axes[0, 0].set_xticks(x + width * 1.5)
                    # Enhanced Y-axis limits for metrics
                    axes[0, 0].set_ylim(0, 1.1)  # Slightly above 1 for better visualization
                    axes[0, 0].grid(True, alpha=0.3)
                    
                    # Robust model name shortening
                    shortened_names = []
                    for model in models:
                        if isinstance(model, str):
                            if model.startswith('gemma-'):
                                shortened_names.append(model.replace('gemma-', ''))
                            elif len(model) > 15:  # Truncate very long names
                                shortened_names.append(model[:12] + '...')
                            else:
                                shortened_names.append(model)
                        else:
                            shortened_names.append(str(model)[:15])
                    
                    axes[0, 0].set_xticklabels(shortened_names, rotation=45)
                    axes[0, 0].legend()
                    
                    # F1 Score comparison (highlight most important metric)
                    f1_scores = classification_values['f1_score']
                    axes[0, 1].bar(range(len(models)), f1_scores, color='lightblue')
                    axes[0, 1].set_title('F1-Score Comparison')
                    axes[0, 1].set_xlabel('Models')
                    axes[0, 1].set_ylabel('F1-Score')
                    axes[0, 1].set_xticks(range(len(models)))
                    axes[0, 1].set_xticklabels(shortened_names, rotation=45)
                    axes[0, 1].set_ylim(0, 1.1)
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    # Add value labels on F1 bars
                    for i, v in enumerate(f1_scores):
                        axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
                    
                    # Processing time comparison
                    processing_times = []
                    for model in models:
                        try:
                            time_val = metrics['per_model'][model]['average_processing_time']
                            # Enhanced validation for processing times
                            if isinstance(time_val, (int, float)) and time_val >= 0 and not np.isnan(time_val):
                                processing_times.append(time_val)
                            else:
                                processing_times.append(0)
                        except (KeyError, TypeError):
                            processing_times.append(0)
                    
                    axes[1, 0].bar(range(len(models)), processing_times)
                    axes[1, 0].set_title('Average Processing Time')
                    axes[1, 0].set_xlabel('Models')
                    axes[1, 0].set_ylabel('Time (seconds)')
                    axes[1, 0].set_xticks(range(len(models)))
                    axes[1, 0].set_xticklabels(shortened_names, rotation=45)
                    # Enhanced Y-axis limits for processing times
                    if processing_times and max(processing_times) > 0:
                        axes[1, 0].set_ylim(0, max(processing_times) * 1.1)
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # Sample count per model
                    sample_counts = []
                    for model in models:
                        try:
                            count = metrics['per_model'][model]['count']
                            # Enhanced validation for sample counts
                            if isinstance(count, int) and count >= 0:
                                sample_counts.append(count)
                            else:
                                sample_counts.append(0)
                        except (KeyError, TypeError):
                            sample_counts.append(0)
                    
                    axes[1, 1].bar(range(len(models)), sample_counts)
                    axes[1, 1].set_title('Samples Evaluated per Model')
                    axes[1, 1].set_xlabel('Models')
                    axes[1, 1].set_ylabel('Number of Samples')
                    axes[1, 1].set_xticks(range(len(models)))
                    axes[1, 1].set_xticklabels(shortened_names, rotation=45)
                    # Enhanced Y-axis limits for sample counts
                    if sample_counts and max(sample_counts) > 0:
                        axes[1, 1].set_ylim(0, max(sample_counts) * 1.1)
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plot_path = output_dir / "model_performance.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    plot_files['model_performance'] = str(plot_path)
                    
            except Exception as e:
                print(f"Warning: Could not generate model performance plot: {e}")
                if 'fig' in locals():
                    plt.close(fig)
        
        # 2. Confusion Matrix
        if 'confusion_matrix' in metrics and 'confusion_matrix_labels' in metrics:
            try:
                # Enhanced data validation for confusion matrix
                cm = np.array(metrics['confusion_matrix'])
                labels = metrics['confusion_matrix_labels']
                
                if cm.size > 0 and labels and len(labels) > 0:
                    # Validate confusion matrix dimensions
                    if cm.shape[0] == cm.shape[1] == len(labels):
                        plt.figure(figsize=(max(8, len(labels) * 1.2), max(6, len(labels) * 1.0)))
                        
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                   xticklabels=labels, yticklabels=labels,
                                   cbar_kws={'label': 'Count'})
                        plt.title('Confusion Matrix (All Models)')
                        plt.xlabel('Predicted')
                        plt.ylabel('Actual')
                        plt.tight_layout()
                        
                        plot_path = output_dir / "confusion_matrix.png"
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        plot_files['confusion_matrix'] = str(plot_path)
                    else:
                        print(f"Warning: Confusion matrix dimensions mismatch: {cm.shape} vs {len(labels)} labels")
                        plt.close()
                else:
                    print("Warning: Empty confusion matrix or labels")
                    plt.close()
                    
            except Exception as e:
                print(f"Warning: Could not generate confusion matrix plot: {e}")
                plt.close()
        
        # 3. Error Analysis
        try:
            # Enhanced data validation for results
            valid_results = [r for r in results if isinstance(r, dict)]
            if not valid_results:
                print("Warning: No valid result dictionaries found for error analysis")
                return plot_files
                
            df = pd.DataFrame(valid_results)
            if not df.empty and 'ground_truth' in df.columns:
                plt.figure(figsize=(12, 6))
                
                # Subplot 1: Error rate by ground truth category
                plt.subplot(1, 2, 1)
                
                # Enhanced error rate calculation with validation
                if 'error' in df.columns:
                    # Convert error column to boolean with enhanced validation
                    df['error_bool'] = df['error'].apply(
                        lambda x: bool(x) if x is not None and x is not False else False
                    )
                    
                    # Group by ground truth and calculate error rates
                    error_by_category = df.groupby('ground_truth')['error_bool'].apply(
                        lambda x: x.sum() / len(x) if len(x) > 0 else 0
                    )
                    
                    if not error_by_category.empty and len(error_by_category) > 0:
                        error_by_category.plot(kind='bar', color='lightcoral')
                        plt.title('Error Rate by Ground Truth Category')
                        plt.xlabel('Category')
                        plt.ylabel('Error Rate')
                        plt.xticks(rotation=45)
                        # Enhanced Y-axis limits for error rate
                        plt.ylim(0, min(1.0, error_by_category.max() * 1.1))
                        plt.grid(True, alpha=0.3)
                        
                        # Add value labels on bars
                        for i, v in enumerate(error_by_category.values):
                            plt.text(i, v + 0.01, f'{v:.2%}', ha='center', va='bottom')
                    else:
                        plt.text(0.5, 0.5, 'No error data available', 
                                ha='center', va='center', transform=plt.gca().transAxes)
                        plt.title('Error Rate by Ground Truth Category')
                
                # Subplot 2: Processing time distribution
                plt.subplot(1, 2, 2)
                if 'processing_time' in df.columns:
                    # Enhanced processing time validation
                    valid_times = df['processing_time'].dropna()
                    valid_times = valid_times[
                        (valid_times >= 0) & 
                        (valid_times < np.inf) & 
                        (~np.isnan(valid_times))
                    ]
                    
                    if not valid_times.empty and len(valid_times) > 0:
                        # Dynamic binning based on data characteristics
                        n_samples = len(valid_times)
                        if n_samples < 10:
                            n_bins = n_samples
                        elif n_samples < 50:
                            n_bins = min(15, n_samples // 2)
                        else:
                            # Use Sturges' rule with modifications
                            n_bins = min(50, max(10, int(np.ceil(np.log2(n_samples) + 1))))
                        
                        # Enhanced histogram with better styling
                        counts, bins, patches = plt.hist(valid_times, bins=n_bins, alpha=0.7, 
                                                        color='skyblue', edgecolor='navy', linewidth=0.5)
                        plt.title('Processing Time Distribution')
                        plt.xlabel('Processing Time (seconds)')
                        plt.ylabel('Frequency')
                        
                        # Enhanced Y-axis limits for histogram
                        plt.ylim(0, max(counts) * 1.1)
                        plt.grid(True, alpha=0.3)
                        
                        # Add statistics text
                        mean_time = valid_times.mean()
                        median_time = valid_times.median()
                        plt.axvline(mean_time, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_time:.2f}s')
                        plt.axvline(median_time, color='green', linestyle='--', alpha=0.7, label=f'Median: {median_time:.2f}s')
                        plt.legend()
                        
                    else:
                        plt.text(0.5, 0.5, 'No valid processing times', 
                                ha='center', va='center', transform=plt.gca().transAxes)
                        plt.title('Processing Time Distribution')
                        plt.xlabel('Processing Time (seconds)')
                        plt.ylabel('Frequency')
                
                plt.tight_layout()
                plot_path = output_dir / "error_analysis.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files['error_analysis'] = str(plot_path)
            else:
                print("Warning: DataFrame is empty or missing 'ground_truth' column")
                
        except Exception as e:
            print(f"Warning: Could not generate error analysis plot: {e}")
            plt.close()
        
        return plot_files
    
    def generate_report(self, 
                       results: List[Dict[str, Any]], 
                       dataset_stats: Dict[str, Any],
                       output_file: str = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: List of evaluation results
            dataset_stats: Dataset statistics
            output_file: Output file path
            
        Returns:
            Path to the generated report
        """
        if output_file is None:
            output_file = self.evals_path / "evaluation_report.json"
        else:
            output_file = Path(output_file)
        
        # Compute metrics
        metrics = self.compute_metrics(results)
        
        # Generate plots
        plot_files = self.generate_plots(results, metrics)
        
        # Create comprehensive report
        report = {
            'evaluation_summary': {
                'total_evaluations': len(results),
                'successful_evaluations': metrics.get('successful_evaluations', 0),
                'error_rate': metrics.get('error_rate', 0),
                'models_evaluated': list(Config.GEMMA_MODELS),
                'prompt_types_used': list(Config.DETECTION_PROMPTS.keys()),
            },
            'dataset_statistics': dataset_stats,
            'performance_metrics': metrics,
            'generated_plots': plot_files,
            'configuration': {
                'batch_size': Config.BATCH_SIZE,
                'max_images_per_category': Config.MAX_IMAGES_PER_CATEGORY,
                'timeout_seconds': Config.TIMEOUT_SECONDS
            }
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Also generate a human-readable summary
        summary_file = output_file.with_suffix('.txt')
        self._generate_text_summary(report, summary_file)
        
        return str(output_file)
    
    def _generate_text_summary(self, report: Dict[str, Any], output_file: Path) -> None:
        """Generate a human-readable text summary of the evaluation."""
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("GEMMA FIRE/SMOKE DETECTION EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary
            summary = report['evaluation_summary']
            f.write("EVALUATION SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Evaluations: {summary['total_evaluations']}\n")
            f.write(f"Successful Evaluations: {summary['successful_evaluations']}\n")
            f.write(f"Error Rate: {summary['error_rate']:.2%}\n")
            f.write(f"Models Evaluated: {', '.join(summary['models_evaluated'])}\n\n")
            
            # Dataset Statistics
            if 'dataset_statistics' in report:
                stats = report['dataset_statistics']
                f.write("DATASET STATISTICS\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total Images: {stats.get('total_images', 'N/A')}\n")
                if 'categories' in stats:
                    for category, count in stats['categories'].items():
                        f.write(f"  {category.title()}: {count}\n")
                f.write("\n")
            
            # Performance Metrics
            if 'performance_metrics' in report:
                metrics = report['performance_metrics']
                f.write("OVERALL PERFORMANCE\n")
                f.write("-" * 20 + "\n")
                
                if 'classification' in metrics:
                    multi = metrics['classification']
                    f.write("Multi-class Classification:\n")
                    f.write(f"  Accuracy: {multi['accuracy']:.3f}\n")
                    f.write(f"  Precision: {multi['precision']:.3f}\n")
                    f.write(f"  Recall: {multi['recall']:.3f}\n")
                    f.write(f"  F1-Score: {multi['f1_score']:.3f}\n\n")
                
                # Per-model performance details
                f.write("\nDetailed Per-Model Performance:\n")
                f.write("=" * 40 + "\n")
                
                for model_name, model_metrics in metrics['per_model'].items():
                    f.write(f"\n{model_name}:\n")
                    f.write(f"  Evaluations: {model_metrics['count']}\n")
                    f.write(f"  Avg Time: {model_metrics['average_processing_time']:.2f}s\n")
                    
                    # Classification metrics
                    classification = model_metrics['classification']
                    f.write(f"  Classification Accuracy: {classification['accuracy']:.3f}\n")
                    f.write(f"  Classification F1-Score: {classification['f1_score']:.3f}\n")
                    f.write(f"  Classification Precision: {classification['precision']:.3f}\n")
                    f.write(f"  Classification Recall: {classification['recall']:.3f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("Report generated by GemmaEval Fire/Smoke Detection System\n")
            f.write("=" * 80 + "\n") 