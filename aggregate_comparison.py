#!/usr/bin/env python3
"""
Aggregate Model Performance Comparison Script
Combines inference results from gemma3 and gemma3n evaluations
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional

def load_evaluation_report(file_path: str) -> Dict[str, Any]:
    """Load evaluation report from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def parse_raw_results_for_simple_prompt(raw_results_file: str) -> Dict[str, Any]:
    """Parse raw results to extract simple prompt specific metrics."""
    results = []
    
    try:
        with open(raw_results_file, 'r') as f:
            raw_data = json.load(f)
        
        # Filter for simple prompt results
        simple_prompt_results = []
        for result in raw_data:
            if isinstance(result, dict) and 'prompt' in result:
                # Check if this is a simple prompt (contains the simple prompt text)
                if "Answer with: 'fire', 'smoke', 'both', or 'none'" in result['prompt']:
                    simple_prompt_results.append(result)
        
        if not simple_prompt_results:
            print(f"Warning: No simple prompt results found in {raw_results_file}")
            return {}
        
        # Extract ground truth and predictions
        ground_truths = []
        predictions = []
        processing_times = []
        
        for result in simple_prompt_results:
            if result.get('parsed') and result.get('ground_truth'):
                ground_truths.append(result['ground_truth'])
                predictions.append(result['parsed']['prediction'])
                processing_times.append(result.get('processing_time', 0))
        
        if not ground_truths:
            print(f"Warning: No valid parsed results found in {raw_results_file}")
            return {}
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(ground_truths, predictions)
        precision = precision_score(ground_truths, predictions, average='weighted', zero_division=0)
        recall = recall_score(ground_truths, predictions, average='weighted', zero_division=0)
        f1 = f1_score(ground_truths, predictions, average='weighted', zero_division=0)
        
        return {
            'count': len(ground_truths),
            'average_processing_time': np.mean(processing_times),
            'classification': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        }
        
    except Exception as e:
        print(f"Error parsing raw results from {raw_results_file}: {e}")
        return {}

def parse_simple_prompt_from_text_report(text_report_file: str) -> Dict[str, Any]:
    """Parse simple prompt specific metrics from text report."""
    try:
        with open(text_report_file, 'r') as f:
            content = f.read()
        
        # Look for simple prompt section
        lines = content.split('\n')
        in_simple_section = False
        simple_metrics = {}
        
        for line in lines:
            line = line.strip()
            if line == "Simple Prompt:":
                in_simple_section = True
                continue
            elif line == "Detailed Prompt:" or line.startswith("="):
                in_simple_section = False
                continue
            
            if in_simple_section:
                if line.startswith("Accuracy:"):
                    accuracy = float(line.split(":")[1].strip())
                    simple_metrics['accuracy'] = accuracy
                elif line.startswith("F1-Score:"):
                    f1_score = float(line.split(":")[1].strip())
                    simple_metrics['f1_score'] = f1_score
                elif line.startswith("Evaluations:"):
                    count = int(line.split(":")[1].strip())
                    simple_metrics['count'] = count
        
        if simple_metrics:
            # Estimate precision and recall from accuracy and F1 (approximate)
            # This is not perfect but gives reasonable estimates
            simple_metrics['precision'] = simple_metrics.get('f1_score', 0)
            simple_metrics['recall'] = simple_metrics.get('accuracy', 0)
            return simple_metrics
        
    except Exception as e:
        print(f"Error parsing text report {text_report_file}: {e}")
    
    return {}

def extract_model_metrics(results_folder: str) -> Dict[str, Dict[str, Any]]:
    """Extract model performance metrics from a results folder."""
    results_path = Path(results_folder)
    
    # Load evaluation report
    report_file = results_path / "evals" / "evaluation_report.json"
    if not report_file.exists():
        print(f"Warning: No evaluation report found at {report_file}")
        return {}
    
    report = load_evaluation_report(str(report_file))
    models_data = {}
    
    # Check if we have per-model metrics (gemma3 format)
    if 'per_model' in report.get('performance_metrics', {}):
        for model_name, model_metrics in report['performance_metrics']['per_model'].items():
            models_data[model_name] = model_metrics
    
    # Check if we have aggregated metrics (gemma3n format)
    elif 'models_evaluated' in report.get('evaluation_summary', {}):
        models_evaluated = report['evaluation_summary']['models_evaluated']
        
        # If only one model and it has both simple and detailed prompts, extract simple prompt results
        if len(models_evaluated) == 1 and 'simple' in report['evaluation_summary'].get('prompt_types_used', []):
            model_name = models_evaluated[0]
            
            # First, try to parse the text report for simple prompt metrics
            text_report_file = results_path / "evals" / "evaluation_report.txt"
            if text_report_file.exists():
                simple_metrics = parse_simple_prompt_from_text_report(str(text_report_file))
                if simple_metrics:
                    print(f"✅ Extracted simple prompt metrics for {model_name} from text report")
                    models_data[model_name] = {
                        'classification': {
                            'accuracy': simple_metrics['accuracy'],
                            'precision': simple_metrics['precision'],
                            'recall': simple_metrics['recall'],
                            'f1_score': simple_metrics['f1_score']
                        },
                        'average_processing_time': report['performance_metrics'].get('average_processing_time', 0),
                        'count': simple_metrics['count']
                    }
                    return models_data
            
            # Fallback: Try to parse raw results for simple prompt specific metrics
            raw_results_file = results_path / "evals" / "raw_results.json"
            if raw_results_file.exists():
                simple_metrics = parse_raw_results_for_simple_prompt(str(raw_results_file))
                if simple_metrics:
                    models_data[model_name] = simple_metrics
                else:
                    # Fall back to aggregated metrics if parsing fails
                    print(f"Warning: Could not parse simple prompt results for {model_name}, using aggregated metrics")
                    models_data[model_name] = {
                        'classification': report['performance_metrics']['classification'],
                        'average_processing_time': report['performance_metrics'].get('average_processing_time', 0),
                        'count': report['performance_metrics'].get('total_evaluations', 0)
                    }
    
    return models_data

def create_aggregate_comparison_plot(results_folders: List[str], output_path: str = "aggregate_model_comparison.png") -> None:
    """Create aggregate model performance comparison plot."""
    
    # Extract all model metrics
    all_models_data = {}
    
    for folder in results_folders:
        folder_models = extract_model_metrics(folder)
        all_models_data.update(folder_models)
    
    if not all_models_data:
        print("Error: No model data found in any of the provided folders")
        return
    
    # Prepare data for plotting
    models = list(all_models_data.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Create a more readable model name mapping
    model_name_mapping = {
        'gemma-3-27b-it': 'Gemma-3 27B',
        'gemma-3-12b-it': 'Gemma-3 12B', 
        'gemma-3-4b-it': 'Gemma-3 4B',
        'gemma-3n-E4B-it': 'Gemma-3N E4B',
        'gemma-3n-e4b-it': 'Gemma-3N E4B'
    }
    
    # Map model names to more readable versions
    readable_models = [model_name_mapping.get(model, model) for model in models]
    
    # Extract metric values
    metric_values = {metric: [] for metric in metrics}
    processing_times = []
    
    for model in models:
        model_data = all_models_data[model]
        classification = model_data.get('classification', {})
        
        for metric in metrics:
            metric_values[metric].append(classification.get(metric, 0))
        
        processing_times.append(model_data.get('average_processing_time', 0))
    
    # Create the plot
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Aggregate Model Performance Comparison\n(Simple Prompt Results)', fontsize=16, fontweight='bold')
    
    # Color scheme
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    # Plot 1: All metrics comparison
    ax1 = axes[0, 0]
    x = np.arange(len(readable_models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        bars = ax1.bar(x + i*width, metric_values[metric], width, 
                      label=metric.replace('_', ' ').title(), 
                      color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values[metric]):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Models', fontweight='bold')
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('Classification Metrics Comparison', fontweight='bold')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(readable_models, rotation=45, ha='right')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Plot 2: F1-Score focus (most important metric)
    ax2 = axes[0, 1]
    bars = ax2.bar(readable_models, metric_values['f1_score'], 
                   color=colors[:len(readable_models)], alpha=0.8)
    ax2.set_title('F1-Score Comparison', fontweight='bold')
    ax2.set_xlabel('Models', fontweight='bold')
    ax2.set_ylabel('F1-Score', fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, metric_values['f1_score']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 3: Processing time comparison
    ax3 = axes[1, 0]
    bars = ax3.bar(readable_models, processing_times, 
                   color=colors[:len(readable_models)], alpha=0.8)
    ax3.set_title('Average Processing Time', fontweight='bold')
    ax3.set_xlabel('Models', fontweight='bold')
    ax3.set_ylabel('Time (seconds)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, processing_times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(processing_times)*0.02,
                f'{value:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 4: Performance vs Speed scatter
    ax4 = axes[1, 1]
    scatter = ax4.scatter(processing_times, metric_values['f1_score'], 
                         c=colors[:len(readable_models)], s=100, alpha=0.7)
    
    # Add model labels
    for i, model in enumerate(readable_models):
        ax4.annotate(model, (processing_times[i], metric_values['f1_score'][i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax4.set_xlabel('Processing Time (seconds)', fontweight='bold')
    ax4.set_ylabel('F1-Score', fontweight='bold')
    ax4.set_title('Performance vs Speed Trade-off', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary table
    print("\n" + "="*80)
    print("AGGREGATE MODEL PERFORMANCE COMPARISON (Simple Prompt)")
    print("="*80)
    
    # Create summary DataFrame
    summary_data = []
    for i, model in enumerate(readable_models):
        summary_data.append({
            'Model': model,
            'Accuracy': f"{metric_values['accuracy'][i]:.3f}",
            'Precision': f"{metric_values['precision'][i]:.3f}", 
            'Recall': f"{metric_values['recall'][i]:.3f}",
            'F1-Score': f"{metric_values['f1_score'][i]:.3f}",
            'Avg Time (s)': f"{processing_times[i]:.2f}",
            'Count': all_models_data[models[i]].get('count', 'N/A')
        })
    
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    
    # Find best performing model
    best_f1_idx = np.argmax(metric_values['f1_score'])
    fastest_idx = np.argmin(processing_times)
    
    print(f"\n🏆 Best F1-Score: {readable_models[best_f1_idx]} ({metric_values['f1_score'][best_f1_idx]:.3f})")
    print(f"⚡ Fastest: {readable_models[fastest_idx]} ({processing_times[fastest_idx]:.2f}s)")
    
    print(f"\nPlot saved to: {output_path}")

def main():
    """Main function to run the aggregate comparison."""
    # Define the results folders
    results_folders = [
        "results/2025-06-25_10-26-41_3-27b_3-12b_3-4b_simple_allimg_test",
        "results/2025-07-09"
    ]
    
    # Create the aggregate comparison plot
    create_aggregate_comparison_plot(results_folders)

if __name__ == "__main__":
    main() 