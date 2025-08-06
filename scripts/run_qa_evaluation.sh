#!/bin/bash
# Example script to run pairwise evaluation for Gemma 3n models

set -e

# Configuration
DATASET_PATH="../survival_qa_dataset.json"  # Path to your Q&A dataset
OUTPUT_DIR="./evaluation_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Model paths - adjust these to your actual model locations
MODEL_BASELINE="baseline:./gemma3n_finetuned_unsloth_8bit_vanilla_final"
MODEL_IT="instruction_tuned:./gemma3n_finetuned_unsloth_8bit_it_final"
MODEL_E2B="e2b_tuned:./gemma3n_e2b_finetuned_unsloth_8bit_it_final"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "🚀 Starting Gemma 3n Pairwise Evaluation"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR/pairwise_results_$TIMESTAMP.json"
echo "Models: $MODEL_BASELINE, $MODEL_IT, $MODEL_E2B"
echo ""

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ Dataset not found: $DATASET_PATH"
    echo "Please run the data collection script first:"
    echo "cd .. && python utils_qa_dataset_collector.py --base_path /path/to/survival_dataset --repo_name your/dataset --local_only --output_path survival_qa_dataset.json"
    exit 1
fi

# Run the evaluation
python ../evaluate_qa_pairwise.py \
    --dataset "$DATASET_PATH" \
    --models "$MODEL_BASELINE" "$MODEL_IT" "$MODEL_E2B" \
    --judges "gpt-4o" "gemma-3-27b-it" \
    --output "$OUTPUT_DIR/pairwise_results_$TIMESTAMP.json" \
    --sample_size 100 \
    --seed 42

echo ""
echo "✅ Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR/pairwise_results_$TIMESTAMP.json"
echo ""
echo "📊 Key metrics computed:"
echo "• Win Rate (WR): w/(w+ℓ) - ignoring ties"
echo "• Net Preference: (w-ℓ)/N - including ties" 
echo "• 95% Wilson Score Interval for confidence"
echo "• Binomial sign test for statistical significance"
echo ""
echo "To analyze results further, you can:"
echo "1. View the JSON file for detailed comparisons"
echo "2. Run additional analysis scripts" 
echo "3. Compare with other evaluation metrics"