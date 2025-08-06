#!/bin/bash

# Run LoRA experiments with recommended parameters
echo "🔬 Running LoRA experiments with recommended parameters..."
echo "This will test r=[4, 8, 16, 32] with 60 training steps and 200 test samples"
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Run the experiments
python ../ablate_lora.py \
    --r-values 4 8 16 32 \
    --max-steps 100 \
    --max-test-samples 2000 \
    --output-dir mushroom_lora_experiments \
    --gpu-device 3

echo ""
echo "Experiment completed! Check mushroom_lora_experiments/ for results:"
echo "  - lora_performance_analysis.png (performance plots)"
echo "  - lora_experiment_results.csv (detailed metrics)"
echo "  - final_results.json (raw results)"