#!/bin/bash
# Fire/Smoke Detection Evaluation Script
# Runs the main Gemma Scout evaluation on the D-Fire dataset

set -e

# Configuration
DATASET_PATH="./data/D-Fire"
OUTPUT_DIR="./results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Default parameters
MODELS="gemma-3-27b-it gemma-3-12b-it gemma-3-4b-it"
PROMPT_TYPES="simple detailed"
MAX_IMAGES=100
MAX_WORKERS=5
QUICK_TEST=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick-test)
            QUICK_TEST=true
            MAX_IMAGES=10
            shift
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --max-images)
            MAX_IMAGES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Fire/Smoke Detection Evaluation Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick-test        Run quick test with 10 images"
            echo "  --models MODELS     Space-separated list of models"
            echo "  --max-images N      Maximum images per category"
            echo "  --output-dir DIR    Output directory"
            echo "  --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --quick-test"
            echo "  $0 --models 'gemma-3-27b-it' --max-images 50"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "🔥 Starting Fire/Smoke Detection Evaluation with Gemma Scout"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR/evaluation_$TIMESTAMP"
echo "Models: $MODELS"
echo "Prompt types: $PROMPT_TYPES"
echo "Max images per category: $MAX_IMAGES"
echo "Quick test: $QUICK_TEST"
echo ""

# Check if dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ D-Fire dataset not found: $DATASET_PATH"
    echo "Please download the D-Fire dataset first:"
    echo "1. Visit: https://github.com/gaiasd/DFireDataset"
    echo "2. Download and extract to $DATASET_PATH"
    exit 1
fi

# Check if API key is configured
if [ ! -f ".env" ]; then
    echo "❌ Environment file not found. Please create .env file with your API key:"
    echo "cp env.example .env"
    echo "Edit .env and add your GEMINI_API_KEY"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "🔄 Activating virtual environment..."
    source .venv/bin/activate
fi

# Build command arguments
ARGS="--output-dir $OUTPUT_DIR"
if [ "$QUICK_TEST" = true ]; then
    ARGS="$ARGS --quick-test"
fi
if [ -n "$MAX_IMAGES" ]; then
    ARGS="$ARGS --max-images $MAX_IMAGES"
fi

# Add models if specified
if [ -n "$MODELS" ]; then
    ARGS="$ARGS --models $MODELS"
fi

# Run the evaluation
echo "🚀 Running evaluation..."
python ../main.py $ARGS

echo ""
echo "✅ Fire/Smoke detection evaluation complete!"
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "📊 Generated files:"
echo "• raw_results.json - Complete evaluation data"
echo "• evaluation_report.json - Computed metrics and statistics"
echo "• evaluation_report.txt - Human-readable summary"
echo "• Visualizations and performance charts"
echo ""
echo "🔍 Next steps:"
echo "1. Review the evaluation report"
echo "2. Analyze model performance comparisons"
echo "3. Consider fine-tuning based on results"