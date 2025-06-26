# GemmaScout: Fire/Smoke Detection Evaluation System

A comprehensive, automated benchmarking framework designed to evaluate Gemma language models on fire and smoke detection tasks using the D-Fire dataset.

🔗 **Project page**: https://summerofcode.withgoogle.com/programs/2025/projects/RSjjE3tM

## 🔥 Overview

This system evaluates the performance of different Gemma models (3-27B, 3-12B, 3-4B, etc.) on fire and smoke detection using visual analysis. It uses the [D-Fire dataset](https://github.com/gaiasd/DFireDataset) which contains over 21,000 images with fire, smoke, and no-fire/smoke scenarios.

### Key Features

- **Multi-Model Evaluation**: Test multiple Gemma model variants simultaneously
- **Flexible Prompting**: Support for different prompt strategies (simple, detailed)
- **Comprehensive Analysis**: Detailed metrics, visualizations, and reports
- **Modular Architecture**: Clean, extensible codebase with proper separation of concerns
- **Concurrent Processing**: Efficient parallel evaluation with configurable concurrency
- **Rich Reporting**: Generate both JSON reports and human-readable summaries

## 📊 Dataset Information

The D-Fire dataset contains:

| Category | # Images | Description |
|----------|----------|-------------|
| Only fire | 1,164 | Images containing only fire |
| Only smoke | 5,867 | Images containing only smoke |
| Fire and smoke | 4,658 | Images containing both fire and smoke |
| None | 9,838 | Images without fire or smoke |
| **Total** | **21,527** | **Complete dataset** |

### Label Format

The D-Fire dataset uses YOLO format labels:

- **Empty file**: No fire or smoke present (`none` category)
- **Class ID 0**: Smoke detection 
- **Class ID 1**: Fire detection
- **Format**: `class_id x_center y_center width height`
- **Coordinates**: Normalized between 0 and 1

Example label file content:
```
1 0.7295 0.6448 0.025 0.0405    # Fire detection
0 0.445 0.4984 0.878 0.9759     # Smoke detection
```

## 🚀 Quick Start

### 1. Setup

```bash
# Clone the repository
git clone <repository-url>
cd GemmaEval

# Run the setup script
python scripts/setup.py
```

### 2. Configuration

Edit the `.env` file and add your Gemini API key:

```bash
GEMINI_API_KEY=your_actual_api_key_here
```

### 3. Run Evaluation

```bash
# Quick test (recommended first run)
python main.py --quick-test

# Full evaluation
python main.py

# Custom evaluation
python main.py --models gemma-3-27b-it gemma-3-4b-it --prompt-types simple --max-images 50
```

## 📁 Project Structure

```
GemmaEval/
├── main.py                 # Main evaluation script
├── requirements.txt        # Python dependencies
├── env.example            # Environment variables template
├── src/                   # Source code modules
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── dataset.py         # D-Fire dataset handling
│   ├── evaluator.py       # Gemma model evaluation
│   └── analyzer.py        # Results analysis and reporting
├── scripts/
│   └── setup.py          # Setup and installation script
├── tests/
│   └── test_system.py    # Test suite
├── data/
│   └── D-Fire/           # D-Fire dataset (download separately)
├── results/              # Evaluation results and reports
└── logs/                 # System logs
```

## 🔧 Usage

### Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --models MODEL [MODEL ...]        # Specific models to evaluate
  --prompt-types TYPE [TYPE ...]    # Prompt types: simple, detailed
  --max-images INT                  # Max images per category (default: 100)
  --max-workers INT                 # Concurrent evaluations (default: 5)
  --output-dir DIR                  # Output directory (default: results)
  --dataset-split {train,test}      # Dataset split to use (default: test)
  --quick-test                      # Run minimal test
  --analyze-only FILE               # Analyze existing results
  --help                           # Show help message
```

### Examples

```bash
# Test specific models with simple prompts
python main.py --models gemma-3-27b-it gemma-3-4b-it --prompt-types simple

# Quick evaluation with minimal images
python main.py --quick-test --max-images 10

# Full evaluation with custom output directory
python main.py --output-dir my_results --max-workers 8

# Analyze existing results
python main.py --analyze-only results/raw_results.json
```

## 📈 Evaluation Metrics

The system computes comprehensive metrics:

### Multi-class Classification
- Accuracy across all categories (fire, smoke, both, none)
- Detailed confusion matrices
- Per-category performance

### Performance Analysis
- Processing time per model
- Error rate analysis
- Scalability metrics

## 🛠️ Configuration

Configuration is managed through environment variables and the `Config` class:

### Environment Variables (.env)
```bash
# API Configuration
GEMINI_API_KEY=your_api_key

# Dataset Configuration  
DATASET_PATH=data/D-Fire
RESULTS_PATH=results

# Evaluation Configuration
BATCH_SIZE=10
MAX_IMAGES_PER_CATEGORY=100
TIMEOUT_SECONDS=30

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/evaluation.log
```

### Supported Models
- gemma-3-27b-it
- gemma-3-12b-it  
- gemma-3-4b-it
- gemma-2-27b-it
- gemma-2-9b-it
- gemma-2-2b-it

### Prompt Types

1. **Simple**: "Is there fire or smoke visible in this image? Answer with: 'fire', 'smoke', 'both', or 'none'."

2. **Detailed**: "Analyze this image for fire and smoke detection. Respond with JSON format: {'fire': true/false, 'smoke': true/false, 'confidence': 0-1}"

## 📊 Results and Reports

The system generates comprehensive reports:

### Raw Results (`raw_results.json`)
- Complete evaluation data for each image/model combination
- Response times, errors, parsed predictions
- Ground truth comparisons

### Analysis Report (`evaluation_report.json`)
- Computed metrics and statistics
- Performance comparisons
- Configuration details

### Text Summary (`evaluation_report.txt`)
- Human-readable summary
- Key findings and metrics
- Model comparisons

### Visualizations
- Model performance comparisons
- Confusion matrices
- Error analysis charts
- Processing time distributions

## 🧪 Testing

Run the test suite to verify system functionality:

```bash
# Run all tests
python tests/test_system.py

# Run specific test categories
python -m unittest tests.test_system.TestConfig
python -m unittest tests.test_system.TestEvaluator
```

# Full Dataset Evaluation

The evaluation system now supports testing on all available images from both train and test datasets, rather than being limited to 100 images per category from just the test set.

## New Command Line Options

### `--use-all-images`
Use all available images, ignoring any limits set by `--max-images`.

### `--dataset-splits`
Specify multiple dataset splits to evaluate on. Can be:
- `--dataset-splits train` (train only)
- `--dataset-splits test` (test only) 
- `--dataset-splits train test` (both train and test)

### `--max-images`
Set to 0 or negative value for unlimited images per category.

## Usage Examples

### Evaluate on ALL images from both train and test sets:
```bash
python main.py --use-all-images --dataset-splits train test
```

### Evaluate on ALL images from test set only:
```bash
python main.py --use-all-images --dataset-splits test
```

### Evaluate with specific limit across both train and test:
```bash
python main.py --max-images 500 --dataset-splits train test
```

### Use unlimited images (alternative to --use-all-images):
```bash
python main.py --max-images 0 --dataset-splits train test
```

### Quick test with multiple splits:
```bash
python main.py --quick-test --dataset-splits train test
```

## Performance Considerations

- **Memory Usage**: Loading all images will require significantly more memory
- **Processing Time**: Full dataset evaluation can take hours depending on:
  - Number of models being evaluated
  - API rate limits
  - Dataset size
- **API Costs**: Evaluate all images will result in many more API calls

## Async Processing

For better performance with large datasets, use the async evaluator:

```bash
python main.py --use-all-images --dataset-splits train test --use-async --max-concurrent 20
```

## Output Organization

Results will be organized with clear labeling of the dataset configuration:
- Folder names include split information (e.g., `train_test_all_simple`)
- Image names are prefixed with split information to avoid conflicts
- Combined statistics show totals across all splits

## Example Full Command

```bash
python main.py \
    --use-all-images \
    --dataset-splits train test \
    --models gemma-3-27b-it gemma-3-12b-it \
    --prompt-types simple detailed \
    --use-async \
    --max-concurrent 15 \
    --max-workers 10
```

This will:
- Use ALL images from both train and test splits
- Evaluate with the two largest Gemma models
- Use both simple and detailed prompts
- Use async processing for better performance
- Limit to 15 concurrent requests to respect API limits 

## 📋 Requirements

### Python Dependencies
- google-genai >= 0.8.0
- python-dotenv >= 1.0.0
- Pillow >= 10.0.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- scikit-learn >= 1.3.0
- rich >= 13.0.0

### System Requirements
- Python 3.8+
- Internet connection for Gemini API
- Sufficient disk space for dataset and results

## 🔒 API Usage and Costs

This system uses the Gemini API for model evaluation:

- **API Key Required**: Obtain from Google AI Studio
- **Usage Tracking**: Each image evaluation counts as one API call
- **Cost Estimation**: Varies by model size and number of evaluations
- **Rate Limiting**: Configurable concurrent workers to manage API usage

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

## 🙏 Acknowledgments

- **D-Fire Dataset**: [Gaia Solutions](https://github.com/gaiasd/DFireDataset)
- **Google Summer of Code 2025**
- **DeepMind Gemma Team**

## Support

1. Run `python main.py --help` for usage information
2. Run `python scripts/setup.py` to verify configuration

---

**Made with 🔥 for Google Summer of Code 2025**
