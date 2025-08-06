# GemmaScout: Multi-Domain Vision-Language Model Training & Evaluation

A comprehensive framework for training and evaluating Gemma language models across multiple vision domains: **fire/smoke detection**, **mushroom identification**, and **plant species recognition**. Features Augmentoolkit integration for synthetic data generation and advanced fine-tuning capabilities.

🔗 **Project page**: https://summerofcode.withgoogle.com/programs/2025/projects/RSjjE3tM

## 🔥 Overview

GemmaScout is an advanced multi-domain system that combines vision-language model fine-tuning with real-world classification tasks. The system leverages **Augmentoolkit** for synthetic data generation and **Unsloth** for efficient LoRA-based fine-tuning of Gemma models across three distinct domains:

1. **🔥 Fire/Smoke Detection**: Safety-critical detection using D-Fire dataset
2. **🍄 Mushroom Identification**: Genus-level fungal classification 
3. **🌱 Plant Species Recognition**: Wild edible and medicinal plant identification

### Key Features

- **🤖 Augmentoolkit Integration**: Generate high-quality synthetic training data for enhanced model performance
- **⚡ Efficient Fine-tuning**: LoRA-based training with Unsloth for faster, memory-efficient model adaptation
- **📊 Multi-Domain Evaluation**: Test models across fire detection, mushroom ID, and plant recognition
- **🔬 LoRA Ablation Studies**: Analyze the relationship between LoRA rank and model performance
- **🎯 Interactive Inference**: Real-time chat interface for testing fine-tuned models
- **📈 Comprehensive Analysis**: Detailed metrics, visualizations, and pairwise evaluations

## 📊 Dataset Information

### 🔥 D-Fire Dataset (Fire/Smoke Detection)
The D-Fire dataset contains over 21,000 images for fire and smoke detection:

| Category | # Images | Description |
|----------|----------|-------------|
| Only fire | 1,164 | Images containing only fire |
| Only smoke | 5,867 | Images containing only smoke |
| Fire and smoke | 4,658 | Images containing both fire and smoke |
| None | 9,838 | Images without fire or smoke |
| **Total** | **21,527** | **Complete dataset** |

**Label Format**: YOLO format with class 0 (smoke) and class 1 (fire)

### 🍄 Mushroom Dataset (Genus Identification)
Comprehensive mushroom genus classification dataset with 6,714 images:

| Genus | Training | Test | Total | Description |
|-------|----------|------|-------|-------------|
| Lactarius | 1,251 | 312 | 1,563 | Milk-producing mushrooms |
| Russula | 919 | 229 | 1,148 | Brittle-fleshed mushrooms |
| Boletus | 859 | 214 | 1,073 | Pore-bearing mushrooms |
| Cortinarius | 669 | 167 | 836 | Cobweb-veiled mushrooms |
| Amanita | 600 | 150 | 750 | Often toxic mushrooms |
| Entoloma | 292 | 72 | 364 | Pink-spored mushrooms |
| Agaricus | 283 | 70 | 353 | Button/portobello type |
| Hygrocybe | 253 | 63 | 316 | Waxy cap mushrooms |
| Suillus | 249 | 62 | 311 | Sticky cap boletes |

**Features**: Detailed genus descriptions, safety information, habitat data

### 🌱 Plant Dataset (Species Recognition)
Wild edible and medicinal plant identification with 6,868 images across 55+ species:

| Category | Training | Test | Total | Use Cases |
|----------|----------|------|-------|-----------|
| Dandelion | 847 | 211 | 1,058 | Edible greens, medicinal |
| Daisy Fleabane | 620 | 155 | 775 | Medicinal, insect repellent |
| Sunflower | 592 | 148 | 740 | Seeds, oil, survival food |
| Chickweed | 124 | 31 | 155 | Edible greens, medicinal |
| Curly Dock | 124 | 31 | 155 | Edible, vitamin C source |
| Henbit | 124 | 31 | 155 | Edible flowers and leaves |
| Lambs Quarters | 124 | 31 | 155 | Nutritious edible greens |
| Peppergrass | 124 | 31 | 155 | Peppery edible seeds |
| **+47 more species** | | | | Various edible/medicinal uses |

**Features**: Survival/bushcraft focus, edibility information, medicinal properties

## 🧠 Augmentoolkit Integration

This system uses [Augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit) to generate synthetic training data that enhances model performance across all domains.

### Supported Data Types

- **Pretraining Data**: Raw text in `{"text": "..."}` format for domain adaptation
- **Factual SFT**: Conversations in ShareGPT format for instruction following
- **RAG Data**: Segmented format with loss masking for retrieval-augmented generation
- **Multimodal Data**: Vision-language pairs for image understanding
- **Correction Data**: Error correction pairs with masked incorrect answers
- **Chain-of-thought**: Reasoning conversations for improved analytical capabilities

### Multi-stage Training Pipeline

1. **Pretrain Stage**: Domain adaptation using raw text data
   - Fire safety manuals, emergency procedures
   - Mycological texts, mushroom identification guides
   - Botanical knowledge, foraging manuals, survival guides
2. **SFT Stage**: Instruction tuning using conversational data
   - Factual SFT, RAG, correction, and generic conversation datasets

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd GemmaEval

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 2. Download Datasets

```bash
# D-Fire dataset (fire/smoke detection)
wget https://github.com/gaiasd/DFireDataset/archive/refs/heads/main.zip
unzip main.zip -d data/
mv data/DFireDataset-main data/D-Fire

# Mushroom and plant datasets should be in:
# data/mushrooms/ and data/plants/
```

### 3. Quick Evaluation Tests

```bash
# Fire/smoke detection evaluation
./scripts/run_fire_smoke_evaluation.sh --quick-test

# LoRA ablation study (mushroom dataset)
./scripts/run_lora_ablation.sh

# Interactive inference
python inference_interactive_chat.py
```

## 📁 Project Structure

```
GemmaEval/
├── 🔥 FIRE/SMOKE DETECTION
│   ├── main.py                    # Main evaluation script for D-Fire dataset
│   └── scripts/
│       └── run_fire_smoke_evaluation.sh  # Fire/smoke detection runner
│
├── 🧠 TRAINING & FINE-TUNING
│   ├── finetune_gemma.py          # Augmentoolkit fine-tuning pipeline
│   ├── ablate_lora.py             # LoRA ablation experiments (mushroom)
│   └── scripts/
│       └── run_lora_ablation.sh   # LoRA experiment runner
│
├── 🎯 INFERENCE & EVALUATION
│   ├── inference_gemma3n.py       # Core inference module
│   ├── inference_interactive_chat.py  # Interactive chat interface
│   ├── evaluate_qa_pairwise.py    # Pairwise model evaluation
│   └── scripts/
│       └── run_qa_evaluation.sh   # QA evaluation runner
│
├── 🛠️ UTILITIES
│   ├── utils_qa_dataset_collector.py  # Dataset collection utilities
│   └── scripts/
│       ├── setup.py               # Environment setup
│       └── demo.py               # Demo and testing
│
├── 📦 CORE MODULES
│   └── src/
│       ├── config.py              # Configuration management
│       ├── dataset.py             # D-Fire dataset handling
│       ├── evaluator.py           # Model evaluation logic
│       └── analyzer.py            # Results analysis
│
├── 📊 DATA & RESULTS
│   ├── data/
│   │   ├── D-Fire/               # Fire/smoke detection dataset
│   │   ├── mushrooms/            # Mushroom genus classification
│   │   └── plants/               # Plant species identification
│   ├── results/                  # Evaluation results
│   └── logs/                     # System logs
│
└── 🧪 TESTING & CONFIG
    ├── tests/                    # Test suite
    ├── notebooks/                # Jupyter notebooks
    ├── requirements.txt          # Dependencies
    ├── env.example              # Environment template
    └── gemma3n_config_unsloth.yaml  # Training configuration
```

## 🎯 Usage Examples

### Fire/Smoke Detection Evaluation

```bash
# Quick test with default models
./scripts/run_fire_smoke_evaluation.sh --quick-test

# Full evaluation with specific models
./scripts/run_fire_smoke_evaluation.sh --models "gemma-3-27b-it gemma-3-12b-it"

# Custom configuration
python main.py \
    --models gemma-3-27b-it gemma-3-4b-it \
    --prompt-types simple detailed \
    --max-images 200 \
    --use-all-images \
    --dataset-splits train test
```

### Multi-Domain Fine-tuning with Augmentoolkit

```bashå
python finetune_gemma.py \
    --use_lora \
    --lora_r 64 \
    --max_steps 1000
```

### LoRA Ablation Studies

```bash
# Run mushroom genus classification ablation
./scripts/run_lora_ablation.sh

# Custom LoRA experiment
python ablate_lora.py \
    --r-values 4 8 16 32 64 128 256 \
    --max-steps 500 \
    --dataset-path "data/mushrooms/train/" \
    --test-dataset-path "data/mushrooms/test/" \
    --output-dir mushroom_lora_ablation
```

### Interactive Multi-Domain Inference

```bash
# Start interactive chat with fine-tuned model
python inference_interactive_chat.py

# Example multimodal usage:
# > image:/path/to/fire_image.jpg Is there fire in this image?
# > image:/path/to/mushroom.jpg What genus is this mushroom?
# > image:/path/to/plant.jpg Is this plant edible?
# > What safety measures should be taken for fire emergencies?
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
# API Configuration
GEMINI_API_KEY=your_api_key
OPENAI_API_KEY=your_api_key
```

### Supported Models
- **Gemma 3n Series**: gemma-3n-e2b-it, gemma-3n-34b-it
- **Gemma 3 Series**: gemma-3-4b-it, gemma-3-12b-it, gemma-3-27b-it

## 🙏 Acknowledgments

- **Augmentoolkit**: [E.P. Armstrong](https://github.com/e-p-armstrong/augmentoolkit) - Synthetic data generation
- **Unsloth**: [Unsloth AI](https://github.com/unslothai/unsloth) - Efficient fine-tuning
- **Google Summer of Code 2025**
- **DeepMind Gemma Team**

---

**Made with 🔥🍄🌱 for Google Summer of Code 2025 **