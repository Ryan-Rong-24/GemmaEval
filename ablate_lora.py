#!/usr/bin/env python3
"""
GemmaScout LoRA Ablation Studies Script

This script systematically tests different LoRA rank values to analyze the relationship
between LoRA rank and model performance on fire/smoke detection tasks.

Features:
- Automated LoRA rank ablation (r=4,8,16,32,64...)
- Performance curve generation and analysis
- Memory and training efficiency metrics
- Mushroom dataset for controlled experiments
- Integration with Unsloth for optimized training

Usage:
    python ablate_lora.py --r-values 4 8 16 32 64 128 256 --max-steps 500
    
Output:
- Performance analysis plots
- CSV results with detailed metrics  
- Training efficiency comparisons
- Memory usage analysis
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import argparse
import logging
from typing import List, Dict, Tuple, Optional
import warnings
import hashlib
import time
warnings.filterwarnings("ignore")

from unsloth import FastModel

import torch
from PIL import Image
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import TextStreamer
import glob

# Fix PyTorch Dynamo for Gemma 3N
import torch._dynamo
torch._dynamo.config.cache_size_limit = 1000
torch._dynamo.config.suppress_errors = True
os.environ['TORCH_LOGS'] = 'recompiles'
os.environ['TORCHDYNAMO_VERBOSE'] = '0'

# Unsloth and training imports
from trl import SFTTrainer, SFTConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants from the notebook
INCLUDE_DETAILED_RESPONSES = False  # Include detailed feature descriptions

IDENTIFICATION_PROMPTS = [
    "What type of mushroom is this? Please respond concisely.",
    "Can you identify this mushroom species? Please respond concisely.", 
    "What genus does this mushroom belong to? Please respond concisely.",
    "Please identify this fungus. Respond concisely.",
    "What kind of mushroom am I looking at? Please respond concisely.",
    "Help me identify this mushroom. Respond concisely."
]

# Mushroom genus descriptions (from prepare_mushroom_dataset.py)
MUSHROOM_DESCRIPTIONS = {
    "Agaricus": {
        "description": "Agaricus mushrooms are characterized by their white gills, white spores, and often have a ring around the stem and a bulbous base. Many species are edible.",
        "features": ["white gills", "white spores", "ring around stem", "bulbous base", "often edible"],
        "habitat": "Often found near trees, particularly oak and pine"
    },
    "Amanita": {
        "description": "Amanita mushrooms are characterized by their distinctive white gills, white spores, and often have a ring around the stem and a bulbous base. Many species are highly toxic.",
        "features": ["white gills", "white spores", "ring around stem", "bulbous base", "often toxic"],
        "habitat": "Often found near trees, particularly oak and pine"
    },
    "Boletus": {
        "description": "Boletus mushrooms have pores instead of gills underneath the cap, and the pore surface is typically yellow to olive. Most species are edible.",
        "features": ["pores instead of gills", "yellow to olive pore surface", "thick stem", "mostly edible"],
        "habitat": "Commonly found in forests, mycorrhizal with trees"
    },
    "Cortinarius": {
        "description": "Cortinarius mushrooms are characterized by their rusty brown spores and cobweb-like partial veil when young.",
        "features": ["rusty brown spores", "cobweb-like veil", "attached gills", "diverse colors"],
        "habitat": "Very diverse genus found in various forest environments"
    },
    "Lactarius": {
        "description": "Lactarius mushrooms are known for producing latex or milk when the flesh is broken. The color and taste of the latex can vary.",
        "features": ["produces latex/milk", "attached gills", "various colors", "milky discharge"],
        "habitat": "Commonly found near trees, mycorrhizal relationships"
    },
    "Russula": {
        "description": "Russula mushrooms have brittle flesh that crumbles easily, white to cream-colored spores, and often bright colored caps.",
        "features": ["brittle flesh", "white to cream spores", "bright colored caps", "crumbly texture"],
        "habitat": "Often found near trees, particularly oak and pine"
    }
}

class DatasetCache:
    """Cache system for processed datasets to avoid expensive HF Dataset operations."""
    
    def __init__(self, cache_dir: str = "./dataset_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_config_hash(self, config_dict: dict) -> str:
        """Generate hash from configuration to ensure cache validity."""
        # Sort keys to ensure consistent hashing
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _get_cache_path(self, config_hash: str) -> Path:
        """Get cache directory path for given configuration."""
        return self.cache_dir / f"dataset_{config_hash}"
    
    def _get_metadata_path(self, config_hash: str) -> Path:
        """Get metadata file path for given configuration."""
        return self.cache_dir / f"dataset_{config_hash}_metadata.json"
    
    def get_cache_config(self, dataset_path, max_images_per_genus, validation_split, 
                        include_detailed_responses, split="train"):
        """Create configuration dict for cache validation."""
        return {
            "dataset_path": str(dataset_path),
            "max_images_per_genus": max_images_per_genus,
            "validation_split": validation_split,
            "include_detailed_responses": include_detailed_responses,
            "split": split,
            "mushroom_descriptions": str(MUSHROOM_DESCRIPTIONS),
            "identification_prompts": str(IDENTIFICATION_PROMPTS)
        }
    
    def load_dataset(self, config_dict: dict) -> tuple:
        """Load cached dataset if available and valid."""
        config_hash = self._get_config_hash(config_dict)
        cache_path = self._get_cache_path(config_hash)
        metadata_path = self._get_metadata_path(config_hash)
        
        if cache_path.exists() and metadata_path.exists():
            print(f"📦 Loading processed dataset from cache: {cache_path}")
            try:
                # Load metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Load dataset using HF datasets
                dataset = Dataset.load_from_disk(str(cache_path))
                dataset_samples = metadata.get("dataset_samples", [])
                
                print(f"✅ Loaded cached dataset with {len(dataset)} samples")
                return dataset, dataset_samples
            except Exception as e:
                print(f"⚠️  Cache load failed: {e}")
                return None, None
        
        return None, None
    
    def save_dataset(self, config_dict: dict, dataset, dataset_samples: list = None):
        """Save processed dataset to cache."""
        config_hash = self._get_config_hash(config_dict)
        cache_path = self._get_cache_path(config_hash)
        metadata_path = self._get_metadata_path(config_hash)
        
        print(f"💾 Saving processed dataset to cache: {cache_path}")
        try:
            # Save dataset using HF datasets
            dataset.save_to_disk(str(cache_path))
            
            # Save metadata (without dataset_samples which can be large)
            metadata = {
                "config": config_dict,
                "timestamp": str(pd.Timestamp.now()),
                "dataset_size": len(dataset)
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Dataset cache saved successfully")
        except Exception as e:
            print(f"❌ Dataset cache save failed: {e}")
    
    def clear_cache(self):
        """Clear all cached datasets."""
        import shutil
        for cache_dir in self.cache_dir.glob("dataset_*"):
            if cache_dir.is_dir():
                shutil.rmtree(cache_dir)
        for metadata_file in self.cache_dir.glob("dataset_*_metadata.json"):
            metadata_file.unlink()
        print("🗑️  All dataset caches cleared")
    
    def list_caches(self):
        """List all available caches with their configurations."""
        metadata_files = list(self.cache_dir.glob("dataset_*_metadata.json"))
        if not metadata_files:
            print("No dataset caches found")
            return
        
        print(f"Found {len(metadata_files)} cached dataset sets:")
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                config = metadata.get("config", {})
                timestamp = metadata.get("timestamp", "Unknown")
                dataset_size = metadata.get("dataset_size", "Unknown")
                
                print(f"  📁 {metadata_file.stem}")
                print(f"     Dataset size: {dataset_size} samples")
                print(f"     Max images per genus: {config.get('max_images_per_genus', 'N/A')}")
                print(f"     Validation split: {config.get('validation_split', 'N/A')}")
                print(f"     Include detailed: {config.get('include_detailed_responses', 'N/A')}")
                print(f"     Split: {config.get('split', 'N/A')}")
                print(f"     Timestamp: {timestamp}")
                print()
            except Exception as e:
                print(f"  ❌ Error reading {metadata_file.name}: {e}")

# Configuration
class Config:
    """Configuration for the LoRA experiments"""
    def __init__(self):
        # Dataset paths
        self.dataset_path = "data/mushrooms/train/"
        self.test_dataset_path = "data/mushrooms/test/"
        
        # Model configuration
        self.model_name = "unsloth/gemma-3n-E2B-it"
        self.max_seq_length = 2048
        self.load_in_4bit = False
        self.load_in_8bit = True
        self.full_finetuning = False
        
        # Training configuration
        self.max_images_per_genus = 1400
        self.validation_split = 0.0
        self.use_validation = False
        self.include_detailed_responses = False  # Include detailed feature descriptions
        self.seed = 3407
        
        # LoRA experiment configuration
        self.r_values = [4, 8, 16, 32, 64]  # Different LoRA r values to test
        self.lora_alpha_ratio = 1.0  # alpha = r * ratio
        self.lora_dropout = 0.0
        
        # Training parameters
        self.per_device_train_batch_size = 1
        self.gradient_accumulation_steps = 4
        self.warmup_steps = 5
        self.max_steps = 60  # Reduced for faster experiments
        self.learning_rate = 2e-4
        self.weight_decay = 0.01
        self.logging_steps = 1
        self.eval_steps = 10
        
        # Evaluation configuration
        self.max_test_samples = 200  # Limit test samples for faster evaluation
        
        # Output configuration
        self.output_dir = "mushroom_lora_experiments"
        self.save_models = False  # Set to True to save trained models
        self.gpu_device = "3"  # GPU to use

class MushroomDatasetLoader:
    """Custom mushroom dataset loader for genus identification."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

    def load_dataset(self, max_per_genus: int = 200, validation_split: float = 0.2, seed: int = 42, split: str = "train") -> list:
        """Load dataset with specified number of images per genus and train/test split."""
        random.seed(seed)
        
        all_sample_data = []
        
        # Process each genus directory
        for genus_dir in self.dataset_path.iterdir():
            if not genus_dir.is_dir():
                continue
                
            genus_name = genus_dir.name
            print(f"Processing genus: {genus_name}")
            
            # Get all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(genus_dir.glob(ext))
            
            print(f"  Found {len(image_files)} images")
            
            # Shuffle files for this genus
            random.shuffle(image_files)
            
            # Limit samples per genus if specified
            if max_per_genus and len(image_files) > max_per_genus:
                image_files = image_files[:max_per_genus]
                print(f"  Limited to {max_per_genus} samples")
            
            # Split this genus's images into train/val
            if validation_split > 0:
                split_idx = int(len(image_files) * (1 - validation_split))
                train_files = image_files[:split_idx]
                val_files = image_files[split_idx:]
                
                # Choose which split to use
                selected_files = train_files if split == "train" else val_files
                print(f"  Using {len(selected_files)} images for {split} split")
            else:
                selected_files = image_files
            
            # Process selected images
            for image_file in selected_files:
                try:
                    image = Image.open(image_file).convert("RGB")
                    all_sample_data.append({
                        'image_path': str(image_file),
                        'image': image,
                        'genus': genus_name,
                        'image_name': image_file.stem
                    })
                except Exception as e:
                    print(f"  Error loading {image_file}: {e}")
                    continue
        
        return all_sample_data

def format_mushroom_conversations(dataset_samples):
    """Convert mushroom samples to conversation format."""
    conversations = []

    for sample in dataset_samples:
        genus = sample['genus']
        image = sample['image']
        
        # Get genus information
        genus_info = MUSHROOM_DESCRIPTIONS.get(genus, {})
        description = genus_info.get("description", f"A mushroom from the {genus} genus.")
        
        # Basic identification conversation
        prompt = random.choice(IDENTIFICATION_PROMPTS)
        basic_response = f"This is a {genus} mushroom."
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": basic_response}
                ]
            }
        ]
        conversations.append(conversation)
        
        # Add detailed feature conversation if enabled
        if INCLUDE_DETAILED_RESPONSES and genus_info:
            features_text = ", ".join(genus_info.get("features", []))
            habitat = genus_info.get("habitat", "Various forest environments")
            
            detailed_response = f"This is a {genus} mushroom. Key identifying features include: {features_text}. Habitat: {habitat}."
            
            detailed_conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Can you describe the key features of this mushroom?"},
                        {"type": "image", "image": image}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": detailed_response}
                    ]
                }
            ]
            conversations.append(detailed_conversation)
            
            # Safety conversation for toxic genera like Amanita
            if genus in ["Amanita"]:
                safety_conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Is this mushroom safe to eat?"},
                            {"type": "image", "image": image}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": f"This is a {genus} mushroom. Many species in this genus are highly toxic and potentially deadly. Never consume wild mushrooms without expert identification."}
                        ]
                    }
                ]
                conversations.append(safety_conversation)

    return conversations

def formatting_prompts_func(examples):
    """Format conversations for training."""
    convos = examples["conversations"]
    texts = [globals()['tokenizer'].apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix('<bos>') for convo in convos]
    return {"text": texts}

def setup_environment(config: Config):
    """Set up the training environment."""
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_device
    logger.info(f"CUDA_VISIBLE_DEVICES set to: {config.gpu_device}")
    
    # Set random seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

def load_model_and_tokenizer(config: Config):
    """Load the base model and tokenizer."""
    logger.info(f"Loading model: {config.model_name}")
    
    model, tokenizer = FastModel.from_pretrained(
        model_name=config.model_name,
        dtype=None,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
        full_finetuning=config.full_finetuning,
    )
    
    return model, tokenizer

def prepare_model_for_training(model, r_value: int, config: Config):
    """Prepare model for training with specific LoRA r value."""
    logger.info(f"Preparing model for training with LoRA r={r_value}")
    
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=r_value,
        lora_alpha=int(r_value * config.lora_alpha_ratio),
        lora_dropout=config.lora_dropout,
        bias="none",
        random_state=config.seed,
    )
    
    return model

def load_and_prepare_datasets(config: Config):
    """Load and prepare training and validation datasets."""
    logger.info("Loading datasets...")
    
    # Check if dataset path exists
    if not os.path.exists(config.dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {config.dataset_path}")
    
    # Load training data
    loader = MushroomDatasetLoader(config.dataset_path)
    dataset_samples = loader.load_dataset(
        max_per_genus=config.max_images_per_genus,
        validation_split=config.validation_split,
        seed=config.seed,
        split="train"
    )
    
    # Load validation data if needed
    val_dataset_samples = None
    if config.use_validation:
        val_dataset_samples = loader.load_dataset(
            max_per_genus=config.max_images_per_genus,
            validation_split=config.validation_split,
            seed=config.seed,
            split="validation"
        )
        logger.info(f"Loaded {len(val_dataset_samples)} validation samples")
    
    return dataset_samples, val_dataset_samples

def prepare_training_datasets(dataset_samples, val_dataset_samples, tokenizer, config: Config, dataset_cache: DatasetCache):
    """Prepare datasets for training using caching."""
    # Make tokenizer available globally for formatting function
    globals()['tokenizer'] = tokenizer
    
    # Prepare training dataset with caching
    if dataset_samples:
        # Create cache configuration for training
        cache_config = dataset_cache.get_cache_config(
            dataset_path=config.dataset_path,
            max_images_per_genus=config.max_images_per_genus,
            validation_split=config.validation_split if config.use_validation else 0,
            include_detailed_responses=INCLUDE_DETAILED_RESPONSES,
            split="train"
        )
        
        # Try to load from cache first
        cached_dataset, cached_samples = dataset_cache.load_dataset(cache_config)
        
        if cached_dataset is not None:
            logger.info("✅ Using cached training dataset - skipping expensive Dataset.from_dict() and dataset.map()!")
            dataset = cached_dataset
            if cached_samples:
                dataset_samples = cached_samples
        else:
            logger.info("⏳ Cache miss - creating training dataset from scratch (this will take time)...")
            start_time = time.time()
            
            # Format conversations (this is relatively fast)
            logger.info("📝 Formatting conversations...")
            conversations = format_mushroom_conversations(dataset_samples)
            
            # Create HF Dataset (this is slow!)
            logger.info("📦 Creating HF Dataset from conversations...")
            dataset = Dataset.from_dict({"conversations": conversations})
            
            # Apply chat template (this is VERY slow!)
            logger.info("🔄 Applying chat template to dataset...")
            dataset = dataset.map(formatting_prompts_func, batched=True)
            
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"✅ Training dataset processing completed in {duration:.2f} seconds")
            
            # Save to cache for next time
            dataset_cache.save_dataset(cache_config, dataset, dataset_samples)
    else:
        dataset = None
    
    # Prepare validation dataset if available
    eval_dataset = None
    if config.use_validation and val_dataset_samples:
        # Create cache configuration for validation
        val_cache_config = dataset_cache.get_cache_config(
            dataset_path=config.dataset_path,
            max_images_per_genus=config.max_images_per_genus,
            validation_split=config.validation_split,
            include_detailed_responses=INCLUDE_DETAILED_RESPONSES,
            split="val"
        )
        
        # Try to load validation dataset from cache
        val_dataset, val_samples = dataset_cache.load_dataset(val_cache_config)
        
        if val_dataset is not None:
            logger.info("✅ Using cached validation dataset!")
            eval_dataset = val_dataset
        else:
            logger.info("⏳ Creating validation dataset from scratch...")
            val_conversations = format_mushroom_conversations(val_dataset_samples)
            val_dataset = Dataset.from_dict({"conversations": val_conversations})
            val_dataset = val_dataset.map(formatting_prompts_func, batched=True)
            eval_dataset = val_dataset
            
            # Save validation cache
            dataset_cache.save_dataset(val_cache_config, val_dataset, val_dataset_samples)
    
    return dataset, eval_dataset

def train_model(model, tokenizer, dataset, eval_dataset, config: Config):
    """Train the model with the given configuration."""
    logger.info("Starting training...")
    
    # Configure training parameters based on dataset size
    dataset_size = len(dataset)
    if dataset_size < 100:
        max_steps = 20
    elif dataset_size < 500:
        max_steps = 40
    else:
        max_steps = config.max_steps
    
    logger.info(f"Training dataset size: {dataset_size}")
    logger.info(f"Training steps: {max_steps}")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            max_steps=max_steps,
            learning_rate=config.learning_rate,
            logging_steps=config.logging_steps,
            optim="adamw_8bit",
            weight_decay=config.weight_decay,
            lr_scheduler_type="linear",
            seed=config.seed,
            report_to="none",
            eval_steps=config.eval_steps,
        ),
    )
    
    # Train the model
    trainer_stats = trainer.train()
    
    return trainer_stats

def load_test_dataset(config: Config):
    """Load test dataset for evaluation."""
    if not os.path.exists(config.test_dataset_path):
        logger.warning(f"Test dataset path not found: {config.test_dataset_path}")
        return []
    
    test_loader = MushroomDatasetLoader(config.test_dataset_path)
    test_samples = test_loader.load_dataset(
        max_per_genus=config.max_images_per_genus // 4,  # Smaller test set
        validation_split=0.0,  # Use all test data
        seed=config.seed,
        split="train"
    )
    
    # Limit test samples for faster evaluation
    if len(test_samples) > config.max_test_samples:
        test_samples = random.sample(test_samples, config.max_test_samples)
        logger.info(f"Limited test set to {config.max_test_samples} samples")
    
    return test_samples

def extract_genus_from_response(response: str, valid_genera: List[str]) -> str:
    """Extract genus prediction from model response."""
    response_lower = response.lower()
    
    # Look for exact genus matches
    for genus in valid_genera:
        if genus.lower() in response_lower:
            return genus
    
    return "Unknown"

def evaluate_model(model, tokenizer, test_samples, config: Config) -> Dict:
    """Evaluate the trained model on test data."""
    if not test_samples:
        logger.warning("No test samples available for evaluation")
        return {}
    
    logger.info(f"Evaluating model on {len(test_samples)} test samples...")
    
    # Get valid genera from test samples and MUSHROOM_DESCRIPTIONS
    valid_genera = list(MUSHROOM_DESCRIPTIONS.keys())
    
    predictions = []
    ground_truths = []
    responses = []
    
    for i, sample in enumerate(test_samples):
        if i % 50 == 0:
            logger.info(f"Evaluating sample {i+1}/{len(test_samples)}")
        
        try:
            # Load and prepare image
            image = sample['image']
            
            # Create messages
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "What type of mushroom is this? Can you identify the genus?"}
                ]
            }]
            
            # Prepare inputs
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to("cuda")
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=64,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Extract model response
            response_parts = full_response.split("<start_of_turn>model")
            if len(response_parts) > 1:
                model_response = response_parts[-1].replace("<end_of_turn>", "").strip()
            else:
                model_response = full_response
            
            # Extract genus prediction
            predicted_genus = extract_genus_from_response(model_response, valid_genera)
            
            # Store results
            predictions.append(predicted_genus)
            ground_truths.append(sample['genus'])
            responses.append(model_response)
            
        except Exception as e:
            logger.error(f"Error processing sample {i}: {e}")
            predictions.append("Error")
            ground_truths.append(sample['genus'])
            responses.append("")
    
    # Calculate metrics
    metrics = calculate_classification_metrics(predictions, ground_truths)
    
    return metrics

def calculate_classification_metrics(predictions: List[str], ground_truths: List[str]) -> Dict:
    """Calculate comprehensive classification metrics."""
    if not predictions or not ground_truths:
        return {}
    
    # Filter out error cases
    valid_indices = [i for i, pred in enumerate(predictions) if pred != "Error"]
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_ground_truths = [ground_truths[i] for i in valid_indices]
    
    if not valid_predictions:
        return {}
    
    # Calculate basic metrics
    accuracy = accuracy_score(valid_ground_truths, valid_predictions)
    
    # Get unique labels
    all_labels = sorted(list(set(valid_ground_truths + valid_predictions)))
    
    # Calculate precision, recall, f1
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        valid_ground_truths, valid_predictions, average='macro', zero_division=0
    )
    
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        valid_ground_truths, valid_predictions, average='weighted', zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'total_samples': len(predictions),
        'valid_samples': len(valid_predictions),
        'error_samples': len(predictions) - len(valid_predictions),
    }
    
    return metrics

def run_lora_experiment(r_value: int, config: Config, dataset_cache: DatasetCache) -> Dict:
    """Run a single LoRA experiment with given r value."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running LoRA experiment with r={r_value}")
    logger.info(f"{'='*60}")
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(config)
        
        # Prepare model for training
        model = prepare_model_for_training(model, r_value, config)
        
        # Load datasets
        dataset_samples, val_dataset_samples = load_and_prepare_datasets(config)
        
        # Prepare training datasets with caching
        dataset, eval_dataset = prepare_training_datasets(
            dataset_samples, val_dataset_samples, tokenizer, config, dataset_cache
        )
        
        # Train model
        trainer_stats = train_model(model, tokenizer, dataset, eval_dataset, config)
        
        # Load test data
        test_samples = load_test_dataset(config)
        
        # Evaluate model
        metrics = evaluate_model(model, tokenizer, test_samples, config)
        
        # Add experiment info
        metrics['r_value'] = r_value
        metrics['training_time'] = trainer_stats.metrics.get('train_runtime', 0)
        metrics['training_loss'] = trainer_stats.metrics.get('train_loss', 0)
        
        # Save model if requested
        if config.save_models:
            model_dir = os.path.join(config.output_dir, f"model_r{r_value}")
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            logger.info(f"Model saved to {model_dir}")
        
        # Clear memory
        del model, tokenizer, dataset, eval_dataset
        torch.cuda.empty_cache()
        
        logger.info(f"Experiment with r={r_value} completed successfully")
        return metrics
        
    except Exception as e:
        logger.error(f"Experiment with r={r_value} failed: {e}")
        return {'r_value': r_value, 'error': str(e)}

def plot_results(results: List[Dict], config: Config):
    """Plot the results of LoRA experiments."""
    logger.info("Creating performance plots...")
    
    # Filter out failed experiments
    successful_results = [r for r in results if 'error' not in r and 'accuracy' in r]
    
    if not successful_results:
        logger.error("No successful experiments to plot")
        return
    
    # Extract data for plotting
    r_values = [r['r_value'] for r in successful_results]
    accuracies = [r['accuracy'] for r in successful_results]
    f1_macros = [r['f1_macro'] for r in successful_results]
    f1_weighteds = [r['f1_weighted'] for r in successful_results]
    training_times = [r['training_time'] for r in successful_results]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LoRA Rank (r) vs Performance Analysis\nGemma3N-4B Mushroom Classification', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy vs r
    axes[0, 0].plot(r_values, accuracies, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('LoRA Rank (r)')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy vs LoRA Rank')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # Add value annotations
    for i, (r, acc) in enumerate(zip(r_values, accuracies)):
        axes[0, 0].annotate(f'{acc:.3f}', (r, acc), textcoords="offset points", 
                           xytext=(0,10), ha='center')
    
    # Plot 2: F1 Scores vs r
    axes[0, 1].plot(r_values, f1_macros, 'ro-', linewidth=2, markersize=8, label='F1 Macro')
    axes[0, 1].plot(r_values, f1_weighteds, 'go-', linewidth=2, markersize=8, label='F1 Weighted')
    axes[0, 1].set_xlabel('LoRA Rank (r)')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('F1 Scores vs LoRA Rank')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 1)
    
    # Plot 3: Training Time vs r
    axes[1, 0].plot(r_values, training_times, 'mo-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('LoRA Rank (r)')
    axes[1, 0].set_ylabel('Training Time (seconds)')
    axes[1, 0].set_title('Training Time vs LoRA Rank')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Summary comparison
    # Normalize metrics for comparison
    norm_acc = np.array(accuracies) / max(accuracies) if accuracies else []
    norm_f1 = np.array(f1_macros) / max(f1_macros) if f1_macros else []
    norm_time = 1 - (np.array(training_times) / max(training_times)) if training_times else []  # Invert time (lower is better)
    
    width = 0.25
    x_pos = np.arange(len(r_values))
    
    axes[1, 1].bar(x_pos - width, norm_acc, width, label='Accuracy (norm)', alpha=0.8)
    axes[1, 1].bar(x_pos, norm_f1, width, label='F1 Macro (norm)', alpha=0.8)
    axes[1, 1].bar(x_pos + width, norm_time, width, label='Speed (norm)', alpha=0.8)
    
    axes[1, 1].set_xlabel('LoRA Rank (r)')
    axes[1, 1].set_ylabel('Normalized Performance')
    axes[1, 1].set_title('Normalized Performance Comparison')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(r_values)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(config.output_dir, 'lora_performance_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Performance plot saved to {plot_path}")
    
    plt.show()
    
    # Create detailed results table
    results_df = pd.DataFrame(successful_results)
    results_df = results_df[['r_value', 'accuracy', 'f1_macro', 'f1_weighted', 
                           'precision_macro', 'recall_macro', 'training_time']]
    results_df = results_df.round(4)
    
    logger.info("\nDetailed Results Summary:")
    logger.info("\n" + results_df.to_string(index=False))
    
    # Save results to CSV
    csv_path = os.path.join(config.output_dir, 'lora_experiment_results.csv')
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

def main():
    """Main function to run LoRA experiments."""
    parser = argparse.ArgumentParser(description="Run LoRA experiments on mushroom classification")
    parser.add_argument("--dataset-path", type=str, default="data/mushrooms/train/",
                       help="Path to training dataset")
    parser.add_argument("--test-dataset-path", type=str, default="data/mushrooms/test/",
                       help="Path to test dataset")
    parser.add_argument("--r-values", type=int, nargs='+', default=[4, 8, 16, 32],
                       help="LoRA r values to test")
    parser.add_argument("--max-steps", type=int, default=60,
                       help="Maximum training steps")
    parser.add_argument("--max-test-samples", type=int, default=200,
                       help="Maximum test samples for evaluation")
    parser.add_argument("--output-dir", type=str, default="mushroom_lora_experiments",
                       help="Output directory for results")
    parser.add_argument("--gpu-device", type=str, default="3",
                       help="GPU device to use")
    parser.add_argument("--save-models", action="store_true",
                       help="Save trained models")
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config()
    config.dataset_path = args.dataset_path
    config.test_dataset_path = args.test_dataset_path
    config.r_values = args.r_values
    config.max_steps = args.max_steps
    config.max_test_samples = args.max_test_samples
    config.output_dir = args.output_dir
    config.gpu_device = args.gpu_device
    config.save_models = args.save_models
    
    logger.info("Starting LoRA experiments...")
    logger.info(f"Configuration: {vars(config)}")
    
    # Setup environment
    setup_environment(config)
    
    # Initialize dataset cache
    dataset_cache = DatasetCache(cache_dir=os.path.join("notebooks/kaggle/dataset_cache"))
    logger.info("🚀 Dataset cache system initialized")
    logger.info("💡 This will cache the expensive Dataset.from_dict() and dataset.map() operations!")
    
    # Show cache status
    dataset_cache.list_caches()
    
    # Run experiments
    results = []
    for r_value in config.r_values:
        result = run_lora_experiment(r_value, config, dataset_cache)
        results.append(result)
        
        # Save intermediate results
        results_path = os.path.join(config.output_dir, 'intermediate_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Save final results
    final_results_path = os.path.join(config.output_dir, 'final_results.json')
    with open(final_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"All experiments completed. Results saved to {final_results_path}")
    
    # Plot results
    plot_results(results, config)
    
    logger.info("LoRA experiments completed successfully!")

if __name__ == "__main__":
    main()