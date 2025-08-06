#!/usr/bin/env python3
"""
Augmentoolkit Fine-tuning Script

A comprehensive script for fine-tuning models on Augmentoolkit's synthetic data.
Supports various data types including pretrain, factual SFT, RAG, and multimodal data.

Features:
- Multiple model architectures (Gemma 3, Gemma 3N, Llama, etc.)
- Text and multimodal training
- Multi-stage training (pretrain -> SFT)
- Configurable data mixing and ratios
- LoRA and full fine-tuning support
- Integration with Augmentoolkit output formats
- Distributed training on multiple GPUs

Data Format Support:
- Pretraining data: Raw text in {"text": "..."} format
- Factual SFT: Conversations in ShareGPT format {"conversations": [...]}
- RAG data: Segments format {"segments": [...]} with loss masking
- Representation variation: Raw text variations
- Correction data: Segments format with masked incorrect answers
- Generic data: Chain-of-thought conversations
- Inferred facts: Domain-specific generated facts
- Text chunks: Original document chunks

Multi-stage Training:
1. Pretrain stage: Uses raw text data (pretraining, representation variation, inferred facts, text chunks)
2. SFT stage: Uses conversational data (factual SFT, RAG, correction, generic datasets)

Loss Masking:
- Segments format data automatically applies loss masking for labeled segments
- Conversation data uses response-only training (masks user inputs)
"""

import os

# CRITICAL: Set environment variables BEFORE any ML library imports
# This prevents PyTorch from detecting multiple GPUs and using DataParallel
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Will be overridden by config, but ensures single GPU default
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['LOCAL_WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import yaml
import argparse
import logging
import io
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import random

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from PIL import Image
import pandas as pd
from datetime import datetime

# Try to import Unsloth if available
try:
    from unsloth import FastLanguageModel, FastVisionModel, FastModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from unsloth.trainer import UnslothVisionDataCollator
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("Unsloth not available. Using standard transformers.")

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoProcessor,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    DataCollatorForLanguageModeling
)
from trl import SFTTrainer, SFTConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch

# Fix PyTorch Dynamo recompile limits for Unsloth + Gemma 3N
import torch._dynamo
torch._dynamo.config.cache_size_limit = 1000  # Increase from default 64
torch._dynamo.config.suppress_errors = True   # Don't fail on compilation errors

# Set up environment for better PyTorch compilation
import os
os.environ['TORCH_LOGS'] = 'recompiles'  # Monitor recompilations
os.environ['TORCHDYNAMO_VERBOSE'] = '0'   # Reduce verbose output

# Initialize single GPU setup
def setup_single_gpu():
    """Initialize single GPU if available."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"CUDA available with {device_count} GPU(s)")
        torch.cuda.set_device(0)  # Use first GPU
        return True, 0
    else:
        logger.warning("CUDA not available, using CPU")
        return False, -1

CUDA_AVAILABLE, GPU_DEVICE = setup_single_gpu()

logger.setLevel(logging.INFO)

@dataclass
class ResponseOnlyDataCollator:
    """Efficient data collator for response-only training using preprocessed data."""
    
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process a batch of already-tokenized data with labels."""
        # Efficient padding using transformers utilities
        padded_batch = self.tokenizer.pad(
            {"input_ids": [f["input_ids"] for f in features]},
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        
        max_length = padded_batch["input_ids"].shape[1]
        padded_labels = []
        
        for f in features:
            labels = f["labels"]
            padding_length = max_length - len(labels)
            if padding_length > 0:
                labels = labels + [-100] * padding_length
            padded_labels.append(labels)
        
        return {
            "input_ids": padded_batch["input_ids"],
            "attention_mask": padded_batch["attention_mask"], 
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


@dataclass
class SegmentsDataCollator:
    """Efficient segments data collator for data with per-segment loss masking."""
    
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process preprocessed segments data - same as ResponseOnly after preprocessing."""
        return ResponseOnlyDataCollator(self.tokenizer, self.pad_to_multiple_of)(features)


@dataclass
class MultimodalDataCollator:
    """Efficient multimodal data collator using preprocessed images."""
    
    def __init__(self, processor, pad_to_multiple_of=8):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process preprocessed multimodal data efficiently."""
        batch_input_ids = []
        batch_labels = []
        batch_pixel_values = []
        
        for feature in features:
            batch_input_ids.append(feature["input_ids"])
            batch_labels.append(feature["labels"])
            # Use preprocessed pixel values (or None for text-only)
            batch_pixel_values.append(feature.get("pixel_values"))
        
        # Efficient padding using transformers utilities
        padded_batch = self.tokenizer.pad(
            {"input_ids": batch_input_ids},
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        
        # Pad labels
        max_length = padded_batch["input_ids"].shape[1]
        padded_labels = []
        for labels in batch_labels:
            padding_length = max_length - len(labels)
            if padding_length > 0:
                labels = labels + [-100] * padding_length
            padded_labels.append(labels)
        
        result = {
            "input_ids": padded_batch["input_ids"],
            "attention_mask": padded_batch["attention_mask"],
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
        
        # Add pixel values if present
        if any(pv is not None for pv in batch_pixel_values):
            # Stack pixel values, using zeros for text-only examples
            stacked_pixels = []
            # Get dimensions from first valid image
            reference_shape = None
            for pv in batch_pixel_values:
                if pv is not None:
                    reference_shape = torch.tensor(pv).shape
                    break
            
            # Default shape if no images found
            if reference_shape is None:
                reference_shape = (3, 224, 224)  # Default for vision transformers
            
            for pv in batch_pixel_values:
                if pv is not None:
                    stacked_pixels.append(torch.tensor(pv))
                else:
                    # Dummy pixel values with correct dimensions
                    stacked_pixels.append(torch.zeros(reference_shape))
            result["pixel_values"] = torch.stack(stacked_pixels)
        
        return result


# Preprocessing functions for efficiency
def preprocess_pretraining_data(examples, tokenizer, max_length=4096):
    """Efficiently preprocess text data for pretraining (language modeling)."""
    # Extract actual tokenizer if we have a processor
    actual_tokenizer = tokenizer
    if hasattr(tokenizer, 'tokenizer'):
        actual_tokenizer = tokenizer.tokenizer
    elif hasattr(tokenizer, 'encode') is False and hasattr(tokenizer, 'tokenizer'):
        actual_tokenizer = tokenizer.tokenizer
    
    input_ids_batch = []
    labels_batch = []
    
    for text in examples["text"]:
        # Tokenize the text
        tokens = actual_tokenizer.encode(
            text, 
            add_special_tokens=True, 
            max_length=max_length, 
            truncation=True
        )
        
        # For language modeling, input_ids and labels are the same
        input_ids_batch.append(tokens)
        labels_batch.append(tokens.copy())  # Copy for safety
    
    return {"input_ids": input_ids_batch, "labels": labels_batch}


def preprocess_response_only_data(examples, tokenizer, response_template="<start_of_turn>model\n", max_length=4096):
    """Efficiently preprocess conversation data for response-only training."""
    # Extract actual tokenizer if we have a processor
    actual_tokenizer = tokenizer
    if hasattr(tokenizer, 'tokenizer'):
        actual_tokenizer = tokenizer.tokenizer
    elif hasattr(tokenizer, 'encode') is False and hasattr(tokenizer, 'tokenizer'):
        actual_tokenizer = tokenizer.tokenizer
    
    # Pre-compute template tokens once
    response_tokens = actual_tokenizer.encode(response_template, add_special_tokens=False)
    user_template = "<start_of_turn>user\n" if "gemma" in response_template else "<|im_start|>user\n"
    user_tokens = actual_tokenizer.encode(user_template, add_special_tokens=False)
    
    response_tensor = torch.tensor(response_tokens)
    user_tensor = torch.tensor(user_tokens)
    
    input_ids_batch = []
    labels_batch = []
    
    for text in examples["text"]:
        # Tokenize once
        tokens = actual_tokenizer.encode(text, add_special_tokens=True, max_length=max_length, truncation=True)
        
        # Efficient masking using vectorized operations
        tokens_array = torch.tensor(tokens)
        
        # Find all positions where response template occurs
        response_positions = []
        for i in range(len(tokens) - len(response_tokens) + 1):
            if torch.equal(tokens_array[i:i+len(response_tokens)], response_tensor):
                response_positions.append(i)
        
        # Initially mask everything (user inputs)
        labels = [-100] * len(tokens)
        
        # Unmask assistant responses only
        for pos in response_positions:
            # Start unmasking after the response template
            start_response = pos + len(response_tokens)
            
            # Find where this response ends (next user turn or end of sequence)
            end_response = len(tokens)
            for i in range(start_response, len(tokens) - len(user_tokens) + 1):
                if torch.equal(tokens_array[i:i+len(user_tokens)], user_tensor):
                    end_response = i
                    break
            
            # Unmask the assistant response (exclude template)
            for j in range(start_response, end_response):
                labels[j] = tokens[j]
        
        input_ids_batch.append(tokens)
        labels_batch.append(labels)
    
    return {"input_ids": input_ids_batch, "labels": labels_batch}


def preprocess_segments_data(examples, tokenizer, max_length=4096):
    """Efficiently preprocess segments data with loss masking."""
    # Extract actual tokenizer if we have a processor
    actual_tokenizer = tokenizer
    if hasattr(tokenizer, 'tokenizer'):
        actual_tokenizer = tokenizer.tokenizer
    elif hasattr(tokenizer, 'encode') is False and hasattr(tokenizer, 'tokenizer'):
        actual_tokenizer = tokenizer.tokenizer
    
    input_ids_batch = []
    labels_batch = []
    
    for segments in examples["segments"]:
        all_tokens = []
        all_labels = []
        
        # Handle None or empty segments
        if segments is None:
            # Create empty tokens for None segments
            all_tokens = []
            all_labels = []
        else:
            for segment in segments:
                # Handle None segments within the list
                if segment is None:
                    continue
                    
                text = segment.get("text", "")
                should_train = segment.get("label", True)
                
                # Skip empty text
                if not text:
                    continue
                
                segment_tokens = actual_tokenizer.encode(text, add_special_tokens=False)
                all_tokens.extend(segment_tokens)
                
                if should_train:
                    all_labels.extend(segment_tokens)
                else:
                    all_labels.extend([-100] * len(segment_tokens))
        
        # Ensure we have at least some tokens (add EOS if empty)
        if not all_tokens:
            all_tokens = [actual_tokenizer.eos_token_id]
            all_labels = [-100]  # Mask empty examples
        
        # Truncate to max length
        all_tokens = all_tokens[:max_length]
        all_labels = all_labels[:max_length]
        
        input_ids_batch.append(all_tokens)
        labels_batch.append(all_labels)
    
    return {"input_ids": input_ids_batch, "labels": labels_batch}


def preprocess_multimodal_data(examples, processor, max_length=4096):
    """Efficiently preprocess multimodal data with image preprocessing."""
    import requests
    
    input_ids_batch = []
    labels_batch = []
    pixel_values_batch = []
    
    for i in range(len(examples["text"])):
        text = examples["text"][i]
        
        # Process text
        if "conversations" in examples and i < len(examples["conversations"]):
            # Handle conversation format
            messages = examples["conversations"][i]
            formatted_text = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            formatted_text = text
        
        tokens = processor.tokenizer.encode(
            formatted_text, add_special_tokens=True, max_length=max_length, truncation=True
        )
        
        # Process image if present
        pixel_values = None
        if "image" in examples and i < len(examples["image"]):
            image_path = examples["image"][i]
            if image_path:
                try:
                    if image_path.startswith(("http://", "https://")):
                        response = requests.get(image_path)
                        image = Image.open(io.BytesIO(response.content))
                    else:
                        image = Image.open(image_path)
                    
                    # Preprocess image once
                    image_inputs = processor(images=image, return_tensors="pt")
                    if hasattr(image_inputs, 'pixel_values'):
                        pixel_values = image_inputs.pixel_values.squeeze(0).numpy()
                except Exception as e:
                    logger.warning(f"Failed to preprocess image {image_path}: {e}")
        
        input_ids_batch.append(tokens)
        labels_batch.append(tokens)  # Remove unnecessary copy
        pixel_values_batch.append(pixel_values)
    
    return {
        "input_ids": input_ids_batch, 
        "labels": labels_batch,
        "pixel_values": pixel_values_batch
    }


@dataclass
class AugmentoolkitConfig:
    """Configuration for Augmentoolkit fine-tuning with Unsloth."""
    
    # Model configuration - Optimized for single GPU Unsloth
    model_name: str = "unsloth/gemma-3n-E2B-it"
    model_type: str = "multimodal"
    load_in_4bit: bool = False
    load_in_8bit: bool = True
    use_gradient_checkpointing: bool = True
    max_seq_length: int = 4096
    
    # Single GPU training
    bf16: bool = True
    fp16: bool = False
    dataloader_num_workers: int = 2  # ✅ Reduced for single GPU
    
    use_unsloth: bool = True   # ✅ Always use Unsloth for this script
    gpu_device: int = 0        # ✅ Which GPU to use
    
    # Data configuration
    data_base_path: str = "./outputs/"
    data_types: List[str] = field(default_factory=lambda: [
        "factual_sft",              # ✅ Available: factual_sft_facts_{type}_{num}/plain_qa_list.jsonl
        "pretrain",                 # ✅ Available: pretraining_run/*.jsonl + individual dirs  
        "rag_data",                 # ✅ Available: simplified_data_rag.jsonl in factual SFT dirs
        "representation_variation", # ✅ Available: pretraining_run/representation_variation_facts.jsonl
        "correction",               # ✅ Available
    ])
    data_mixing_ratios: Optional[Dict[str, float]] = None
    max_samples_per_type: int = 100000
    validation_split: float = 0.1
    
    # Training configuration - Updated for multi-stage
    training_stages: List[str] = field(default_factory=lambda: ["sft"])  # ✅ Only SFT stage
    
    # LoRA configuration - Conditional based on stage
    use_lora: bool = True
    full_finetuning: bool = False
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: str = "all-linear"
    
    # Training hyperparameters
    per_device_train_batch_size: int = 2    # ✅ Moderate batch size for single GPU
    per_device_eval_batch_size: int = 2     # ✅ Match training batch size
    gradient_accumulation_steps: int = 16   # ✅ Effective batch = 2 * 16 = 32
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    num_train_epochs: int = 1
    max_steps: Optional[int] = None
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    
    # Vision specific
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    report_to: str = "none"
    
    # Other settings
    seed: int = 3407
    use_fast_tokenizer: bool = True
    chat_template: str = "gemma-3n"
    remove_unused_columns: bool = False
    output_dir: str = "./gemma3n_finetuned_unsloth_8bit_it"
    
    # Caching settings
    enable_dataset_caching: bool = True
    cache_dir: str = "./dataset_cache"
    force_reprocess: bool = False

    def __post_init__(self):
        """Post-initialization to adjust settings for single GPU Unsloth training."""
        # Ensure only one quantization method is enabled
        if self.load_in_8bit and self.load_in_4bit:
            self.load_in_4bit = False
            self.load_in_8bit = True
            logger.info("✅ Using 8-bit quantization (preferred for Unsloth)")
        
        # Ensure Unsloth is enabled
        if not self.use_unsloth:
            self.use_unsloth = True
            logger.info("✅ Enabled Unsloth for single GPU training")
        
        # Set appropriate device
        if CUDA_AVAILABLE:
            torch.cuda.set_device(self.gpu_device)
            logger.info(f"✅ Using GPU {self.gpu_device} for training")


class AugmentoolkitDataLoader:
    """Loads and processes various types of Augmentoolkit synthetic data."""
    
    def __init__(self, config: AugmentoolkitConfig):
        self.config = config
        # Expand user path (~) if present
        expanded_path = os.path.expanduser(config.data_base_path) 
        self.base_path = Path(expanded_path)
        
    def load_factual_sft_data(self) -> Dataset:
        """Load factual SFT data from multiple sources and formats."""
        datasets = []
        
        # Load from sft_run directory (completion format) - may not exist
        sft_run_dir = self.base_path / "sft_run"
        if sft_run_dir.exists():
            logger.info("Found sft_run directory, loading completion data...")
            # Look for factual_sft_completion directory
            completion_dir = sft_run_dir / "factual_sft_completion"
            if completion_dir.exists():
                for jsonl_file in completion_dir.glob("*.jsonl"):
                    logger.info(f"Loading factual SFT completion data from {jsonl_file}")
                    try:
                        dataset = load_dataset("json", data_files=str(jsonl_file), split="train")
                        dataset = dataset.add_column("data_type", ["factual_sft_completion"] * len(dataset))
                        dataset = dataset.add_column("format_type", ["completion"] * len(dataset))
                        datasets.append(dataset)
                        logger.info(f"  -> Loaded {len(dataset)} completion samples")
                    except Exception as e:
                        logger.warning(f"Failed to load factual SFT completion data: {e}")
            
            # Look for combined_factual_data directory  
            combined_dir = sft_run_dir / "combined_factual_data"
            if combined_dir.exists():
                for jsonl_file in combined_dir.glob("*.jsonl"):
                    logger.info(f"Loading combined factual data from {jsonl_file}")
                    try:
                        dataset = load_dataset("json", data_files=str(jsonl_file), split="train")
                        dataset = dataset.add_column("data_type", ["factual_sft_conversations"] * len(dataset))
                        dataset = dataset.add_column("format_type", ["conversations"] * len(dataset))
                        datasets.append(dataset)
                        logger.info(f"  -> Loaded {len(dataset)} conversation samples")
                    except Exception as e:
                        logger.warning(f"Failed to load combined factual data: {e}")
        else:
            logger.info("No sft_run directory found, looking for individual factual SFT directories...")
        
        # Load from individual factual_sft directories (main source based on your structure)
        sft_types = ["openended", "hallucination", "negative", "vague", "followup"]
        total_loaded = 0
        
        for sft_type in sft_types:
            type_count = 0
            # Look for individual pipeline outputs with the actual naming pattern
            for possible_dir in self.base_path.glob(f"factual_sft_facts_{sft_type}_*"):
                qa_file = possible_dir / "plain_qa_list.jsonl"
                if qa_file.exists():
                    logger.info(f"Loading {sft_type} SFT data from {qa_file}")
                    try:
                        dataset = load_dataset("json", data_files=str(qa_file), split="train")
                        dataset = dataset.add_column("data_type", [f"factual_sft_{sft_type}"] * len(dataset))
                        dataset = dataset.add_column("format_type", ["conversations"] * len(dataset))
                        datasets.append(dataset)
                        type_count += len(dataset)
                        logger.info(f"  -> Loaded {len(dataset)} {sft_type} samples from {possible_dir.name}")
                    except Exception as e:
                        logger.warning(f"Failed to load {sft_type} data from {qa_file}: {e}")
                else:
                    logger.debug(f"No plain_qa_list.jsonl found in {possible_dir}")
            
            if type_count > 0:
                total_loaded += type_count
                logger.info(f"📊 Total {sft_type} samples: {type_count}")
            else:
                logger.warning(f"❌ No {sft_type} data found")
        
        if datasets:
            combined = concatenate_datasets(datasets)
            logger.info(f"📊 Total factual SFT data loaded: {len(combined)} samples from {len(datasets)} files")
            return combined
        else:
            logger.warning("❌ No factual SFT data found")
            return None
    
    def load_pretrain_data(self) -> Dataset:
        """Load pretraining data from pretraining_run directory."""
        datasets = []
        
        # Look for pretraining_run directory (main source based on your structure)
        pretrain_dir = self.base_path / "pretraining_run"
        if pretrain_dir.exists():
            logger.info("Found pretraining_run directory, loading pretraining data...")
            pretrain_files = list(pretrain_dir.glob("*.jsonl"))
            
            if pretrain_files:
                for jsonl_file in pretrain_files:
                    logger.info(f"Loading pretraining data from {jsonl_file}")
                    try:
                        dataset = load_dataset("json", data_files=str(jsonl_file), split="train")
                        
                        # Determine data type based on filename
                        if "inferred_facts" in jsonl_file.name:
                            data_type = "inferred_facts"
                        elif "representation_variation" in jsonl_file.name:
                            data_type = "representation_variation"
                        elif "text_chunks" in jsonl_file.name:
                            data_type = "text_chunks"
                        else:
                            data_type = "pretrain"
                        
                        dataset = dataset.add_column("data_type", [data_type] * len(dataset))
                        dataset = dataset.add_column("format_type", ["text"] * len(dataset))
                        datasets.append(dataset)
                        logger.info(f"  -> Loaded {len(dataset)} {data_type} samples")
                    except Exception as e:
                        logger.warning(f"Failed to load pretrain data from {jsonl_file}: {e}")
            else:
                logger.warning("No .jsonl files found in pretraining_run directory")
        else:
            logger.warning("No pretraining_run directory found")
        
        # Also check for pretraining subset in sft_run (mixed into SFT) - may not exist
        sft_pretrain_dir = self.base_path / "sft_run"
        if sft_pretrain_dir.exists():
            logger.info("Checking sft_run for pretraining subset...")
            for jsonl_file in sft_pretrain_dir.glob("pretraining_subset_*.jsonl"):
                logger.info(f"Loading pretraining subset from {jsonl_file}")
                try:
                    dataset = load_dataset("json", data_files=str(jsonl_file), split="train")
                    dataset = dataset.add_column("data_type", ["pretrain_subset"] * len(dataset))
                    dataset = dataset.add_column("format_type", ["completion"] * len(dataset))
                    datasets.append(dataset)
                    logger.info(f"  -> Loaded {len(dataset)} pretraining subset samples")
                except Exception as e:
                    logger.warning(f"Failed to load pretrain subset: {e}")
        
        # Also check individual pipeline outputs for additional pretrain data
        logger.info("Checking for individual pretrain pipeline outputs...")
        for possible_dir in [self.base_path / "inferred_facts_facts", self.base_path / "representation_variation_facts"]:
            if possible_dir.exists():
                output_file = possible_dir / "final_output.jsonl"
                if output_file.exists():
                    logger.info(f"Loading additional pretrain data from {output_file}")
                    try:
                        dataset = load_dataset("json", data_files=str(output_file), split="train")
                        # Determine type from directory name
                        if "inferred_facts" in possible_dir.name:
                            data_type = "inferred_facts_individual"
                        elif "representation_variation" in possible_dir.name:
                            data_type = "representation_variation_individual"
                        else:
                            data_type = "pretrain_individual"
                        
                        dataset = dataset.add_column("data_type", [data_type] * len(dataset))
                        dataset = dataset.add_column("format_type", ["text"] * len(dataset))
                        datasets.append(dataset)
                        logger.info(f"  -> Loaded {len(dataset)} {data_type} samples")
                    except Exception as e:
                        logger.warning(f"Failed to load pretrain data from {output_file}: {e}")
        
        if datasets:
            combined = concatenate_datasets(datasets)
            logger.info(f"📊 Total pretraining data loaded: {len(combined)} samples from {len(datasets)} files")
            
            # Log breakdown by data type
            if "data_type" in combined.column_names:
                from collections import Counter
                type_counts = Counter(combined["data_type"])
                logger.info("📈 Pretraining data breakdown:")
                for data_type, count in type_counts.items():
                    percentage = (count / len(combined)) * 100
                    logger.info(f"  -> {data_type}: {count} samples ({percentage:.1f}%)")
            
            return combined
        else:
            logger.warning("❌ No pretraining data found")
            return None
    
    def load_rag_data(self) -> Dataset:
        """Load RAG data in segments format."""
        datasets = []
        
        # Look for RAG data in sft_run directory (segments format) - may not exist
        sft_run_dir = self.base_path / "sft_run"
        if sft_run_dir.exists():
            logger.info("Checking sft_run directory for RAG data...")
            for jsonl_file in sft_run_dir.glob("axolotl_rag_conversations_*.jsonl"):
                logger.info(f"Loading RAG segments data from {jsonl_file}")
                try:
                    dataset = load_dataset("json", data_files=str(jsonl_file), split="train")
                    dataset = dataset.add_column("data_type", ["rag_segments"] * len(dataset))
                    dataset = dataset.add_column("format_type", ["segments"] * len(dataset))
                    datasets.append(dataset)
                    logger.info(f"  -> Loaded {len(dataset)} RAG segments samples")
                except Exception as e:
                    logger.warning(f"Failed to load RAG segments data: {e}")
        
        # Look for RAG data in factual SFT directories (simplified_data_rag.jsonl)
        # This appears to be where your RAG data actually is based on your file structure
        logger.info("Looking for RAG data in factual SFT directories...")
        rag_count = 0
        
        for possible_dir in self.base_path.glob("factual_sft_facts_*"):
            rag_file = possible_dir / "simplified_data_rag.jsonl"
            if rag_file.exists():
                logger.info(f"Loading RAG data from {rag_file}")
                try:
                    dataset = load_dataset("json", data_files=str(rag_file), split="train")
                    dataset = dataset.add_column("data_type", ["rag_simplified"] * len(dataset))
                    dataset = dataset.add_column("format_type", ["segments"] * len(dataset))
                    dataset = dataset.add_column("source_sft_type", [possible_dir.name.split('_')[-2] if len(possible_dir.name.split('_')) > 3 else "unknown"] * len(dataset))
                    datasets.append(dataset)
                    rag_count += len(dataset)
                    logger.info(f"  -> Loaded {len(dataset)} RAG samples from {possible_dir.name}")
                except Exception as e:
                    logger.warning(f"Failed to load RAG data from {rag_file}: {e}")
        
        # Look for main RAG data directory (rag_data_facts)
        rag_data_dir = self.base_path / "rag_data_facts"
        if rag_data_dir.exists():
            rag_file = rag_data_dir / "axolotl_rag_conversations.jsonl"
            if rag_file.exists():
                logger.info(f"Loading main RAG data from {rag_file}")
                try:
                    dataset = load_dataset("json", data_files=str(rag_file), split="train")
                    dataset = dataset.add_column("data_type", ["rag_main"] * len(dataset))
                    dataset = dataset.add_column("format_type", ["segments"] * len(dataset))
                    datasets.append(dataset)
                    rag_count += len(dataset)
                    logger.info(f"  -> Loaded {len(dataset)} main RAG samples")
                except Exception as e:
                    logger.warning(f"Failed to load main RAG data: {e}")
        
        # Look for individual RAG pipeline outputs (legacy pattern)
        for possible_dir in self.base_path.glob("rag_*"):
            if possible_dir.is_dir() and possible_dir.name != "rag_data_facts":  # Skip main dir (already processed)
                rag_file = possible_dir / "axolotl_rag_conversations.jsonl"
                if rag_file.exists():
                    logger.info(f"Loading RAG data from {rag_file}")
                    try:
                        dataset = load_dataset("json", data_files=str(rag_file), split="train")
                        dataset = dataset.add_column("data_type", ["rag_segments"] * len(dataset))
                        dataset = dataset.add_column("format_type", ["segments"] * len(dataset))
                        datasets.append(dataset)
                        rag_count += len(dataset)
                        logger.info(f"  -> Loaded {len(dataset)} RAG samples from {possible_dir.name}")
                    except Exception as e:
                        logger.warning(f"Failed to load RAG data: {e}")
        
        if datasets:
            combined = concatenate_datasets(datasets)
            logger.info(f"📊 Total RAG data loaded: {len(combined)} samples from {len(datasets)} files")
            return combined
        else:
            logger.warning("❌ No RAG data found")
            return None
    
    def load_representation_variation_data(self) -> Dataset:
        """Load representation variation data for domain adaptation."""
        datasets = []
        
        # Look for representation variation outputs in pretraining_run
        pretrain_dir = self.base_path / "pretraining_run"
        if pretrain_dir.exists():
            for jsonl_file in pretrain_dir.glob("representation_variation_*.jsonl"):
                logger.info(f"Loading representation variation data from {jsonl_file}")
                try:
                    dataset = load_dataset("json", data_files=str(jsonl_file), split="train")
                    # Ensure text format for pretraining
                    dataset = dataset.add_column("data_type", ["representation_variation"] * len(dataset))
                    dataset = dataset.add_column("format_type", ["text"] * len(dataset))
                    datasets.append(dataset)
                    logger.info(f"  -> Loaded {len(dataset)} representation variation samples")
                except Exception as e:
                    logger.warning(f"Failed to load representation variation data: {e}")
            
            # Also load inferred facts and text chunks (similar domain adaptation data)
            for jsonl_file in pretrain_dir.glob("inferred_facts_*.jsonl"):
                logger.info(f"Loading inferred facts data from {jsonl_file}")
                try:
                    dataset = load_dataset("json", data_files=str(jsonl_file), split="train")
                    dataset = dataset.add_column("data_type", ["inferred_facts"] * len(dataset))
                    dataset = dataset.add_column("format_type", ["text"] * len(dataset))
                    datasets.append(dataset)
                    logger.info(f"  -> Loaded {len(dataset)} inferred facts samples")
                except Exception as e:
                    logger.warning(f"Failed to load inferred facts data: {e}")
            
            for jsonl_file in pretrain_dir.glob("text_chunks_*.jsonl"):
                logger.info(f"Loading text chunks data from {jsonl_file}")
                try:
                    dataset = load_dataset("json", data_files=str(jsonl_file), split="train")
                    dataset = dataset.add_column("data_type", ["text_chunks"] * len(dataset))
                    dataset = dataset.add_column("format_type", ["text"] * len(dataset))
                    datasets.append(dataset)
                    logger.info(f"  -> Loaded {len(dataset)} text chunks samples")
                except Exception as e:
                    logger.warning(f"Failed to load text chunks data: {e}")
        
        # Look for individual representation variation pipeline outputs
        for possible_dir in self.base_path.glob("representation_variation_*"):
            if possible_dir.is_dir():
                output_file = possible_dir / "final_output.jsonl"
                if output_file.exists():
                    logger.info(f"Loading representation variation data from {output_file}")
                    try:
                        dataset = load_dataset("json", data_files=str(output_file), split="train")
                        dataset = dataset.add_column("data_type", ["representation_variation"] * len(dataset))
                        dataset = dataset.add_column("format_type", ["text"] * len(dataset))
                        datasets.append(dataset)
                        logger.info(f"  -> Loaded {len(dataset)} representation variation samples")
                    except Exception as e:
                        logger.warning(f"Failed to load representation variation data: {e}")
        
        if datasets:
            combined = concatenate_datasets(datasets)
            logger.info(f"📊 Total representation variation data: {len(combined)} samples")
            return combined
        else:
            logger.warning("No representation variation data found")
            return None
    
    def load_correction_data(self) -> Dataset:
        """Load correction pipeline data in segments format."""
        datasets = []
        
        # Look for correction data in sft_run directory (segments format)
        sft_run_dir = self.base_path / "sft_run"
        if sft_run_dir.exists():
            for json_file in sft_run_dir.glob("axolotl_correction_conversations*.json"):
                logger.info(f"Loading correction segments data from {json_file}")
                try:
                    dataset = load_dataset("json", data_files=str(json_file), split="train")
                    dataset = dataset.add_column("data_type", ["correction_segments"] * len(dataset))
                    dataset = dataset.add_column("format_type", ["segments"] * len(dataset))
                    datasets.append(dataset)
                except Exception as e:
                    logger.warning(f"Failed to load correction segments data: {e}")
        
        # Look for main corrections directory (corrections_facts)  
        corrections_dir = self.base_path / "corrections_facts"
        if corrections_dir.exists():
            correction_file = corrections_dir / "axolotl_correction_conversations.json"
            if correction_file.exists():
                logger.info(f"Loading main corrections data from {correction_file}")
                try:
                    dataset = load_dataset("json", data_files=str(correction_file), split="train")
                    dataset = dataset.add_column("data_type", ["correction_main"] * len(dataset))
                    dataset = dataset.add_column("format_type", ["segments"] * len(dataset))
                    datasets.append(dataset)
                    logger.info(f"  -> Loaded {len(dataset)} correction samples")
                except Exception as e:
                    logger.warning(f"Failed to load main correction data: {e}")
        
        # Look for individual correction pipeline outputs
        for possible_dir in self.base_path.glob("correction_*"):
            if possible_dir.is_dir() and possible_dir.name != "corrections_facts":  # Skip main dir
                correction_file = possible_dir / "axolotl_correction_conversations.json"
                if correction_file.exists():
                    logger.info(f"Loading correction data from {correction_file}")
                    try:
                        dataset = load_dataset("json", data_files=str(correction_file), split="train")
                        dataset = dataset.add_column("data_type", ["correction_segments"] * len(dataset))
                        dataset = dataset.add_column("format_type", ["segments"] * len(dataset))
                        datasets.append(dataset)
                    except Exception as e:
                        logger.warning(f"Failed to load correction data: {e}")
        
        if datasets:
            return concatenate_datasets(datasets)
        else:
            logger.warning("No correction data found")
            return None
    
    def load_generic_datasets(self) -> List[Dataset]:
        """Load generic datasets and transformed generic data."""
        datasets = []
        
        # Load from sft_run directory (organized completion format)
        sft_run_dir = self.base_path / "sft_run"
        if sft_run_dir.exists():
            # Look for generic_sft_completion directory (your actual structure)
            generic_completion_dir = sft_run_dir / "generic_sft_completion"
            if generic_completion_dir.exists():
                logger.info("Found organized generic SFT completion data in sft_run/")
                for jsonl_file in generic_completion_dir.glob("*.jsonl"):
                    logger.info(f"Loading generic SFT completion data from {jsonl_file}")
                    try:
                        dataset = load_dataset("json", data_files=str(jsonl_file), split="train")
                        dataset = dataset.add_column("data_type", ["generic_sft_completion"] * len(dataset))
                        dataset = dataset.add_column("format_type", ["completion"] * len(dataset))
                        datasets.append(dataset)
                        logger.info(f"  -> Loaded {len(dataset)} samples from {jsonl_file.name}")
                    except Exception as e:
                        logger.warning(f"Failed to load generic SFT completion data: {e}")
        
        # Load from main generic_sft directory (your actual structure)
        generic_sft_dir = self.base_path / "generic_sft"
        if generic_sft_dir.exists():
            logger.info("Found main generic_sft directory")
            for jsonl_file in generic_sft_dir.glob("*.jsonl"):
                logger.info(f"Loading generic data from {jsonl_file}")
                try:
                    dataset = load_dataset("json", data_files=str(jsonl_file), split="train")
                    dataset = dataset.add_column("data_type", ["generic_sft"] * len(dataset))
                    dataset = dataset.add_column("format_type", ["conversations"] * len(dataset))
                    datasets.append(dataset)
                    logger.info(f"  -> Loaded {len(dataset)} samples from {jsonl_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to load generic data: {e}")
        
        # Look for transformed generic data outputs (your actual structure)
        for possible_dir in self.base_path.glob("transformed_generic_data_*"):
            if possible_dir.is_dir():
                final_output = possible_dir / "final_total_output.jsonl"
                if final_output.exists():
                    logger.info(f"Loading transformed generic data from {final_output}")
                    try:
                        dataset = load_dataset("json", data_files=str(final_output), split="train")
                        dataset = dataset.add_column("data_type", ["generic_transformed"] * len(dataset))
                        dataset = dataset.add_column("format_type", ["conversations"] * len(dataset))
                        datasets.append(dataset)
                        logger.info(f"  -> Loaded {len(dataset)} samples from {possible_dir.name}")
                    except Exception as e:
                        logger.warning(f"Failed to load transformed generic data: {e}")
        
        # Also check legacy transform_generic_* pattern
        for possible_dir in self.base_path.glob("transform_generic_*"):
            if possible_dir.is_dir():
                for jsonl_file in possible_dir.glob("*.jsonl"):
                    if jsonl_file.name != "final_total_output.jsonl":
                        continue  # Skip individual file outputs, use total
                    logger.info(f"Loading transformed generic data from {jsonl_file}")
                    try:
                        dataset = load_dataset("json", data_files=str(jsonl_file), split="train")
                        dataset = dataset.add_column("data_type", ["generic_transformed"] * len(dataset))
                        dataset = dataset.add_column("format_type", ["conversations"] * len(dataset))
                        datasets.append(dataset)
                    except Exception as e:
                        logger.warning(f"Failed to load transformed generic data: {e}")
        
        # Look for generic dataset configuration (external datasets)
        config_path = self.base_path / "dataset_config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                dataset_config = yaml.safe_load(f)
            
            if "generic_dataset_paths" in dataset_config:
                for dataset_info in dataset_config["generic_dataset_paths"]:
                    path = dataset_info.get("path")
                    if path:
                        try:
                            # Try loading from HuggingFace Hub or local path
                            dataset = load_dataset(path, split="train")
                            dataset = dataset.add_column("data_type", [f"generic_{path.split('/')[-1]}"] * len(dataset))
                            dataset = dataset.add_column("format_type", ["conversations"] * len(dataset))
                            datasets.append(dataset)
                            logger.info(f"Loaded generic dataset: {path}")
                        except Exception as e:
                            logger.warning(f"Failed to load generic dataset {path}: {e}")
        
        return datasets
    
    def load_multimodal_data(self) -> Dataset:
        """Load multimodal (vision) data if available."""
        datasets = []
        
        # Look for multimodal data in various formats
        multimodal_files = [
            "multimodal_data.jsonl",
            "vision_data.jsonl", 
            "image_text_data.jsonl"
        ]
        
        for filename in multimodal_files:
            data_path = self.base_path / filename
            if data_path.exists():
                logger.info(f"Loading multimodal data from {data_path}")
                try:
                    dataset = load_dataset("json", data_files=str(data_path), split="train")
                    dataset = dataset.add_column("data_type", ["multimodal"] * len(dataset))
                    dataset = dataset.add_column("format_type", ["multimodal"] * len(dataset))
                    datasets.append(dataset)
                except Exception as e:
                    logger.warning(f"Failed to load multimodal data from {filename}: {e}")
        
        # Look for multimodal data in sft_run directory
        sft_run_dir = self.base_path / "sft_run"
        if sft_run_dir.exists():
            for jsonl_file in sft_run_dir.glob("*multimodal*.jsonl"):
                logger.info(f"Loading multimodal data from {jsonl_file}")
                try:
                    dataset = load_dataset("json", data_files=str(jsonl_file), split="train")
                    dataset = dataset.add_column("data_type", ["multimodal_sft"] * len(dataset))
                    dataset = dataset.add_column("format_type", ["multimodal"] * len(dataset))
                    datasets.append(dataset)
                except Exception as e:
                    logger.warning(f"Failed to load multimodal data from {jsonl_file}: {e}")
        
        if datasets:
            return concatenate_datasets(datasets)
        else:
            logger.warning("No multimodal data found")
            return None
    
    def load_all_data(self) -> Dict[str, Dataset]:
        """Load all specified data types."""
        data_dict = {}
        
        for data_type in self.config.data_types:
            if data_type == "factual_sft":
                dataset = self.load_factual_sft_data()
            elif data_type == "pretrain":
                dataset = self.load_pretrain_data()
            elif data_type == "rag_data":
                dataset = self.load_rag_data()
            elif data_type == "representation_variation":
                dataset = self.load_representation_variation_data()
            elif data_type == "correction":
                dataset = self.load_correction_data()
            elif data_type == "multimodal":
                dataset = self.load_multimodal_data()
            elif data_type == "generic_sft":
                # Load generic datasets if requested
                generic_datasets = self.load_generic_datasets()
                if generic_datasets:
                    dataset = concatenate_datasets(generic_datasets)
                else:
                    dataset = None
            else:
                logger.warning(f"Unknown data type: {data_type}")
                continue
            
            if dataset is not None:
                # Apply sampling if max_samples_per_type is set
                if len(dataset) > self.config.max_samples_per_type:
                    dataset = dataset.shuffle(seed=self.config.seed)
                    dataset = dataset.select(range(self.config.max_samples_per_type))
                
                data_dict[data_type] = dataset
                logger.info(f"Loaded {len(dataset)} samples for {data_type}")
        
        return data_dict
    
    def validate_data_availability(self) -> Dict[str, bool]:
        """Validate which data types are actually available in the data directory."""
        logger.info("🔍 Validating data availability...")
        
        availability = {}
        
        # Check factual SFT
        factual_sft_dirs = list(self.base_path.glob("factual_sft_facts_*"))
        availability["factual_sft"] = bool(factual_sft_dirs)
        if availability["factual_sft"]:
            logger.info(f"✅ Factual SFT: Found {len(factual_sft_dirs)} directories")
            # Log SFT types found
            sft_types = set()
            for d in factual_sft_dirs:
                parts = d.name.split('_')
                if len(parts) >= 4:
                    sft_types.add(parts[3])  # factual_sft_facts_{type}_{num}
            logger.info(f"   SFT types: {sorted(sft_types)}")
        else:
            logger.warning("❌ Factual SFT: No factual_sft_facts_* directories found")
        
        # Check pretraining data
        pretrain_dir = self.base_path / "pretraining_run"
        availability["pretrain"] = pretrain_dir.exists()
        if availability["pretrain"]:
            pretrain_files = list(pretrain_dir.glob("*.jsonl"))
            logger.info(f"✅ Pretraining: Found {len(pretrain_files)} files in pretraining_run/")
            logger.info(f"   Files: {[f.name for f in pretrain_files]}")
        else:
            logger.warning("❌ Pretraining: No pretraining_run/ directory found")
        
        # Check RAG data (in factual SFT directories)
        rag_files = []
        for d in factual_sft_dirs:
            rag_file = d / "simplified_data_rag.jsonl"
            if rag_file.exists():
                rag_files.append(rag_file)
        availability["rag_data"] = bool(rag_files)
        if availability["rag_data"]:
            logger.info(f"✅ RAG Data: Found {len(rag_files)} simplified_data_rag.jsonl files")
        else:
            logger.warning("❌ RAG Data: No simplified_data_rag.jsonl files found")
        
        # Check representation variation
        repr_var_files = []
        if pretrain_dir.exists():
            repr_var_files.extend(pretrain_dir.glob("representation_variation_*.jsonl"))
        repr_var_dirs = list(self.base_path.glob("representation_variation_*"))
        availability["representation_variation"] = len(repr_var_files) > 0 or len(repr_var_dirs) > 0
        if availability["representation_variation"]:
            logger.info(f"✅ Representation Variation: Found {len(repr_var_files)} files + {len(repr_var_dirs)} directories")
        else:
            logger.warning("❌ Representation Variation: No representation variation data found")
        
        # Check correction data (including main corrections_facts directory)
        correction_dirs = list(self.base_path.glob("correction*"))  # Includes corrections_facts
        sft_correction = (self.base_path / "sft_run").exists() and \
                        len(list((self.base_path / "sft_run").glob("axolotl_correction_*"))) > 0
        main_corrections = (self.base_path / "corrections_facts" / "axolotl_correction_conversations.json").exists()
        availability["correction"] = bool(correction_dirs) or sft_correction or main_corrections
        if availability["correction"]:
            logger.info(f"✅ Correction: Found {len(correction_dirs)} directories + main corrections_facts")
        else:
            logger.info("ℹ️ Correction: No correction data found (optional)")
        
        # Check multimodal data
        multimodal_files = []
        multimodal_files.extend(self.base_path.glob("*multimodal*.jsonl"))
        multimodal_files.extend(self.base_path.glob("*vision*.jsonl"))
        multimodal_files.extend(self.base_path.glob("*image*.jsonl"))
        availability["multimodal"] = bool(multimodal_files)
        if availability["multimodal"]:
            logger.info(f"✅ Multimodal: Found {len(multimodal_files)} files")
        else:
            logger.info("ℹ️ Multimodal: No multimodal data found (optional)")
        
        # Check generic SFT data (your large datasets)
        generic_sft_dir = self.base_path / "generic_sft"
        generic_completion_dir = self.base_path / "sft_run" / "generic_sft_completion"
        transformed_dirs = list(self.base_path.glob("transformed_generic_data_*"))
        availability["generic_sft"] = generic_sft_dir.exists() or generic_completion_dir.exists() or bool(transformed_dirs)
        if availability["generic_sft"]:
            logger.info(f"✅ Generic SFT: Found main dir + completion dir + {len(transformed_dirs)} transformed dirs")
        else:
            logger.info("ℹ️ Generic SFT: No generic SFT data found (optional)")
        
        # Summary
        available_count = sum(availability.values())
        total_checked = len(availability)
        logger.info(f"📊 Data availability summary: {available_count}/{total_checked} data types available")
        
        # Filter config data types to only include available ones
        if hasattr(self.config, 'data_types'):
            available_data_types = [dt for dt in self.config.data_types if availability.get(dt, False)]
            unavailable_data_types = [dt for dt in self.config.data_types if not availability.get(dt, False)]
            
            if unavailable_data_types:
                logger.warning(f"⚠️ Configured but unavailable data types: {unavailable_data_types}")
                logger.info(f"✅ Will proceed with available data types: {available_data_types}")
                # Update config to only include available types
                self.config.data_types = available_data_types
        
        return availability


class AugmentoolkitTrainer:
    """Main trainer class for Augmentoolkit fine-tuning with Unsloth."""
    
    def __init__(self, config: AugmentoolkitConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.data_loader = AugmentoolkitDataLoader(config)
        self.current_stage = None
        
        # Set random seeds
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
    
    def get_stage_config(self, stage: str) -> dict:
        """Get stage-specific configuration overrides."""
        if stage == "pretrain":
            return {
                "use_lora": False,           # Disable LoRA for full finetuning
                "full_finetuning": True,     # Enable full finetuning
                "learning_rate": 5e-5,       # Lower LR for pretraining
                "per_device_train_batch_size": 2,  # Can use larger batch with LoRA
            }
        else:  # sft stage
            return {
                "use_lora": False,           # Disable LoRA for full finetuning
                "full_finetuning": True,     # Enable full finetuning
                "learning_rate": 2e-4,       # Higher LR for SFT
            }
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer using Unsloth."""
        logger.info(f"Loading model with Unsloth: {self.config.model_name}")
        
        if not UNSLOTH_AVAILABLE:
            raise RuntimeError("Unsloth is required but not available. Install with: pip install unsloth")
        
        # Update CUDA_VISIBLE_DEVICES to the configured GPU
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.config.gpu_device)
        logger.info(f"Using GPU {self.config.gpu_device}: CUDA_VISIBLE_DEVICES={self.config.gpu_device}")
        logger.info("Single GPU mode enforced to prevent DataParallel issues")
        
        # Load model using Unsloth based on model type
        if self.config.model_type == "multimodal":
            # Load multimodal model (Gemma 3N)
            self.model, self.tokenizer = FastModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=self.config.max_seq_length,
                dtype=None,  # Auto-detect best dtype
                load_in_4bit=self.config.load_in_4bit,
                device_map={"": 0},  # Force everything on GPU 0 (relative to visible devices)
                full_finetuning=self.config.full_finetuning,
            )
            # For multimodal models, the tokenizer is actually a processor
            self.processor = self.tokenizer
            logger.info("✅ Loaded multimodal model with Unsloth FastModel")
            
        elif self.config.model_type == "vision":
            # Load vision model
            self.model, self.processor = FastVisionModel.from_pretrained(
                self.config.model_name,
                load_in_4bit=self.config.load_in_4bit,
                use_gradient_checkpointing=self.config.use_gradient_checkpointing,
            )
            self.tokenizer = self.processor.tokenizer
            logger.info("✅ Loaded vision model with Unsloth FastVisionModel")
            
        else:
            # Load text-only model
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=self.config.max_seq_length,
                dtype=None,
                load_in_4bit=self.config.load_in_4bit,
            )
            logger.info("✅ Loaded text model with Unsloth FastLanguageModel")
        
        # Apply chat template
        if self.config.model_type in ["text", "multimodal"]:
            self.tokenizer = get_chat_template(
                self.tokenizer,
                chat_template=self.config.chat_template,
            )
            logger.info("✅ Applied chat template")
        
        # Test chat template
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            try:
                test_messages = [
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
                formatted = self.tokenizer.apply_chat_template(test_messages, tokenize=False)
                logger.info(f"✅ Chat template test: {formatted[:100]}...")
            except Exception as e:
                logger.warning(f"Chat template test failed: {e}")
        
        logger.info("✅ Model and tokenizer loaded successfully with Unsloth")
    
    def reconfigure_model_for_stage(self, stage: str):
        """Reconfigure model for a specific training stage using Unsloth."""
        logger.info(f"Configuring model for {stage} stage with Unsloth")
        
        stage_config = self.get_stage_config(stage)
        use_lora = stage_config.get("use_lora", self.config.use_lora)
        
        # Apply LoRA configuration based on stage and model type
        if use_lora:
            logger.info(f"Applying LoRA for {stage} stage")
            
            if self.config.model_type == "multimodal":
                # Apply LoRA for multimodal (Gemma 3N)
                self.model = FastModel.get_peft_model(
                    self.model,
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                                  "gate_proj", "up_proj", "down_proj"],
                    bias="none",
                    use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                    random_state=self.config.seed,
                )
                FastModel.for_training(self.model)
                
            elif self.config.model_type == "vision":
                # Apply LoRA for vision model
                self.model = FastVisionModel.get_peft_model(
                    self.model,
                    finetune_vision_layers=self.config.finetune_vision_layers,
                    finetune_language_layers=self.config.finetune_language_layers,
                    finetune_attention_modules=self.config.finetune_attention_modules,
                    finetune_mlp_modules=self.config.finetune_mlp_modules,
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=self.config.target_modules,
                )
                FastVisionModel.for_training(self.model)
                
            else:
                # Apply LoRA for text model
                self.model = FastLanguageModel.get_peft_model(
                    self.model,
                    r=self.config.lora_r,
                    target_modules=self.config.target_modules,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    bias="none",
                    use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                    random_state=self.config.seed,
                )
                FastLanguageModel.for_training(self.model)
                
        else:
            # Full fine-tuning (no LoRA)
            logger.info(f"Using full fine-tuning for {stage} stage")
            if self.config.model_type == "multimodal":
                FastModel.for_training(self.model)
            elif self.config.model_type == "vision":
                FastVisionModel.for_training(self.model)
            else:
                FastLanguageModel.for_training(self.model)
        
        logger.info(f"✅ Model configured for {stage} stage: LoRA={use_lora}")
    
    def format_conversations(self, examples):
        """Format conversations based on data type and format."""
        # Get format type and data type
        format_types = examples.get("format_type", ["conversations"] * len(examples.get("text", examples.get("conversations", [""]))))
        data_types = examples.get("data_type", ["unknown"] * len(examples.get("text", examples.get("conversations", [""]))))
        
        texts = []
        
        # Process each example
        for i in range(len(format_types)):
            format_type = format_types[i]
            data_type = data_types[i]
            
            try:
                if format_type == "text":
                    # Raw text for pretraining - no chat template
                    if "text" in examples:
                        text = examples["text"][i]
                    else:
                        text = ""
                
                elif format_type == "conversations":
                    # ShareGPT conversation format - apply chat template
                    if "conversations" in examples:
                        conversations = examples["conversations"][i]
                        
                        # Convert ShareGPT format to standard chat format
                        messages = []
                        for turn in conversations:
                            role_mapping = {
                                "human": "user",
                                "user": "user", 
                                "gpt": "assistant",
                                "assistant": "assistant",
                                "system": "system"
                            }
                            role = role_mapping.get(turn.get("from", "user"), "user")
                            content = turn.get("value", "")
                            messages.append({"role": role, "content": content})
                        
                        # Apply chat template
                        text = self.tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=False
                        )
                    else:
                        text = ""
                
                elif format_type == "completion":
                    # Completion format - use as-is or extract text
                    if "text" in examples:
                        text = examples["text"][i]
                    elif "full_input" in examples and "full_response" in examples:
                        # Meta dataset completion format
                        text = examples["full_input"][i] + examples["full_response"][i]
                    else:
                        text = ""
                
                elif format_type == "segments":
                    # Segments format - process for loss masking
                    if "segments" in examples:
                        segments = examples["segments"][i]
                        # Concatenate all segments
                        text = "".join([seg.get("text", "") for seg in segments])
                    else:
                        text = ""
                
                else:
                    # Fallback - try to extract any text
                    if "text" in examples:
                        text = examples["text"][i]
                    elif "conversations" in examples:
                        # Try to apply chat template as fallback
                        try:
                            conversations = examples["conversations"][i]
                            messages = []
                            for turn in conversations:
                                role = "user" if turn.get("from") in ["human", "user"] else "assistant"
                                content = turn.get("value", "")
                                messages.append({"role": role, "content": content})
                            text = self.tokenizer.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=False
                            )
                        except:
                            text = str(examples["conversations"][i])
                    else:
                        text = ""
                
                texts.append(text)
                
            except Exception as e:
                logger.warning(f"Failed to format example {i} (type: {data_type}, format: {format_type}): {e}")
                texts.append("")
        
        return {"text": texts}
    
    def prepare_datasets(self) -> Dict[str, Dataset]:
        """Prepare datasets for training."""
        logger.info("Loading and preparing datasets...")
        
        # Setup caching
        cache_enabled = self.config.enable_dataset_caching
        cache_dir = Path(self.config.cache_dir)
        if cache_enabled:
            cache_dir.mkdir(exist_ok=True)
            logger.info(f"Dataset caching enabled. Cache directory: {cache_dir}")
        
        # Generate cache key based on config
        import hashlib
        cache_key_data = {
            "model_name": self.config.model_name,
            "max_seq_length": self.config.max_seq_length,
            "chat_template": self.config.chat_template,
            "training_stages": self.config.training_stages,
            "data_types": self.config.data_types,
            "data_base_path": self.config.data_base_path,
        }
        cache_key = hashlib.md5(str(sorted(cache_key_data.items())).encode()).hexdigest()[:8]
        cache_file = cache_dir / f"cache_info_{cache_key}.json"
        
        # Try to load from cache first
        if cache_enabled and cache_file.exists() and not self.config.force_reprocess:
            logger.info(f"Loading preprocessed datasets from cache: {cache_file}")
            try:
                cached_datasets = {}

                import json
                with open(cache_file, 'r') as f:
                    cache_info = json.load(f)
                
                for data_type in cache_info['datasets']:
                    dataset_cache_path = cache_dir / f"{data_type}_{cache_key}"
                    if dataset_cache_path.exists():
                        cached_datasets[data_type] = load_from_disk(str(dataset_cache_path))
                        logger.info(f"Loaded cached dataset: {data_type} ({len(cached_datasets[data_type])} samples)")
                
                if cached_datasets:
                    logger.info(f"✅ Successfully loaded {len(cached_datasets)} datasets from cache")
                    return cached_datasets
                    
            except Exception as e:
                logger.warning(f"Failed to load cached datasets: {e}. Proceeding with fresh processing.")
        
        # First validate what data is available
        self.data_loader.validate_data_availability()
        
        # Load all data
        data_dict = self.data_loader.load_all_data()
        
        if not data_dict:
            raise ValueError("No datasets loaded. Check your data paths.")
        
        # Apply data mixing ratios if specified
        if self.config.data_mixing_ratios:
            for data_type, ratio in self.config.data_mixing_ratios.items():
                if data_type in data_dict:
                    dataset = data_dict[data_type]
                    target_size = int(len(dataset) * ratio)
                    if target_size < len(dataset):
                        dataset = dataset.shuffle(seed=self.config.seed)
                        dataset = dataset.select(range(target_size))
                        data_dict[data_type] = dataset
        
        # Format datasets based on their format type
        formatted_datasets = {}
        for data_type, dataset in data_dict.items():
            logger.info(f"Processing dataset: {data_type} with {len(dataset)} samples")
            logger.info(f"Dataset columns: {dataset.column_names}")
            
            try:
                # Check if this is segments data
                sample_row = dataset[0] if dataset else {}
                if "segments" in sample_row:
                    # Preprocess segments data efficiently
                    logger.info(f"Preprocessing {data_type} segments data with loss masking")
                    preprocessed_dataset = dataset.map(
                        lambda examples: preprocess_segments_data(examples, self.tokenizer, self.config.max_seq_length),
                        batched=True,
                        remove_columns=[col for col in dataset.column_names if col not in ["input_ids", "labels"]],
                        desc=f"Preprocessing {data_type} segments"
                    )
                    formatted_datasets[data_type] = preprocessed_dataset
                else:
                    # Apply text formatting and preprocessing for other data types
                    formatted_dataset = dataset.map(
                        self.format_conversations,
                        batched=True,
                        remove_columns=[col for col in dataset.column_names if col not in ["data_type", "format_type"]],
                        desc=f"Formatting {data_type} conversations"
                    )
                    
                    # Determine if this is pretraining data that needs tokenization
                    pretrain_keywords = {"pretrain", "representation_variation", "inferred_facts", "text_chunks"}
                    is_pretrain_data = any(keyword in data_type for keyword in pretrain_keywords)
                    
                    logger.info(f"Dataset {data_type}: is_pretrain_data={is_pretrain_data}, training_stages={self.config.training_stages}")
                    
                    if is_pretrain_data and "pretrain" in self.config.training_stages:
                        # For multimodal models, format_conversations returns original data
                        # So we need to extract text directly for pretraining
                        if self.config.model_type == "multimodal":
                            # Extract text directly from original dataset for pretraining
                            logger.info(f"Preprocessing {data_type} for multimodal pretraining (language modeling)")
                            preprocessed_dataset = dataset.map(
                                lambda examples: preprocess_pretraining_data(
                                    {"text": examples["text"]}, self.tokenizer, self.config.max_seq_length
                                ),
                                batched=True,
                                remove_columns=[col for col in dataset.column_names if col not in ["input_ids", "labels"]],
                                desc=f"Tokenizing {data_type} for multimodal pretraining"
                            )
                        else:
                            # Apply pretraining tokenization for language modeling
                            logger.info(f"Preprocessing {data_type} for pretraining (language modeling)")
                            preprocessed_dataset = formatted_dataset.map(
                                lambda examples: preprocess_pretraining_data(
                                    examples, self.tokenizer, self.config.max_seq_length
                                ),
                                batched=True,
                                remove_columns=[col for col in formatted_dataset.column_names if col not in ["input_ids", "labels"]],
                                desc=f"Tokenizing {data_type} for pretraining"
                            )
                        formatted_datasets[data_type] = preprocessed_dataset
                        logger.info(f"Final preprocessed dataset {data_type} columns: {preprocessed_dataset.column_names}")
                    # Apply efficient preprocessing for response-only training (SFT data)
                    elif "sft" in self.config.training_stages and not is_pretrain_data:
                        response_template = "<start_of_turn>model\n" if "gemma" in self.config.chat_template else "<|im_start|>assistant\n"
                        logger.info(f"Preprocessing {data_type} for SFT (response-only training)")
                        preprocessed_dataset = formatted_dataset.map(
                            lambda examples: preprocess_response_only_data(
                                examples, self.tokenizer, response_template, self.config.max_seq_length
                            ),
                            batched=True,
                            remove_columns=[col for col in formatted_dataset.column_names if col not in ["input_ids", "labels"]],
                            desc=f"Preprocessing {data_type} for response-only training"
                        )
                        formatted_datasets[data_type] = preprocessed_dataset
                        logger.info(f"Final preprocessed dataset {data_type} columns: {preprocessed_dataset.column_names}")
                    else:
                        logger.info(f"No preprocessing applied to {data_type} - using formatted dataset")
                        formatted_datasets[data_type] = formatted_dataset
                        logger.info(f"Final dataset {data_type} columns: {formatted_dataset.column_names}")
            except Exception as e:
                logger.warning(f"Failed to format {data_type}: {e}")
                continue
        
        # Save processed datasets to cache
        if cache_enabled and formatted_datasets:
            logger.info(f"Saving {len(formatted_datasets)} processed datasets to cache...")
            try:
                import json
                
                # Save each dataset separately
                saved_datasets = []
                for data_type, dataset in formatted_datasets.items():
                    dataset_cache_path = cache_dir / f"{data_type}_{cache_key}"
                    dataset.save_to_disk(str(dataset_cache_path))
                    saved_datasets.append(data_type)
                    logger.info(f"Cached dataset: {data_type} → {dataset_cache_path}")
                
                # Save cache metadata
                cache_info = {
                    "cache_key": cache_key,
                    "datasets": saved_datasets,
                    "config": cache_key_data,
                    "timestamp": str(pd.Timestamp.now()) if 'pd' in globals() else str(datetime.now()),
                    "total_datasets": len(saved_datasets)
                }
                
                cache_file = cache_dir / f"cache_info_{cache_key}.json"
                with open(cache_file, 'w') as f:
                    json.dump(cache_info, f, indent=2)
                
                logger.info(f"✅ Successfully cached {len(saved_datasets)} datasets with key: {cache_key}")
                
            except Exception as e:
                logger.warning(f"Failed to save datasets to cache: {e}")
        
        return formatted_datasets
    
    def train_stage(self, stage: str, datasets: Dict[str, Dataset]):
        """Train a specific stage (pretrain or sft) using Unsloth."""
        logger.info(f"Starting {stage} training stage with Unsloth")
        
        # Set current stage and get stage-specific config
        self.current_stage = stage
        stage_config = self.get_stage_config(stage)
        logger.info(f"Stage config for {stage}: {stage_config}")
        
        # Reconfigure model for this stage (handle LoRA vs full finetuning)
        self.reconfigure_model_for_stage(stage)
        
        # Filter datasets for this stage based on data content and purpose
        stage_datasets = {}
        
        # Define data types for each stage
        pretrain_keywords = {"pretrain", "representation_variation", "inferred_facts", "text_chunks"}
        
        if stage == "pretrain":
            # Include pretraining-oriented data
            stage_datasets = {k: v for k, v in datasets.items() 
                            if any(keyword in k for keyword in pretrain_keywords)}
        else:  # sft stage
            # Include non-pretraining data (but allow pretrain_subset for mixed training)
            stage_datasets = {k: v for k, v in datasets.items() 
                            if not ("pretrain" in k and "pretrain_subset" not in k)}
        
        if not stage_datasets:
            logger.warning(f"No datasets found for {stage} stage")
            return
        
        logger.info(f"Using {len(stage_datasets)} dataset types for {stage}: {list(stage_datasets.keys())}")
        
        # Separate segments data from text data for different handling
        segments_datasets = []
        text_datasets = []
        
        for name, dataset in stage_datasets.items():
            sample_row = dataset[0] if dataset else {}
            if "segments" in sample_row:
                segments_datasets.append(dataset)
                logger.info(f"Adding {name} to segments datasets")
            else:
                text_datasets.append(dataset)
                logger.info(f"Adding {name} to text datasets")
        
        # Combine and prioritize datasets
        if not text_datasets and not segments_datasets:
            logger.error(f"No valid datasets found for {stage} stage")
            raise ValueError(f"No datasets available for {stage} training. Check your data configuration.")
        
        # Prefer segments data (for loss masking), otherwise use text data
        if segments_datasets:
            combined_dataset = concatenate_datasets(segments_datasets)
            logger.info(f"Using segments dataset for {stage} training (supports loss masking)")
        else:
            combined_dataset = concatenate_datasets(text_datasets)
            logger.info(f"Using text dataset for {stage} training")
        
        logger.info(f"Combined dataset size for {stage}: {len(combined_dataset)}")
        
        # Validate dataset is not empty
        if len(combined_dataset) == 0:
            raise ValueError(f"Combined dataset for {stage} is empty after processing")
        
        # Check if this is primarily segments data
        is_segments_data = combined_dataset and "segments" in combined_dataset[0]
        
        # Split train/validation
        if self.config.validation_split > 0:
            split_dataset = combined_dataset.train_test_split(
                test_size=self.config.validation_split,
                seed=self.config.seed
            )
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
        else:
            train_dataset = combined_dataset
            eval_dataset = None
        
        # Setup training arguments - use different configs for pretrain vs SFT
        # Apply stage-specific overrides
        learning_rate = stage_config.get("learning_rate", self.config.learning_rate)
        batch_size = stage_config.get("per_device_train_batch_size", self.config.per_device_train_batch_size)
        
        if stage == "pretrain":
            # For pretraining, use standard TrainingArguments (language modeling)
            training_args = TrainingArguments(
                output_dir=f"{self.config.output_dir}_{stage}",
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=self.config.per_device_eval_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=learning_rate,
                warmup_ratio=self.config.warmup_ratio,
                num_train_epochs=self.config.num_train_epochs,
                max_steps=self.config.max_steps,
                weight_decay=self.config.weight_decay,
                lr_scheduler_type=self.config.lr_scheduler_type,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                eval_steps=self.config.eval_steps,
                save_total_limit=self.config.save_total_limit,
                report_to=self.config.report_to,
                bf16=self.config.bf16,
                fp16=self.config.fp16,
                dataloader_num_workers=self.config.dataloader_num_workers,
                remove_unused_columns=self.config.remove_unused_columns,
                gradient_checkpointing=self.config.use_gradient_checkpointing,  # ✅ Use config setting (Dynamo limits fixed)
                seed=self.config.seed,
                # Use standard language modeling objective
                prediction_loss_only=True,
                # Explicitly force single GPU mode
                dataloader_drop_last=False,
                ddp_find_unused_parameters=False,
                local_rank=-1,  # Force no distributed training
            )
        else:
            # For SFT, use SFTConfig
            training_args = SFTConfig(
                output_dir=f"{self.config.output_dir}_{stage}",
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=self.config.per_device_eval_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=learning_rate,
                warmup_ratio=self.config.warmup_ratio,
                num_train_epochs=self.config.num_train_epochs,
                max_steps=self.config.max_steps,
                weight_decay=self.config.weight_decay,
                lr_scheduler_type=self.config.lr_scheduler_type,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                eval_steps=self.config.eval_steps,
                save_total_limit=self.config.save_total_limit,
                report_to=self.config.report_to,
                bf16=self.config.bf16,
                fp16=self.config.fp16,
                dataloader_num_workers=self.config.dataloader_num_workers,
                remove_unused_columns=self.config.remove_unused_columns,
                gradient_checkpointing=self.config.use_gradient_checkpointing,  # ✅ Use config setting (Dynamo limits fixed)
                seed=self.config.seed,
                # Explicitly force single GPU mode
                dataloader_drop_last=False,
                ddp_find_unused_parameters=False,
                local_rank=-1,  # Force no distributed training
            )
        
        # Setup data collator based on data type and stage
        data_collator = None
        
        # Check if we have multimodal data
        is_multimodal_data = any("multimodal" in name for name in stage_datasets.keys())
        
        if is_segments_data:
            # Use segments data collator for loss masking
            # Extract actual tokenizer for multimodal models
            actual_tokenizer = self.tokenizer
            if hasattr(self.tokenizer, 'tokenizer'):
                actual_tokenizer = self.tokenizer.tokenizer
            elif hasattr(self, 'processor') and hasattr(self.processor, 'tokenizer'):
                actual_tokenizer = self.processor.tokenizer
                
            data_collator = SegmentsDataCollator(actual_tokenizer, pad_to_multiple_of=8)
            logger.info("Using SegmentsDataCollator for loss masking")
        elif is_multimodal_data and self.config.model_type == "multimodal":
            # Use multimodal data collator for image-text data
            if hasattr(self, 'processor') and self.processor:
                data_collator = MultimodalDataCollator(self.processor, pad_to_multiple_of=8)
                logger.info("Using MultimodalDataCollator for image-text data")
            else:
                logger.warning("No processor available for multimodal data, falling back to text-only")
        elif stage == "pretrain":
            # For pretraining, don't use any special data collator here
            # Will be set up in trainer setup with DataCollatorForLanguageModeling
            data_collator = None
            logger.info("Will use DataCollatorForLanguageModeling for pretraining")
        elif not is_segments_data and stage != "pretrain":
            # Use response-only data collator when not using Unsloth for SFT conversation data
            # Extract actual tokenizer for multimodal models
            actual_tokenizer = self.tokenizer
            if hasattr(self.tokenizer, 'tokenizer'):
                actual_tokenizer = self.tokenizer.tokenizer
            elif hasattr(self, 'processor') and hasattr(self.processor, 'tokenizer'):
                actual_tokenizer = self.processor.tokenizer
                
            data_collator = ResponseOnlyDataCollator(
                actual_tokenizer, 
                pad_to_multiple_of=8
            )
            logger.info("Using ResponseOnlyDataCollator for response-only training")
        
        # Setup trainer - use different trainer types for pretrain vs SFT
        if stage == "pretrain":            
            # Use language modeling data collator for pretraining if no custom collator
            if data_collator is None:
                # For multimodal models, extract tokenizer from processor
                actual_tokenizer = self.tokenizer
                if hasattr(self.tokenizer, 'tokenizer'):
                    actual_tokenizer = self.tokenizer.tokenizer
                elif hasattr(self, 'processor') and hasattr(self.processor, 'tokenizer'):
                    actual_tokenizer = self.processor.tokenizer
                
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=actual_tokenizer,
                    mlm=False,  # Causal LM, not masked LM
                    pad_to_multiple_of=8,  # Efficient for tensor cores
                    return_tensors="pt",
                )
                logger.info("Using DataCollatorForLanguageModeling for pretraining")
            
            if UNSLOTH_AVAILABLE:
                if self.config.model_type == "text":
                    FastLanguageModel.for_training(self.model)
                elif self.config.model_type == "vision":
                    FastVisionModel.for_training(self.model)
                elif self.config.model_type == "multimodal":
                    FastModel.for_training(self.model)
            
            # Use standard Trainer for pretraining
            trainer = Trainer(
                model=self.model,
                processing_class=actual_tokenizer,  # Use actual tokenizer, not processor
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                args=training_args,
            )
            logger.info("Using standard Trainer for pretraining stage")
            
        else:
            # For SFT, use SFTTrainer
            if UNSLOTH_AVAILABLE:
                if self.config.model_type == "vision":
                    FastVisionModel.for_training(self.model)
                    
                    # Extract actual tokenizer for multimodal models
                    actual_tokenizer = self.tokenizer
                    if hasattr(self.tokenizer, 'tokenizer'):
                        actual_tokenizer = self.tokenizer.tokenizer
                    elif hasattr(self, 'processor') and hasattr(self.processor, 'tokenizer'):
                        actual_tokenizer = self.processor.tokenizer
                    
                    trainer = SFTTrainer(
                        model=self.model,
                        tokenizer=actual_tokenizer,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        data_collator=data_collator or UnslothVisionDataCollator(self.model, self.processor),
                        args=training_args,
                    )
                elif self.config.model_type == "multimodal":
                    FastModel.for_training(self.model)
                    
                    # Extract actual tokenizer for multimodal models
                    actual_tokenizer = self.tokenizer
                    if hasattr(self.tokenizer, 'tokenizer'):
                        actual_tokenizer = self.tokenizer.tokenizer
                    elif hasattr(self, 'processor') and hasattr(self.processor, 'tokenizer'):
                        actual_tokenizer = self.processor.tokenizer
                    
                    trainer = SFTTrainer(
                        model=self.model,
                        tokenizer=actual_tokenizer,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        data_collator=data_collator,
                        args=training_args,
                    )
                else:
                    FastLanguageModel.for_training(self.model)
                    
                    # Extract actual tokenizer for consistency
                    actual_tokenizer = self.tokenizer
                    if hasattr(self.tokenizer, 'tokenizer'):
                        actual_tokenizer = self.tokenizer.tokenizer
                    elif hasattr(self, 'processor') and hasattr(self.processor, 'tokenizer'):
                        actual_tokenizer = self.processor.tokenizer
                    
                    trainer = SFTTrainer(
                        model=self.model,
                        tokenizer=actual_tokenizer,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        data_collator=data_collator,
                        args=training_args,
                    )
                    
                    # Apply response-only training for text models (only if not using segments)
                    if not is_segments_data:
                        trainer = train_on_responses_only(
                            trainer,
                            instruction_part="<start_of_turn>user\n" if "gemma" in self.config.chat_template else "<|im_start|>user\n",
                            response_part="<start_of_turn>model\n" if "gemma" in self.config.chat_template else "<|im_start|>assistant\n",
                        )
            else:
                # Standard SFTTrainer for non-Unsloth SFT
                if self.config.model_type == "multimodal" and hasattr(self, 'processor'):
                    # For multimodal models, use the processor instead of tokenizer
                    trainer = SFTTrainer(
                        model=self.model,
                        processing_class=self.processor,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        data_collator=data_collator,
                        args=training_args,
                    )
                else:
                    # Standard text-only SFT
                    trainer = SFTTrainer(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        data_collator=data_collator,
                        args=training_args,
                    )
            logger.info("Using SFTTrainer for SFT stage")
        
        # # Add early stopping if evaluation is enabled
        # if eval_dataset:
        #     trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))
        
        # Train
        logger.info(f"Starting {stage} training...")
        
        # Log system info for debugging
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {gpu_props.name} ({gpu_props.total_memory / 1024**3:.1f}GB)")
                if i == torch.cuda.current_device():
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    cached = torch.cuda.memory_reserved(i) / 1024**3
                    logger.info(f"GPU {i} Memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached")
        
        try:
            trainer.train()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("CUDA out of memory! Try:")
                logger.error("1. Reduce batch size: --batch-size 1")
                logger.error("2. Enable gradient checkpointing in config")
                logger.error("3. Use --use-unsloth for single GPU mode")
                raise
            else:
                raise
        
        # Save model
        trainer.save_model()
        if hasattr(self, 'tokenizer') and self.tokenizer:
            self.tokenizer.save_pretrained(f"{self.config.output_dir}_{stage}")
        if hasattr(self, 'processor') and self.processor:
            self.processor.save_pretrained(f"{self.config.output_dir}_{stage}")
        
        logger.info(f"Completed {stage} training stage")
        
        return trainer
    
    def run_training(self):
        """Run the complete training pipeline."""
        logger.info("Starting Augmentoolkit fine-tuning...")
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Prepare datasets
        datasets = self.prepare_datasets()
        
        # Run training stages
        for stage in self.config.training_stages:
            trainer = self.train_stage(stage, datasets)
            
            # Update model for next stage if needed
            if len(self.config.training_stages) > 1:
                logger.info(f"Completed stage: {stage}")
        
        logger.info("Training completed!")
        
        # Save final model with Unsloth
        try:
            if self.config.model_type == "vision":
                FastVisionModel.for_inference(self.model)
            elif self.config.model_type == "text":
                FastLanguageModel.for_inference(self.model)
            elif self.config.model_type == "multimodal":
                FastModel.for_inference(self.model)
            
            final_output_dir = f"{self.config.output_dir}_final"
            
            # Check if we're using LoRA or full fine-tuning
            use_lora = self.config.use_lora
            
            if use_lora:
                # For LoRA: save merged model
                logger.info("Saving LoRA merged model...")
                self.model.save_pretrained_merged(final_output_dir, self.tokenizer)
                logger.info(f"Saved LoRA merged model to {final_output_dir}")
            else:
                # For full fine-tuning: save normally
                logger.info("Saving full fine-tuned model...")
                self.model.save_pretrained(final_output_dir)
                
                # Save tokenizer/processor
                if hasattr(self, 'tokenizer') and self.tokenizer:
                    if hasattr(self.tokenizer, 'save_pretrained'):
                        self.tokenizer.save_pretrained(final_output_dir)
                if hasattr(self, 'processor') and self.processor:
                    self.processor.save_pretrained(final_output_dir)
                
                logger.info(f"Saved full fine-tuned model to {final_output_dir}")
            
            # Save GGUF if requested (Q8_0 for quality)
            try:
                self.model.save_pretrained_gguf(
                    final_output_dir,
                    quantization_type="Q8_0",  # High quality quantization
                    # Removed use_fast_tokenizer - not a valid parameter
                )
                logger.info(f"Saved GGUF model to {final_output_dir}")
            except Exception as e:
                logger.warning(f"Failed to save GGUF: {e}")
                 
        except Exception as e:
            logger.warning(f"Failed to save model with Unsloth: {e}")
            
            # Fallback: save standard model
            final_output_dir = f"{self.config.output_dir}_final"
            try:
                self.model.save_pretrained(final_output_dir)
                if hasattr(self, 'tokenizer') and self.tokenizer:
                    self.tokenizer.save_pretrained(final_output_dir)
                if hasattr(self, 'processor') and self.processor:
                    self.processor.save_pretrained(final_output_dir)
                logger.info(f"Saved model using fallback to {final_output_dir}")
            except Exception as fallback_e:
                logger.error(f"Fallback save also failed: {fallback_e}")
    
    def test_model(self, test_prompt: str = "What is artificial intelligence?"):
        """Test the trained model with a sample prompt using Unsloth."""
        if not self.model or not self.tokenizer:
            logger.error("Model not loaded. Run training first.")
            return
        
        logger.info("Testing trained model with Unsloth...")
        
        # Set model to inference mode with Unsloth
        if self.config.model_type == "vision":
            FastVisionModel.for_inference(self.model)
        elif self.config.model_type == "text":
            FastLanguageModel.for_inference(self.model)
        elif self.config.model_type == "multimodal":
            FastModel.for_inference(self.model)
        
        # Prepare input
        if self.config.model_type in ["vision", "multimodal"]:
            if hasattr(self, 'processor') and self.processor:
                # Use processor for multimodal models like Gemma 3N
                # Try standard chat format first
                try:
                    messages = [{"role": "user", "content": test_prompt}]
                    if hasattr(self.processor, 'apply_chat_template'):
                        input_text = self.processor.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=False
                        )
                    else:
                        input_text = None
                    
                    # If chat template failed or returned None, use raw prompt
                    if input_text is None:
                        input_text = test_prompt
                    
                    inputs = self.processor(
                        text=input_text,
                        return_tensors="pt",
                        add_special_tokens=True
                    ).to("cuda" if CUDA_AVAILABLE else "cpu")
                except Exception as e:
                    logger.warning(f"Failed to use chat template, using raw prompt: {e}")
                    # Fallback to simple text processing
                    inputs = self.processor(
                        text=test_prompt,
                        return_tensors="pt",
                        add_special_tokens=True
                    ).to("cuda" if CUDA_AVAILABLE else "cpu")
            else:
                # Fallback to tokenizer for vision models
                try:
                    messages = [{"role": "user", "content": test_prompt}]
                    if hasattr(self.tokenizer, 'apply_chat_template'):
                        formatted_text = self.tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=False
                        )
                        if formatted_text is not None:
                            inputs = self.tokenizer(formatted_text, return_tensors="pt").to("cuda" if CUDA_AVAILABLE else "cpu")
                        else:
                            inputs = self.tokenizer(test_prompt, return_tensors="pt").to("cuda" if CUDA_AVAILABLE else "cpu")
                    else:
                        inputs = self.tokenizer(test_prompt, return_tensors="pt").to("cuda" if CUDA_AVAILABLE else "cpu")
                except Exception as e:
                    logger.warning(f"Failed to format chat template, using raw prompt: {e}")
                    inputs = self.tokenizer(test_prompt, return_tensors="pt").to("cuda" if CUDA_AVAILABLE else "cpu")
        else:
            # Text model
            try:
                messages = [{"role": "user", "content": test_prompt}]
                input_text = self.tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=True, 
                    tokenize=False
                )
                if input_text is not None:
                    inputs = self.tokenizer(input_text, return_tensors="pt").to("cuda" if CUDA_AVAILABLE else "cpu")
                else:
                    inputs = self.tokenizer(test_prompt, return_tensors="pt").to("cuda" if CUDA_AVAILABLE else "cpu")
            except Exception as e:
                logger.warning(f"Failed to apply chat template for text model, using raw prompt: {e}")
                inputs = self.tokenizer(test_prompt, return_tensors="pt").to("cuda" if CUDA_AVAILABLE else "cpu")
        
        # Generate response
        with torch.no_grad():
            # Get proper pad token id
            try:
                if hasattr(self, 'processor') and self.processor and hasattr(self.processor, 'tokenizer'):
                    pad_token_id = self.processor.tokenizer.eos_token_id
                elif hasattr(self.tokenizer, 'eos_token_id'):
                    pad_token_id = self.tokenizer.eos_token_id
                else:
                    pad_token_id = None
            except:
                pad_token_id = None
            
            generation_kwargs = {
                "max_new_tokens": 256,
                "temperature": 0.1,  # Lower temperature for more consistent testing
                "top_p": 0.95,
                "do_sample": True,
            }
            
            if pad_token_id is not None:
                generation_kwargs["pad_token_id"] = pad_token_id
            
            outputs = self.model.generate(**inputs, **generation_kwargs)
        
        # Decode response and extract only the new tokens
        try:
            if hasattr(self, 'processor') and self.processor and self.config.model_type == "multimodal":
                # For multimodal models, try processor first, then fallback to tokenizer
                try:
                    full_response = self.processor.decode(outputs[0], skip_special_tokens=True)
                except Exception as e:
                    logger.warning(f"Processor decode failed, using tokenizer: {e}")
                    # Extract actual tokenizer if processor has one
                    actual_tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor
                    full_response = actual_tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove input prompt)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            
            if hasattr(self, 'processor') and self.processor and self.config.model_type == "multimodal":
                try:
                    response = self.processor.decode(generated_tokens, skip_special_tokens=True)
                except:
                    actual_tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor
                    response = actual_tokenizer.decode(generated_tokens, skip_special_tokens=True)
            else:
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
        except Exception as e:
            logger.error(f"Failed to decode response: {e}")
            response = f"<Decoding failed: {str(e)}>"
        
        logger.info(f"Test prompt: {test_prompt}")
        logger.info(f"Model response: {response}")
        
        return response


def main():
    parser = argparse.ArgumentParser(description="Fine-tune models on Augmentoolkit synthetic data using Unsloth")
    parser.add_argument("--config", type=str, help="Path to configuration YAML file")
    parser.add_argument("--model-name", type=str, help="Model name or path")
    parser.add_argument("--model-type", type=str, choices=["text", "vision", "multimodal"], help="Model type")
    parser.add_argument("--data-path", type=str, help="Path to Augmentoolkit output data")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA fine-tuning")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit")
    parser.add_argument("--max-steps", type=int, help="Maximum training steps")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--batch-size", type=int, help="Batch size per device")
    parser.add_argument("--gradient-accumulation-steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--test-only", action="store_true", help="Only test the model without training")
    parser.add_argument("--gpu-device", type=int, default=0, help="Which GPU to use")
    parser.add_argument("--num-train-epochs", type=int, help="Number of training epochs")
    
    # Dataset caching arguments
    parser.add_argument("--disable-caching", action="store_true", help="Disable dataset caching")
    parser.add_argument("--cache-dir", type=str, help="Directory for cached datasets")
    parser.add_argument("--force-reprocess", action="store_true", help="Force reprocessing of datasets (ignore cache)")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = AugmentoolkitConfig(**config_dict)
    else:
        config = AugmentoolkitConfig()
    
    # Override config with command line arguments (only if explicitly provided)
    if args.model_name is not None:
        config.model_name = args.model_name
    if args.model_type is not None:
        config.model_type = args.model_type
    if args.data_path is not None:
        config.data_base_path = args.data_path
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.use_lora:
        config.use_lora = True
    if args.load_in_4bit:
        config.load_in_4bit = True
        config.load_in_8bit = False  # Disable 8-bit when 4-bit is enabled
    if args.load_in_8bit:
        config.load_in_4bit = False
        config.load_in_8bit = True  # Disable 4-bit when 8-bit is enabled
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.num_train_epochs is not None:
        config.num_train_epochs = args.num_train_epochs
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.per_device_train_batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    
    # GPU device override
    if args.gpu_device is not None:
        config.gpu_device = args.gpu_device
    
    # Caching overrides
    if args.disable_caching:
        config.enable_dataset_caching = False
        logger.info("📁 Dataset caching disabled via command line")
    if args.cache_dir is not None:
        config.cache_dir = args.cache_dir
        logger.info(f"📁 Cache directory set to: {args.cache_dir}")
    if args.force_reprocess:
        config.force_reprocess = True
        logger.info("🔄 Force reprocessing enabled - will ignore cached datasets")
    
    # Ensure Unsloth is enabled (this is an Unsloth-specific script)
    config.use_unsloth = True
    
    # Create trainer
    trainer = AugmentoolkitTrainer(config)
    
    if args.test_only:
        # Load existing model and test
        trainer.load_model_and_tokenizer()
        trainer.test_model()
    else:
        # Run full training
        trainer.run_training()
        trainer.test_model()


if __name__ == "__main__":
    main() 