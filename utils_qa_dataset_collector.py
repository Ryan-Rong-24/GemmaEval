#!/usr/bin/env python3
"""
Script to collect all Q&A pairs from the survival dataset and upload to Hugging Face.
Excludes Augmentoolkit's generic SFT and transformed generic data.
"""

import json
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Set
from datasets import Dataset
from huggingface_hub import HfApi
import argparse


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return data


def load_json(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return [data]
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def extract_qa_from_rag_data(item: Dict[str, Any]) -> Dict[str, Any]:
    """Extract Q&A from RAG data format."""
    # RAG data has 'text' field with content and metadata
    text = item.get('text', '')
    metadata = item.get('metadata', '')
    
    return {
        'instruction': text,  # Using text as instruction for now
        'response': '',  # RAG data doesn't have explicit Q&A structure
        'source': 'rag_data',
        'metadata': metadata,
        'rag_failed': item.get('rag_failed', False),
        'judgement': item.get('judgement', True)
    }


def extract_qa_from_plain_qa(item: Dict[str, Any]) -> Dict[str, Any]:
    """Extract Q&A from plain Q&A format."""
    # Plain Q&A typically has question and answer fields
    question = item.get('question', item.get('instruction', ''))
    answer = item.get('answer', item.get('response', ''))
    
    return {
        'instruction': question,
        'response': answer,
        'source': 'plain_qa',
        'metadata': item.get('metadata', ''),
        'category': item.get('category', 'general')
    }


def extract_qa_from_conversations(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract Q&A from conversation format."""
    qa_pairs = []
    
    if 'conversations' in item:
        conversations = item['conversations']
        current_instruction = ""
        
        for conv in conversations:
            role = conv.get('from', '')
            content = conv.get('value', '')
            
            if role in ['human', 'user']:
                current_instruction = content
            elif role in ['gpt', 'assistant'] and current_instruction:
                qa_pairs.append({
                    'instruction': current_instruction,
                    'response': content,
                    'source': 'conversation',
                    'metadata': item.get('metadata', ''),
                    'category': item.get('category', 'conversation')
                })
                current_instruction = ""
    
    elif 'segments' in item:
        # Handle segments format
        for segment in item.get('segments', []):
            if segment.get('label') and 'text' in segment:
                # This appears to be instruction-response pairs
                text = segment['text']
                if 'Human:' in text and 'AI:' in text:
                    parts = text.split('AI:', 1)
                    if len(parts) == 2:
                        instruction = parts[0].replace('Human:', '').strip()
                        response = parts[1].strip()
                        qa_pairs.append({
                            'instruction': instruction,
                            'response': response,
                            'source': 'segments',
                            'metadata': item.get('metadata', ''),
                            'category': 'segments'
                        })
    
    return qa_pairs if qa_pairs else [extract_qa_from_plain_qa(item)]


def calculate_hash(text: str) -> str:
    """Calculate hash for deduplication."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def collect_qa_pairs(base_path: str) -> List[Dict[str, Any]]:
    """Collect all Q&A pairs from the survival dataset."""
    base_path = Path(base_path)
    all_qa_pairs = []
    seen_hashes: Set[str] = set()
    
    # Define file paths to collect (excluding generic SFT and transformed generic)
    file_paths = [
        # Main RAG files (use only one to avoid duplicates)
        "rag_prepared_data.jsonl",
        
        # Plain Q&A files from factual SFT
        "factual_sft_facts_vague_1/plain_qa_list.jsonl",
        "factual_sft_facts_vague_0/plain_qa_list.jsonl",
        "factual_sft_facts_negative_1/plain_qa_list.jsonl",
        "factual_sft_facts_negative_0/plain_qa_list.jsonl",
        "factual_sft_facts_openended_0/plain_qa_list.jsonl",
        "factual_sft_facts_openended_1/plain_qa_list.jsonl",
        "factual_sft_facts_followup_0/plain_qa_list.jsonl",
        "factual_sft_facts_followup_1/plain_qa_list.jsonl",
        "factual_sft_facts_hallucination_1/plain_qa_list.jsonl",
        "factual_sft_facts_hallucination_0/plain_qa_list.jsonl",
        
        # Correction conversations
        "corrections_facts/axolotl_correction_conversations.json",
    ]
    
    for file_path in file_paths:
        full_path = base_path / file_path
        if not full_path.exists():
            print(f"File not found: {full_path}")
            continue
            
        print(f"Processing: {full_path}")
        
        # Load data based on file extension
        if file_path.endswith('.jsonl'):
            data = load_jsonl(str(full_path))
        else:
            data = load_json(str(full_path))
        
        # Extract Q&A pairs based on file type
        for item in data:
            if 'conversations' in item or 'segments' in item:
                qa_pairs = extract_qa_from_conversations(item)
            elif 'question' in item or 'instruction' in item:
                qa_pairs = [extract_qa_from_plain_qa(item)]
            else:
                qa_pairs = [extract_qa_from_rag_data(item)]
            
            # Add to collection with deduplication
            for qa_pair in qa_pairs:
                # Skip empty instructions or responses
                instruction = qa_pair.get('instruction', '').strip()
                response = qa_pair.get('response', '').strip()
                
                if not instruction:
                    continue
                
                # Create hash for deduplication
                content_hash = calculate_hash(instruction + response)
                
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    qa_pair['file_source'] = file_path
                    all_qa_pairs.append(qa_pair)
    
    print(f"Collected {len(all_qa_pairs)} unique Q&A pairs")
    return all_qa_pairs


def create_huggingface_dataset(qa_pairs: List[Dict[str, Any]]) -> Dataset:
    """Create a Hugging Face dataset from Q&A pairs."""
    
    # Prepare data in the format expected by Hugging Face
    dataset_data = {
        'instruction': [],
        'response': [],
        'source': [],
        'metadata': [],
        'category': []
    }
    
    for qa_pair in qa_pairs:
        dataset_data['instruction'].append(qa_pair.get('instruction', ''))
        dataset_data['response'].append(qa_pair.get('response', ''))
        dataset_data['source'].append(qa_pair.get('source', ''))
        dataset_data['metadata'].append(qa_pair.get('metadata', ''))
        dataset_data['category'].append(qa_pair.get('category', 'general'))
    
    dataset = Dataset.from_dict(dataset_data)
    return dataset


def upload_to_huggingface(dataset: Dataset, repo_name: str, token: str, private: bool = False):
    """Upload dataset to Hugging Face Hub."""
    try:
        # Push to hub
        dataset.push_to_hub(
            repo_name,
            token=token,
            private=private
        )
        print(f"Successfully uploaded dataset to: https://huggingface.co/datasets/{repo_name}")
        
    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")
        print("Please check your token and repo name.")


def save_local_dataset(dataset: Dataset, output_path: str):
    """Save dataset locally as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to pandas and save as JSON
    df = dataset.to_pandas()
    df.to_json(output_path, orient='records', indent=2)
    print(f"Dataset saved locally to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Collect and upload Q&A dataset to Hugging Face')
    parser.add_argument('--base_path', required=True, help='Path to survival_dataset directory')
    parser.add_argument('--repo_name', required=True, help='Hugging Face repo name (e.g., username/dataset-name)')
    parser.add_argument('--token', help='Hugging Face token')
    parser.add_argument('--private', action='store_true', help='Make dataset private')
    parser.add_argument('--local_only', action='store_true', help='Only save locally, do not upload')
    parser.add_argument('--output_path', default='survival_qa_dataset.json', help='Local output path')
    
    args = parser.parse_args()
    
    # Collect Q&A pairs
    print("Collecting Q&A pairs...")
    qa_pairs = collect_qa_pairs(args.base_path)
    
    if not qa_pairs:
        print("No Q&A pairs found!")
        return
    
    # Create dataset
    print("Creating Hugging Face dataset...")
    dataset = create_huggingface_dataset(qa_pairs)
    
    # Print dataset info
    print(f"Dataset created with {len(dataset)} examples")
    print("Dataset features:", dataset.features)
    
    # Save locally
    save_local_dataset(dataset, args.output_path)
    
    # Upload to Hugging Face if not local only
    if not args.local_only:
        if not args.token:
            print("Error: Hugging Face token required for upload. Use --token or set --local_only")
            return
        
        print(f"Uploading to Hugging Face: {args.repo_name}")
        upload_to_huggingface(dataset, args.repo_name, args.token, args.private)
    
    print("Done!")


if __name__ == "__main__":
    main()