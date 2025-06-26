"""
Dataset module for handling D-Fire fire/smoke detection dataset.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from PIL import Image
import numpy as np

class DFireDataset:
    """D-Fire dataset handler for fire and smoke detection evaluation."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize the D-Fire dataset handler.
        
        Args:
            dataset_path: Path to the D-Fire dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.train_path = self.dataset_path / "train"
        self.test_path = self.dataset_path / "test"
        
        # Validate dataset structure
        self._validate_structure()
        
        # Cache for loaded data
        self._train_data: Optional[Dict] = None
        self._test_data: Optional[Dict] = None
    
    def _validate_structure(self) -> None:
        """Validate that the dataset has the expected structure."""
        required_paths = [
            self.train_path / "images",
            self.train_path / "labels", 
            self.test_path / "images",
            self.test_path / "labels"
        ]
        
        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"Required dataset path not found: {path}")
    
    def _load_yolo_labels(self, labels_dir: Path) -> Dict[str, List[Dict]]:
        """
        Load YOLO format labels from directory.
        
        Args:
            labels_dir: Directory containing .txt label files
            
        Returns:
            Dictionary mapping image names to list of bounding box annotations
        """
        labels = {}
        
        for label_file in labels_dir.glob("*.txt"):
            image_name = label_file.stem
            annotations = []
            
            if label_file.stat().st_size > 0:  # Non-empty file
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            annotations.append({
                                'class_id': class_id,
                                'x_center': x_center,
                                'y_center': y_center,
                                'width': width,
                                'height': height,
                                'class_name': 'smoke' if class_id == 0 else 'fire'
                            })
            
            labels[image_name] = annotations
        
        return labels
    
    def _categorize_images(self, labels: Dict[str, List[Dict]]) -> Dict[str, List[str]]:
        """
        Categorize images based on their annotations.
        
        Args:
            labels: Dictionary of image labels
            
        Returns:
            Dictionary mapping categories to lists of image names
        """
        categories = {
            'fire': [],
            'smoke': [],
            'both': [],
            'none': []
        }
        
        for image_name, annotations in labels.items():
            if not annotations:
                categories['none'].append(image_name)
            else:
                has_fire = any(ann['class_name'] == 'fire' for ann in annotations)
                has_smoke = any(ann['class_name'] == 'smoke' for ann in annotations)
                
                if has_fire and has_smoke:
                    categories['both'].append(image_name)
                elif has_fire:
                    categories['fire'].append(image_name)
                elif has_smoke:
                    categories['smoke'].append(image_name)
                else:
                    categories['none'].append(image_name)
        
        return categories
    
    def load_split(self, split: str = "test") -> Dict:
        """
        Load a dataset split (train or test).
        
        Args:
            split: Dataset split to load ('train' or 'test')
            
        Returns:
            Dictionary containing images, labels, and categories
        """
        if split == "train" and self._train_data is not None:
            return self._train_data
        elif split == "test" and self._test_data is not None:
            return self._test_data
        
        split_path = self.train_path if split == "train" else self.test_path
        images_dir = split_path / "images"
        labels_dir = split_path / "labels"
        
        # Load labels
        labels = self._load_yolo_labels(labels_dir)
        
        # Categorize images
        categories = self._categorize_images(labels)
        
        # Get list of available images
        available_images = set(img.stem for img in images_dir.glob("*.[jp][pn]g"))
        
        # Filter categories to only include available images
        for category in categories:
            categories[category] = [
                img for img in categories[category] 
                if img in available_images
            ]
        
        data = {
            'images_dir': images_dir,
            'labels_dir': labels_dir,
            'labels': labels,
            'categories': categories,
            'total_images': len(available_images)
        }
        
        # Cache the data
        if split == "train":
            self._train_data = data
        else:
            self._test_data = data
        
        return data
    
    def get_sample_images(self, split: str = "test", 
                         max_per_category: Optional[int] = 100,
                         seed: int = 42) -> List[Tuple[str, str, str]]:
        """
        Get a sample of images for evaluation.
        
        Args:
            split: Dataset split to sample from
            max_per_category: Maximum images per category (None for no limit)
            seed: Random seed for reproducibility
            
        Returns:
            List of tuples (image_path, category, image_name)
        """
        random.seed(seed)
        
        data = self.load_split(split)
        categories = data['categories']
        images_dir = data['images_dir']
        
        sample_images = []
        
        for category, image_names in categories.items():
            if not image_names:
                continue
                
            # Sample up to max_per_category images, or all if max_per_category is None
            if max_per_category is None:
                sampled_names = image_names[:]  # Copy all images
            else:
                sampled_names = random.sample(
                    image_names, 
                    min(len(image_names), max_per_category)
                )
            
            for image_name in sampled_names:
                # Find the actual image file (could be .jpg or .png)
                image_path = None
                for ext in ['.jpg', '.png', '.jpeg']:
                    candidate_path = images_dir / f"{image_name}{ext}"
                    if candidate_path.exists():
                        image_path = str(candidate_path)
                        break
                
                if image_path:
                    sample_images.append((image_path, category, image_name))
        
        return sample_images

    def get_all_images(self, splits: List[str] = ["train", "test"], 
                      max_per_category: Optional[int] = None,
                      seed: int = 42) -> List[Tuple[str, str, str]]:
        """
        Get all images from multiple dataset splits for evaluation.
        
        Args:
            splits: List of dataset splits to include (e.g., ["train", "test"])
            max_per_category: Maximum images per category across all splits (None for no limit)
            seed: Random seed for reproducibility
            
        Returns:
            List of tuples (image_path, category, image_name)
        """
        random.seed(seed)
        
        all_images = []
        category_counts = {}
        
        # Collect images from all splits first
        for split in splits:
            split_images = self.get_sample_images(split=split, max_per_category=None, seed=seed)
            for image_path, category, image_name in split_images:
                # Add split prefix to image name to avoid conflicts
                prefixed_name = f"{split}_{image_name}"
                all_images.append((image_path, category, prefixed_name))
        
        # If no limit, return all images
        if max_per_category is None:
            return all_images
        
        # Otherwise, sample up to max_per_category from each category
        categorized_images = {}
        for image_path, category, image_name in all_images:
            if category not in categorized_images:
                categorized_images[category] = []
            categorized_images[category].append((image_path, category, image_name))
        
        # Sample from each category
        sampled_images = []
        for category, images in categorized_images.items():
            if max_per_category is None:
                sampled_images.extend(images)
            else:
                sampled = random.sample(images, min(len(images), max_per_category))
                sampled_images.extend(sampled)
        
        return sampled_images
    
    def get_dataset_stats(self, split: str = "test") -> Dict:
        """
        Get statistics about the dataset split.
        
        Args:
            split: Dataset split to analyze
            
        Returns:
            Dictionary containing dataset statistics
        """
        data = self.load_split(split)
        categories = data['categories']
        
        stats = {
            'total_images': data['total_images'],
            'categories': {
                category: len(images) 
                for category, images in categories.items()
            }
        }
        
        # Add annotation statistics
        labels = data['labels']
        total_annotations = sum(len(anns) for anns in labels.values())
        fire_annotations = sum(
            sum(1 for ann in anns if ann['class_name'] == 'fire')
            for anns in labels.values()
        )
        smoke_annotations = sum(
            sum(1 for ann in anns if ann['class_name'] == 'smoke')
            for anns in labels.values()
        )
        
        stats['annotations'] = {
            'total': total_annotations,
            'fire': fire_annotations,
            'smoke': smoke_annotations
        }
        
        return stats 