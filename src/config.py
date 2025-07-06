"""
Configuration module for the fire/smoke detection evaluation system.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the evaluation system."""
    
    # API Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Dataset Configuration
    DATASET_PATH: str = os.getenv("DATASET_PATH", "data/D-Fire")
    BASE_RESULTS_PATH: str = os.getenv("BASE_RESULTS_PATH", "results")
    BASE_LOGS_PATH: str = os.getenv("BASE_LOGS_PATH", "logs")
    
    # Evaluation Configuration
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "10"))
    MAX_IMAGES_PER_CATEGORY: int = int(os.getenv("MAX_IMAGES_PER_CATEGORY", "100"))
    TIMEOUT_SECONDS: int = int(os.getenv("TIMEOUT_SECONDS", "30"))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Session-specific paths (set dynamically)
    RESULTS_PATH: str = ""
    EVALS_PATH: str = ""
    PLOTS_PATH: str = ""
    LOG_FILE: str = ""
    
    # Gemma Models to Evaluate
    GEMMA_MODELS: List[str] = [
        # "gemma-3-27b-it",
        # "gemma-3-12b-it",
        # "gemma-3-4b-it",
        "gemma-3n-e2b-it",
        "gemma-3n-e4b-it"
    ]
    
    # Fire/Smoke Detection Prompts
    DETECTION_PROMPTS: Dict[str, str] = {
        "simple": "Is there fire or smoke visible in this image? Answer with: 'fire', 'smoke', 'both', or 'none'.",
        # "detailed": "Analyze this image for fire and smoke detection. Respond with JSON format: {'fire': true/false, 'smoke': true/false, 'confidence': 0-1}"
    }
    
    # Dataset Categories (based on D-Fire structure)
    CATEGORIES: Dict[str, str] = {
        "fire": "only_fire",
        "smoke": "only_smoke", 
        "both": "fire_and_smoke",
        "none": "no_fire_smoke"
    }
    
    @classmethod
    def generate_session_config_string(cls, models: List[str], prompt_types: List[str], 
                                     max_images: int, dataset_split: str) -> str:
        """Generate a configuration string for the session."""
        model_str = "_".join([m.replace("gemma-", "").replace("-it", "") for m in models])
        prompt_str = "_".join(prompt_types)
        return f"{model_str}_{prompt_str}_{max_images}img_{dataset_split}"
    
    @classmethod
    def setup_session_paths(cls, models: List[str], prompt_types: List[str], 
                          max_images: int, dataset_split: str,
                          timestamp: Optional[str] = None) -> None:
        """Setup session-specific paths with timestamp and configuration."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        config_string = cls.generate_session_config_string(models, prompt_types, max_images, dataset_split)
        
        # Create folder names with timestamp and config
        session_name = f"{timestamp}_{config_string}"
        
        # Setup results paths
        cls.RESULTS_PATH = str(Path(cls.BASE_RESULTS_PATH) / session_name)
        cls.EVALS_PATH = str(Path(cls.RESULTS_PATH) / "evals")
        cls.PLOTS_PATH = str(Path(cls.RESULTS_PATH) / "plots")
        
        # Setup logs path
        logs_session_name = f"logs_{session_name}"
        cls.LOG_FILE = str(Path(cls.BASE_LOGS_PATH) / logs_session_name / "evaluation.log")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration values."""
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required")
        
        dataset_path = Path(cls.DATASET_PATH)
        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {cls.DATASET_PATH}")
            
        return True
    
    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories."""
        if cls.RESULTS_PATH:
            Path(cls.RESULTS_PATH).mkdir(parents=True, exist_ok=True)
            Path(cls.EVALS_PATH).mkdir(parents=True, exist_ok=True)
            Path(cls.PLOTS_PATH).mkdir(parents=True, exist_ok=True)
        
        if cls.LOG_FILE:
            Path(cls.LOG_FILE).parent.mkdir(parents=True, exist_ok=True) 