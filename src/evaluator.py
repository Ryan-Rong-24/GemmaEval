"""
Model evaluation module for fire/smoke detection using Gemma models via Gemini API.
"""

import time
import json
import asyncio
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp
import random

from google import genai
from PIL import Image
import numpy as np

from .config import Config

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_requests_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def wait_if_needed(self):
        """Wait if we would exceed the rate limit."""
        async with self.lock:
            now = time.time()
            
            # Remove requests older than 1 minute
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            # If we're at the limit, wait until the oldest request is 1 minute old
            if len(self.requests) >= self.max_requests:
                oldest_request = min(self.requests)
                wait_time = 60 - (now - oldest_request)
                if wait_time > 0:
                    print(f"⏱️  Rate limit reached, waiting {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                    # Clean up old requests again after waiting
                    now = time.time()
                    self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            # Record this request
            self.requests.append(now)

class GemmaEvaluator:
    """Evaluator for Gemma models on fire/smoke detection tasks."""
    
    def __init__(self, api_key: str):
        """
        Initialize the Gemma evaluator.
        
        Args:
            api_key: Gemini API key
        """
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.logger = self._setup_logger()
        self.rate_limiter = None  # Will be set based on model
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the evaluation process."""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, Config.LOG_LEVEL))
        
        # Create log file handler
        Config.create_directories()
        handler = logging.FileHandler(Config.LOG_FILE)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Also log to console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _create_client(self) -> Any:
        """Create a new client instance for thread-safe operations."""
        return genai.Client(api_key=self.api_key)
    
    def _upload_image(self, image_path: str, client: Optional[Any] = None) -> Optional[Any]:
        """
        Upload an image to Gemini.
        
        Args:
            image_path: Path to the image file
            client: Optional specific client instance to use
            
        Returns:
            Uploaded file object or None if failed
        """
        if client is None:
            client = self.client
            
        try:
            uploaded_file = client.files.upload(file=image_path)
            return uploaded_file
        except Exception as e:
            self.logger.error(f"Failed to upload image {image_path}: {e}")
            return None
    
    def _evaluate_single_image(self, 
                             model_name: str,
                             image_path: str, 
                             prompt: str,
                             timeout: int = 30,
                             client: Optional[Any] = None) -> Dict[str, Any]:
        """
        Evaluate a single image with a specific model.
        
        Args:
            model_name: Name of the Gemma model
            image_path: Path to the image
            prompt: Evaluation prompt
            timeout: Timeout in seconds
            client: Optional specific client instance to use
            
        Returns:
            Dictionary containing evaluation results
        """
        start_time = time.time()
        result = {
            'model': model_name,
            'image_path': image_path,
            'prompt': prompt,
            'response': None,
            'error': None,
            'processing_time': 0,
            'timestamp': start_time
        }
        
        # Create thread-specific client if not provided
        if client is None:
            client = self._create_client()
        
        try:
            # Upload image
            uploaded_file = self._upload_image(image_path, client)
            if uploaded_file is None:
                result['error'] = "Failed to upload image"
                return result
            
            self.logger.info(
                f"Starting evaluation of {Path(image_path).name} "
                f"with {model_name}..."
            )
            
            # Generate content
            response = client.models.generate_content(
                model=model_name,
                contents=[uploaded_file, prompt],
            )
            
            result['response'] = response.text
            result['processing_time'] = time.time() - start_time
            
            self.logger.info(
                f"✅ Completed {Path(image_path).name} "
                f"with {model_name} in {result['processing_time']:.2f}s"
            )
            
        except Exception as e:
            result['error'] = str(e)
            result['processing_time'] = time.time() - start_time
            self.logger.error(
                f"❌ Failed {Path(image_path).name} "
                f"with {model_name}: {e}"
            )
        
        return result

    def _evaluate_single_image_threaded(self, 
                                      model_name: str,
                                      image_path: str, 
                                      prompt: str,
                                      timeout: int = 30) -> Dict[str, Any]:
        """
        Thread-safe wrapper for single image evaluation.
        Creates a new client instance for each thread.
        """
        client = self._create_client()
        return self._evaluate_single_image(model_name, image_path, prompt, timeout, client)
    
    def _parse_response(self, response: str, prompt_type: str) -> Dict[str, Any]:
        """
        Parse model response based on prompt type.
        
        Args:
            response: Raw model response
            prompt_type: Type of prompt used
            
        Returns:
            Parsed response dictionary
        """
        response_lower = response.lower().strip()
        
        if prompt_type == "simple":
            # Expected responses: 'fire', 'smoke', 'both', 'none'
            if 'both' in response_lower:
                return {'fire': True, 'smoke': True, 'prediction': 'both'}
            elif 'fire' in response_lower:
                return {'fire': True, 'smoke': False, 'prediction': 'fire'}
            elif 'smoke' in response_lower:
                return {'fire': False, 'smoke': True, 'prediction': 'smoke'}
            else:
                return {'fire': False, 'smoke': False, 'prediction': 'none'}
                
        elif prompt_type == "detailed":
            # Try to parse JSON response
            try:
                parsed = json.loads(response)
                return {
                    'fire': parsed.get('fire', False),
                    'smoke': parsed.get('smoke', False),
                    'confidence': parsed.get('confidence', 0.0),
                    'prediction': 'both' if (parsed.get('fire', False) and parsed.get('smoke', False))
                                else 'fire' if parsed.get('fire', False)
                                else 'smoke' if parsed.get('smoke', False)
                                else 'none'
                }
            except json.JSONDecodeError:
                # Fallback to simple parsing
                return self._parse_response(response, "simple")
        
        # Default fallback
        return {'fire': False, 'smoke': False, 'prediction': 'none'}
    
    def evaluate_images(self,
                       images: List[Tuple[str, str, str]],
                       models: List[str],
                       prompt_type: str = "simple",
                       max_workers: int = 5,
                       use_tqdm: bool = False) -> List[Dict[str, Any]]:
        """
        Evaluate multiple images with multiple models.
        
        Args:
            images: List of (image_path, ground_truth_category, image_name) tuples
            models: List of model names to evaluate
            prompt_type: Type of prompt to use
            max_workers: Maximum number of concurrent evaluations
            use_tqdm: Use tqdm progress bar instead of logging
            
        Returns:
            List of evaluation results
        """
        prompt = Config.DETECTION_PROMPTS[prompt_type]
        results = []
        total_evaluations = len(images) * len(models)
        
        self.logger.info(
            f"Starting evaluation of {len(images)} images "
            f"with {len(models)} models ({total_evaluations} total evaluations)"
        )
        
        # Initialize progress bar if using tqdm
        pbar = None
        if use_tqdm and tqdm is not None:
            pbar = tqdm(total=total_evaluations, desc="Evaluating", unit="eval")
        elif use_tqdm and tqdm is None:
            self.logger.warning("tqdm not available, falling back to logging")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all evaluation tasks
            future_to_info = {}
            
            for image_path, ground_truth, image_name in images:
                for model_name in models:
                    future = executor.submit(
                        self._evaluate_single_image_threaded,
                        model_name,
                        image_path,
                        prompt
                    )
                    future_to_info[future] = {
                        'ground_truth': ground_truth,
                        'image_name': image_name
                    }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_info):
                try:
                    result = future.result()
                    info = future_to_info[future]
                    
                    # Add ground truth and parsed response
                    result['ground_truth'] = info['ground_truth']
                    result['image_name'] = info['image_name']
                    
                    if result['response'] and not result['error']:
                        result['parsed'] = self._parse_response(
                            result['response'], prompt_type
                        )
                    else:
                        result['parsed'] = {'fire': False, 'smoke': False, 'prediction': 'none'}
                    
                    results.append(result)
                    completed += 1
                    
                    # Update progress
                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix({
                            'Model': result['model'].replace('gemma-', ''),
                            'Time': f"{result['processing_time']:.1f}s"
                        })
                    elif completed % 10 == 0:
                        self.logger.info(f"Completed {completed}/{total_evaluations} evaluations")
                        
                except Exception as e:
                    if pbar:
                        pbar.update(1)
                    self.logger.error(f"Error processing evaluation result: {e}")
        
        # Close progress bar
        if pbar:
            pbar.close()
        
        self.logger.info(f"Completed all {total_evaluations} evaluations")
        return results
    
    def evaluate_dataset_sample(self,
                               dataset_images: List[Tuple[str, str, str]],
                               models: Optional[List[str]] = None,
                               prompt_types: Optional[List[str]] = None,
                               max_workers: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Evaluate a dataset sample with multiple models and prompt types.
        
        Args:
            dataset_images: List of dataset images to evaluate
            models: List of models to use (default: all Gemma models)
            prompt_types: List of prompt types to use (default: all types)
            max_workers: Maximum concurrent evaluations
            
        Returns:
            Dictionary mapping prompt types to evaluation results
        """
        if models is None:
            models = Config.GEMMA_MODELS
        
        if prompt_types is None:
            prompt_types = list(Config.DETECTION_PROMPTS.keys())
        
        all_results = {}
        
        for prompt_type in prompt_types:
            self.logger.info(f"Evaluating with prompt type: {prompt_type}")
            results = self.evaluate_images(
                dataset_images,
                models,
                prompt_type,
                max_workers
            )
            all_results[prompt_type] = results
        
        return all_results

    async def _evaluate_single_image_async(self,
                                          session: aiohttp.ClientSession,
                                          model_name: str,
                                          image_path: str,
                                          prompt: str,
                                          timeout: int = 30) -> Dict[str, Any]:
        """
        Async version of single image evaluation.
        
        Args:
            session: aiohttp session for HTTP requests
            model_name: Name of the Gemma model
            image_path: Path to the image
            prompt: Evaluation prompt
            timeout: Timeout in seconds
            
        Returns:
            Dictionary containing evaluation results
        """
        start_time = time.time()
        result = {
            'model': model_name,
            'image_path': image_path,
            'prompt': prompt,
            'response': None,
            'error': None,
            'processing_time': 0,
            'timestamp': start_time
        }
        
        try:
            # Create a new client for this async operation
            # Note: We fall back to sync client as genai doesn't have async support yet
            # This still helps with concurrency control
            loop = asyncio.get_event_loop()
            client = self._create_client()
            
            # Run the sync operation in a thread pool
            result_sync = await loop.run_in_executor(
                None,
                self._evaluate_single_image,
                model_name,
                image_path,
                prompt,
                timeout,
                client
            )
            
            return result_sync
            
        except Exception as e:
            result['error'] = str(e)
            result['processing_time'] = time.time() - start_time
            self.logger.error(
                f"Error in async evaluation {Path(image_path).name} "
                f"with {model_name}: {e}"
            )
            return result

    async def evaluate_images_async(self,
                                  images: List[Tuple[str, str, str]],
                                  models: List[str],
                                  prompt_type: str = "simple",
                                  max_concurrent: int = 10,
                                  use_tqdm: bool = False) -> List[Dict[str, Any]]:
        """
        Async version of evaluate_images with better concurrency and rate limiting.
        
        Args:
            images: List of (image_path, ground_truth_category, image_name) tuples
            models: List of model names to evaluate
            prompt_type: Type of prompt to use
            max_concurrent: Maximum number of concurrent evaluations
            use_tqdm: Use tqdm progress bar instead of logging
            
        Returns:
            List of evaluation results
        """
        prompt = Config.DETECTION_PROMPTS[prompt_type]
        results = []
        total_evaluations = len(images) * len(models)
        
        # Set up rate limiter based on the most restrictive model
        if models:
            min_rpm = min(self._get_model_rpm_limit(model) for model in models)
            self.rate_limiter = RateLimiter(min_rpm)
            self.logger.info(f"Rate limiter set to {min_rpm} RPM for models: {models}")
        
        # Adjust max_concurrent to not exceed rate limits
        # Use 80% of rate limit to account for other API usage
        safe_concurrent = min(max_concurrent, int(min_rpm * 0.8)) if models else max_concurrent
        if safe_concurrent != max_concurrent:
            self.logger.info(f"Reducing max_concurrent from {max_concurrent} to {safe_concurrent} to respect rate limits")
            max_concurrent = safe_concurrent
        
        self.logger.info(
            f"Starting async evaluation of {len(images)} images "
            f"with {len(models)} models ({total_evaluations} total evaluations, max_concurrent={max_concurrent})"
        )
        
        # Initialize progress bar if using tqdm
        pbar = None
        if use_tqdm and tqdm is not None:
            pbar = tqdm(total=total_evaluations, desc="Evaluating (Rate-Limited)", unit="eval")
        
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def evaluate_with_semaphore_and_retry(session, image_path, ground_truth, image_name, model_name):
            async with semaphore:
                result = await self._evaluate_single_image_with_retry(
                    session, model_name, image_path, prompt
                )
                
                # Add ground truth and parsed response
                result['ground_truth'] = ground_truth
                result['image_name'] = image_name
                
                if result['response'] and not result['error']:
                    result['parsed'] = self._parse_response(
                        result['response'], prompt_type
                    )
                else:
                    result['parsed'] = {'fire': False, 'smoke': False, 'prediction': 'none'}
                
                # Update progress
                if pbar:
                    pbar.update(1)
                    status = "✅" if not result.get('error') else "❌"
                    pbar.set_postfix({
                        'Model': result['model'].replace('gemma-', '').replace('gemini-', ''),
                        'Time': f"{result['processing_time']:.1f}s",
                        'Status': status
                    })
                
                return result
        
        # Create all tasks
        async with aiohttp.ClientSession() as session:
            tasks = []
            for image_path, ground_truth, image_name in images:
                for model_name in models:
                    task = evaluate_with_semaphore_and_retry(
                        session, image_path, ground_truth, image_name, model_name
                    )
                    tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and log them
            filtered_results = []
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Task failed with exception: {result}")
                else:
                    filtered_results.append(result)
            
            results = filtered_results
        
        # Close progress bar
        if pbar:
            pbar.close()
        
        # Log final statistics
        successful = len([r for r in results if not r.get('error')])
        error_rate = (len(results) - successful) / len(results) * 100 if results else 0
        
        self.logger.info(f"Completed {len(results)} async evaluations: {successful} successful, {error_rate:.1f}% error rate")
        return results

    def _get_model_rpm_limit(self, model_name: str) -> int:
        """Get the RPM limit for a specific model."""
        # Conservative estimates for free tier
        model_limits = {
            'gemma-3-27b-it': 30,
            'gemma-3-12b-it': 30,
            'gemma-3-4b-it': 30,
        }
        
        # Extract base model name
        for model_key in model_limits:
            if model_key in model_name.lower():
                return model_limits[model_key]
        
        # Default conservative limit
        return 10

    async def _evaluate_single_image_with_retry(self,
                                              session: aiohttp.ClientSession,
                                              model_name: str,
                                              image_path: str,
                                              prompt: str,
                                              timeout: int = 30,
                                              max_retries: int = 3) -> Dict[str, Any]:
        """
        Async version with retry logic for rate limit errors.
        """
        for attempt in range(max_retries + 1):
            try:
                # Wait for rate limiter
                if self.rate_limiter:
                    await self.rate_limiter.wait_if_needed()
                
                result = await self._evaluate_single_image_async(
                    session, model_name, image_path, prompt, timeout
                )
                
                # If successful, return immediately
                if not result.get('error'):
                    return result
                
                # Check if it's a rate limit error
                error_str = str(result.get('error', ''))
                if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str:
                    if attempt < max_retries:
                        # Exponential backoff with jitter
                        wait_time = (2 ** attempt) + random.uniform(0, 5)
                        self.logger.warning(f"Rate limited, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries + 1})")
                        await asyncio.sleep(wait_time)
                        continue
                
                # Non-rate-limit error, return as-is
                return result
                
            except Exception as e:
                if attempt < max_retries:
                    wait_time = (2 ** attempt) + random.uniform(0, 2)
                    self.logger.warning(f"Request failed, retrying in {wait_time:.1f}s: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Max retries exceeded
                    return {
                        'model': model_name,
                        'image_path': image_path,
                        'prompt': prompt,
                        'response': None,
                        'error': f"Max retries exceeded: {e}",
                        'processing_time': 0,
                        'timestamp': time.time()
                    }
        
        # Should never reach here, but just in case
        return result 