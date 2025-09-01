#!/usr/bin/env python3
"""
Inference script for fine-tuned Gemma 3N model.
Supports both text-only and multimodal (text + image) inputs.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Union, List
import json
from PIL import Image
import requests
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Try to import Unsloth for optimized inference
    from unsloth import FastModel
    UNSLOTH_AVAILABLE = True
    logger.info("✅ Unsloth available for optimized inference")
except ImportError:
    UNSLOTH_AVAILABLE = False
    logger.info("ℹ️ Unsloth not available, using standard transformers")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
    from transformers import BitsAndBytesConfig
except ImportError:
    logger.error("Please install transformers: pip install transformers")
    exit(1)
    
import torch


class Gemma3NInference:
    """Inference class for fine-tuned Gemma 3N model."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        do_sample: bool = True,
        use_unsloth: bool = False
    ):
        self.model_path = model_path
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.use_unsloth = use_unsloth
        
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        # Load the model
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer."""
        logger.info(f"Loading model from: {self.model_path}")
        
        if self.use_unsloth and UNSLOTH_AVAILABLE:
            self._load_with_unsloth()
        else:
            self._load_with_transformers()
        
        logger.info("✅ Model loaded successfully")
    
    def _load_with_unsloth(self):
        """Load model using Unsloth for optimized inference."""
        logger.info("Loading with Unsloth optimization...")
        
        try:
            # Load with Unsloth
            self.model, self.tokenizer = FastModel.from_pretrained(
                model_name=self.model_path,
                dtype=None,  # Auto-detect
                load_in_4bit=self.load_in_4bit,
                device_map={"": 0} if self.device == "cuda" else None,
            )
            
            # Set to inference mode
            FastModel.for_inference(self.model)
                
        except Exception as e:
            logger.warning(f"Unsloth loading failed: {e}")
            logger.info("Falling back to standard transformers...")
            self._load_with_transformers()
    
    def _load_with_transformers(self):
        """Load model using standard transformers."""
        logger.info("Loading with standard transformers...")
        
        # Configure quantization if requested
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load model with proper device handling
        device_map = None
        if self.device == "auto":
            if not self.load_in_4bit and not self.load_in_8bit:
                # For non-quantized models, use single GPU if available
                device_map = {"": 0} if torch.cuda.is_available() else "cpu"
            else:
                device_map = "auto"
        else:
            device_map = self.device
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        
        # Try to load processor for multimodal
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            logger.info("✅ Loaded multimodal processor")
        except Exception as e:
            logger.info(f"No processor found (text-only model): {e}")
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def format_chat_message(self, user_message: str, system_message: Optional[str] = None) -> str:
        """Format message using Gemma chat template."""
        if system_message:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        else:
            messages = [{"role": "user", "content": user_message}]
        
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback for Gemma format
            formatted = "<bos>"
            if system_message:
                formatted += f"<start_of_turn>system\n{system_message}<end_of_turn>\n"
            formatted += f"<start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n"
            return formatted
    
    def load_image(self, image_input: Union[str, Image.Image]) -> Image.Image:
        """Load image from path, URL, or PIL Image."""
        if isinstance(image_input, Image.Image):
            return image_input
        elif isinstance(image_input, str):
            if image_input.startswith(('http://', 'https://')):
                # Download from URL
                response = requests.get(image_input)
                return Image.open(BytesIO(response.content)).convert('RGB')
            else:
                # Load from file path
                return Image.open(image_input).convert('RGB')
        else:
            raise ValueError(f"Invalid image input type: {type(image_input)}")
    
    def generate_text(
        self, 
        prompt: str, 
        image: Optional[Union[str, Image.Image]] = None,
        system_message: Optional[str] = None,
        **generation_kwargs
    ) -> str:
        """Generate text response from prompt and optional image."""
        
        # Format the prompt
        formatted_prompt = self.format_chat_message(prompt, system_message)
        
        if image is not None:
            if self.use_unsloth and UNSLOTH_AVAILABLE:
                # Unsloth multimodal generation (uses tokenizer directly)
                return self._generate_multimodal_unsloth(prompt, image, system_message, **generation_kwargs)
            elif self.processor is not None:
                # Standard transformers multimodal generation
                return self._generate_multimodal(formatted_prompt, image, **generation_kwargs)
            else:
                logger.warning("Image provided but no multimodal support available")
                return self._generate_text_only(formatted_prompt, **generation_kwargs)
        else:
            # Text-only generation
            return self._generate_text_only(formatted_prompt, **generation_kwargs)
    
    def _generate_text_only(self, prompt: str, **generation_kwargs) -> str:
        """Generate text-only response."""
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=4096
        )
        
        # Move inputs to the same device as the model
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        # Set generation parameters
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": 0.95,
            "top_k": 64,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        gen_kwargs.update(generation_kwargs)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):], 
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def _generate_multimodal_unsloth(
        self, 
        prompt: str, 
        image: Union[str, Image.Image], 
        system_message: Optional[str] = None,
        **generation_kwargs
    ) -> str:
        """Generate multimodal response using Unsloth approach."""
        
        # Load and process image
        pil_image = self.load_image(image)
        
        # Prepare messages in the format expected by Unsloth
        # Use the original image path if it's a string, otherwise we'll need to save the PIL image
        image_input = image
        if not isinstance(image, str):
            # For PIL images, we need to use the image object directly
            image_input = pil_image
            
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_input},
                {"type": "text", "text": prompt}
            ]
        }]
        
        # Add system message if provided
        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})
        
        # Set generation parameters
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": 0.95,
            "top_k": 64,
            "do_sample": self.do_sample,
        }
        gen_kwargs.update(generation_kwargs)
        
        # Generate using Unsloth approach
        with torch.no_grad():
            outputs = self.model.generate(
                **self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(next(self.model.parameters()).device),
                **gen_kwargs
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the model response from the full conversation
        # Look for the last part after the generation prompt
        if "<start_of_turn>model\n" in response:
            response = response.split("<start_of_turn>model\n")[-1].strip()
        
        return response.strip()
    
    def _generate_multimodal(
        self, 
        prompt: str, 
        image: Union[str, Image.Image], 
        **generation_kwargs
    ) -> str:
        """Generate multimodal response."""
        
        # Load and process image
        pil_image = self.load_image(image)
        
        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt",
            padding=True
        )
        
        # Move inputs to the same device as the model
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Set generation parameters
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": 0.95,
            "top_k": 64,
            "do_sample": self.do_sample,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        gen_kwargs.update(generation_kwargs)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )
        
        # Decode response
        response = self.processor.decode(
            outputs[0][len(inputs["input_ids"][0]):], 
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def chat(self):
        """Interactive chat mode."""
        logger.info("🤖 Starting interactive chat mode. Type 'quit' to exit.")
        logger.info("💡 For multimodal: use 'image:/path/to/image.jpg your question'")
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    logger.info("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Check for image input
                image = None
                if user_input.startswith('image:'):
                    parts = user_input.split(' ', 1)
                    if len(parts) == 2:
                        image_path = parts[0][6:]  # Remove 'image:' prefix
                        user_input = parts[1]
                        try:
                            image = self.load_image(image_path)
                            logger.info(f"📷 Loaded image: {image_path}")
                        except Exception as e:
                            logger.error(f"Failed to load image: {e}")
                            continue
                    else:
                        logger.error("Invalid image format. Use: image:/path/to/image.jpg your question")
                        continue
                
                # Generate response
                logger.info("🤔 Thinking...")
                response = self.generate_text(user_input, image=image)
                
                print(f"\n🤖 Assistant: {response}")
                
            except KeyboardInterrupt:
                logger.info("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error during generation: {e}")


def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned Gemma 3N model")
    parser.add_argument("--model-path", type=str, required=True, 
                       help="Path to the fine-tuned model directory")
    parser.add_argument("--prompt", type=str, help="Single prompt for inference")
    parser.add_argument("--image", type=str, help="Path to image file (for multimodal)")
    parser.add_argument("--system", type=str, help="System message")
    parser.add_argument("--interactive", action="store_true", help="Start interactive chat mode")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit")
    parser.add_argument("--use-unsloth", action="store_true", help="Use Unsloth for optimized inference")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--no-sample", action="store_true", help="Use greedy decoding")
    
    args = parser.parse_args()
    
    # Validate model path
    if not Path(args.model_path).exists():
        logger.error(f"Model path does not exist: {args.model_path}")
        return
    
    # Initialize inference engine
    try:
        inference = Gemma3NInference(
            model_path=args.model_path,
            device=args.device,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=not args.no_sample,
            use_unsloth=args.use_unsloth,
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    if args.interactive:
        # Interactive mode
        inference.chat()
    elif args.prompt:
        # Single prompt mode
        try:
            image = None
            if args.image:
                image = inference.load_image(args.image)
                logger.info(f"📷 Loaded image: {args.image}")
            
            response = inference.generate_text(
                args.prompt, 
                image=image, 
                system_message=args.system
            )
            
            print(f"\n🤖 Response: {response}")
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
    else:
        logger.error("Please provide either --prompt or --interactive")


if __name__ == "__main__":
    main() 