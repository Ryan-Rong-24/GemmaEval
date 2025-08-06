#!/usr/bin/env python3
"""
GemmaScout Interactive Chat Interface

Interactive chat interface for testing fine-tuned Gemma models with streaming responses.

Features:
- Real-time streaming text generation
- Multimodal input support (text + images)
- Conversation history management
- Customizable generation parameters
- Error handling and recovery
- Special commands for advanced usage

Usage:
    python inference_interactive_chat.py
    
Commands:
- Normal chat: Just type your message
- Multimodal: image:/path/to/image.jpg What do you see?
- /clear: Clear conversation history
- /help: Show available commands
- /quit: Exit application

Based on Unsloth documentation and optimized for Gemma 3n models.
"""

import torch
from inference_gemma3n import Gemma3NInference
from transformers import TextStreamer
from PIL import Image
import os
import sys
from typing import List, Dict, Optional

# Fix PyTorch Dynamo recompile limits for Unsloth + Gemma 3N
import torch._dynamo
torch._dynamo.config.cache_size_limit = 1000  # Increase from default 64
torch._dynamo.config.suppress_errors = True   # Don't fail on compilation errors

# Set up environment for better PyTorch compilation
import os
os.environ['TORCH_LOGS'] = 'recompiles'  # Monitor recompilations
os.environ['TORCHDYNAMO_VERBOSE'] = '0'   # Reduce verbose output

class StreamingChatBot:
    """Interactive chatbot with streaming responses using TextStreamer."""
    
    def __init__(self, model_path: str, repetition_penalty: float = 1.1):
        self.model_path = model_path
        self.repetition_penalty = repetition_penalty
        self.conversation_history: List[Dict[str, str]] = []
        
        print("🚀 Loading fine-tuned Gemma 3N model...")
        self.inference = Gemma3NInference(
            model_path=model_path,
            load_in_4bit=False,
            temperature=1.0,
            max_new_tokens=512,
            use_unsloth=True,
        )
        
        # Get the actual tokenizer for TextStreamer
        self.tokenizer = self.inference.tokenizer
        if hasattr(self.tokenizer, 'tokenizer'):
            self.tokenizer = self.tokenizer.tokenizer
        
        print("✅ Model loaded successfully!")
        print(f"🔄 Repetition penalty: {self.repetition_penalty}")
        print("💡 Tips:")
        print("  - Type 'image:/path/to/image.jpg What do you see?' for multimodal")
        print("  - Type 'clear' to clear conversation history")
        print("  - Type 'history' to see conversation history")
        print("  - Type 'quit' to exit")
        print("=" * 60)
    
    def add_to_history(self, role: str, content: str):
        """Add message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("🧹 Conversation history cleared!")
    
    def show_history(self):
        """Display conversation history."""
        if not self.conversation_history:
            print("📝 No conversation history yet.")
            return
        
        print("\n📚 Conversation History:")
        print("-" * 40)
        for i, msg in enumerate(self.conversation_history, 1):
            role_emoji = "👤" if msg["role"] == "user" else "🤖"
            print(f"{i}. {role_emoji} {msg['role'].title()}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
        print("-" * 40)
    
    def format_conversation_for_model(self, new_message: str, system_message: Optional[str] = None) -> str:
        """Format the conversation history + new message for the model."""
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add new user message
        messages.append({"role": "user", "content": new_message})
        
        # Format using chat template
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
            
            for msg in messages:
                if msg["role"] == "user":
                    formatted += f"<start_of_turn>user\n{msg['content']}<end_of_turn>\n"
                elif msg["role"] == "assistant":
                    formatted += f"<start_of_turn>model\n{msg['content']}<end_of_turn>\n"
            
            formatted += "<start_of_turn>model\n"
            return formatted
    
    def generate_streaming_response(
        self, 
        prompt: str, 
        image: Optional[Image.Image] = None,
        system_message: Optional[str] = None
    ) -> str:
        """Generate streaming response using TextStreamer."""
        
        # Format the full conversation
        formatted_prompt = self.format_conversation_for_model(prompt, system_message)
        
        if image is not None and self.inference.processor is not None:
            # Multimodal generation with streaming
            return self._generate_multimodal_streaming(formatted_prompt, image)
        else:
            # Text-only generation with streaming
            return self._generate_text_streaming(formatted_prompt)
    
    def _generate_text_streaming(self, prompt: str) -> str:
        """Generate text-only response with streaming."""
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=4096
        )
        
        # Move inputs to the same device as the model
        model_device = next(self.inference.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        # Create TextStreamer for real-time output
        streamer = TextStreamer(
            self.tokenizer, 
            skip_prompt=True,  # Don't repeat the prompt
            skip_special_tokens=True
        )
        
        print("\n🤖 Assistant: ", end="", flush=True)
        
        # Generate with streaming
        with torch.no_grad():
            outputs = self.inference.model.generate(
                **inputs,
                max_new_tokens=self.inference.max_new_tokens,
                temperature=self.inference.temperature,
                do_sample=self.inference.do_sample,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                streamer=streamer,  # This enables streaming output
            )
        
        # Extract just the generated part
        generated_ids = outputs[0][len(inputs["input_ids"][0]):]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print()  # New line after streaming
        return response.strip()
    
    def _generate_multimodal_streaming(self, prompt: str, image: Image.Image) -> str:
        """Generate multimodal response with streaming."""
        
        # Process inputs
        inputs = self.inference.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Move inputs to the same device as the model
        model_device = next(self.inference.model.parameters()).device
        inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Create TextStreamer 
        streamer = TextStreamer(
            self.inference.processor.tokenizer if hasattr(self.inference.processor, 'tokenizer') else self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        print("\n🤖 Assistant: ", end="", flush=True)
        
        # Generate with streaming
        with torch.no_grad():
            outputs = self.inference.model.generate(
                **inputs,
                max_new_tokens=self.inference.max_new_tokens,
                temperature=self.inference.temperature,
                do_sample=self.inference.do_sample,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                streamer=streamer,  # This enables streaming output
            )
        
        # Extract generated part
        generated_ids = outputs[0][len(inputs["input_ids"][0]):]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print()  # New line after streaming
        return response.strip()
    
    def load_image_from_path(self, image_path: str) -> Image.Image:
        """Load image from file path."""
        try:
            return Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to load image from {image_path}: {e}")
    
    def parse_user_input(self, user_input: str) -> tuple[str, Optional[Image.Image]]:
        """Parse user input for image commands."""
        image = None
        
        if user_input.startswith('image:'):
            parts = user_input.split(' ', 1)
            if len(parts) == 2:
                image_path = parts[0][6:]  # Remove 'image:' prefix
                try:
                    image = self.load_image_from_path(image_path)
                    user_input = parts[1]
                    print(f"📷 Loaded image: {image_path}")
                except Exception as e:
                    print(f"❌ Error loading image: {e}")
                    return "", None
            else:
                print("❌ Invalid image format. Use: image:/path/to/image.jpg your question")
                return "", None
        
        return user_input, image
    
    def chat(self):
        """Start the interactive multi-turn conversation."""
        print("\n🤖 Hello! I'm your fine-tuned Gemma 3N assistant. How can I help you today?")
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye! Thanks for chatting!")
                    break
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    continue
                elif user_input.lower() == 'history':
                    self.show_history()
                    continue
                elif not user_input:
                    continue
                
                # Parse input for images
                parsed_input, image = self.parse_user_input(user_input)
                if not parsed_input:  # Error in parsing
                    continue
                
                # Generate streaming response
                response = self.generate_streaming_response(
                    parsed_input, 
                    image=image,
                    system_message="You are a helpful and knowledgeable AI assistant. Provide detailed, accurate, and engaging responses."
                )
                
                # Add to conversation history
                self.add_to_history("user", parsed_input)
                self.add_to_history("assistant", response)
                
                # Show conversation stats
                print(f"\n💬 Conversation turns: {len(self.conversation_history) // 2}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error during generation: {e}")
                print("💡 Try again or type 'quit' to exit.")


def main():
    """Main function to start the streaming chat."""
    model_path = "./gemma3n_e2b_finetuned_unsloth_8bit_it_final"
    repetition_penalty = 1.1  # Default repetition penalty, adjust as needed
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("💡 Make sure you've completed training and the model directory exists.")
        print("💡 Or update the model_path in this script.")
        sys.exit(1)
    
    # Start the chat
    try:
        chatbot = StreamingChatBot(model_path, repetition_penalty=repetition_penalty)
        chatbot.chat()
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 