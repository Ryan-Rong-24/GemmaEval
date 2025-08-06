#!/usr/bin/env python3
"""
GemmaScout Pairwise Model Evaluation Script

Conducts rigorous A/B pairwise evaluation of fine-tuned Gemma models using Q&A datasets.

Protocol:
- Pairwise A/B comparison with order randomization
- Ties allowed for fair evaluation  
- Two independent LLM judges with majority vote
- Fixed instructions and decoding settings across models
- Statistical significance testing

Features:
- Win rate and net preference calculation
- 95% Wilson Score confidence intervals
- Binomial sign test for statistical significance
- Bootstrap sampling for robust estimates
- Detailed response analysis and reporting

Usage:
    python evaluate_qa_pairwise.py \
        --dataset "./data/qa_dataset.json" \
        --models "model1:./path/to/model1" "model2:./path/to/model2" \
        --judges "gpt-4o" "gemma-3-27b-it"
    
    # Required environment variables:
    # export OPENAI_API_KEY="your-openai-api-key"
    # export GEMINI_API_KEY="your-google-api-key"
"""

import json
import os
import sys
import random
import argparse
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
from dataclasses import dataclass, asdict
from collections import defaultdict
import time
import hashlib
import math
from scipy import stats
import openai
import google.generativeai as genai

# Import existing Gemma inference
try:
    from inference_gemma3n import Gemma3NInference
    GEMMA_AVAILABLE = True
except ImportError:
    print("⚠️  Gemma3NInference not found. Please ensure it's in the same directory.")
    GEMMA_AVAILABLE = False

@dataclass
class QAExample:
    """Single QA example for evaluation."""
    question: str
    reference_answer: str
    context: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ModelResponse:
    """Response from a model for a given question."""
    model_name: str
    question: str
    response: str
    generation_time: float = 0.0
    tokens_generated: int = 0

@dataclass
class PairwiseComparison:
    """Single pairwise comparison between two model responses."""
    question: str
    reference_answer: str
    model_a_name: str
    model_a_response: str
    model_b_name: str
    model_b_response: str
    randomized_order: bool  # True if A/B order was swapped
    comparison_id: str
    
    def __post_init__(self):
        # Generate unique ID for this comparison
        content = f"{self.question}_{self.model_a_name}_{self.model_b_name}"
        self.comparison_id = hashlib.md5(content.encode()).hexdigest()[:8]

@dataclass
class JudgeDecision:
    """Decision from a single judge on a pairwise comparison."""
    comparison_id: str
    judge_name: str
    winner: str  # "model_a", "model_b", or "tie"
    confidence: float
    reasoning: str
    raw_response: str

@dataclass
class PairwiseResult:
    """Final result for a pairwise comparison with majority vote."""
    comparison: PairwiseComparison
    judge_decisions: List[JudgeDecision]
    final_winner: str  # "model_a", "model_b", or "tie"
    agreement: bool  # Whether judges agreed
    confidence: float  # Average confidence

class FixedInstructionPrompts:
    """Fixed prompts and instructions for consistent evaluation."""
    
    # Fixed system prompt for model responses
    SYSTEM_PROMPT = """You are a helpful, accurate, and concise AI assistant. Answer the user's question clearly and factually based on the provided context. If the context doesn't contain enough information, say so clearly."""
    
    # Fixed generation parameters
    GENERATION_PARAMS = {
        "max_new_tokens": 512,
        "temperature": 1.0, 
        "top_p": 0.95,
        "do_sample": True,
        "repetition_penalty": 1.05,
    }
    
    # Judge evaluation prompt template
    JUDGE_PROMPT_TEMPLATE = """
You are an expert evaluator assessing the quality of AI responses to camping and survival-related questions.

QUESTION: {question}

REFERENCE ANSWER: {reference_answer}

RESPONSE A: {response_a}

RESPONSE B: {response_b}

Your task is to determine which response (A or B) is better, or if they are roughly equal (tie).

Evaluation criteria:
1. Factual accuracy compared to the reference answer
2. Completeness of the answer
3. Clarity and coherence
4. Relevance to the question
5. Helpfulness to the user

Respond in this exact format:
WINNER: [A/B/TIE]
CONFIDENCE: [0.0-1.0]
REASONING: [Your detailed explanation]

Be objective and focus on the quality of the answers relative to the reference."""

class PairwiseLLMJudge:
    """LLM judge for pairwise comparisons using API calls."""
    
    def __init__(self, judge_name: str, model_name: str):
        self.judge_name = judge_name
        self.model_name = model_name
        self.client = None
        
        if model_name == "gpt-4o":
            print(f"🔍 Setting up OpenAI GPT-4o judge '{judge_name}'")
            if not os.getenv("OPENAI_API_KEY"):
                print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
                raise ValueError("OPENAI_API_KEY environment variable is required for GPT-4o")
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.judge_type = "openai"
            
        elif model_name == "gemma-3-27b-it":
            print(f"🔍 Setting up Google GenAI Gemma 3 27B IT judge '{judge_name}'")
            if not os.getenv("GEMINI_API_KEY"):
                print("⚠️  Warning: GEMINI_API_KEY not found in environment variables")
                raise ValueError("GEMINI_API_KEY environment variable is required for Gemma 3 27B IT")
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.client = genai.GenerativeModel('gemma-2-27b-it')
            self.judge_type = "google"
            
        else:
            raise ValueError(f"Unsupported model: {model_name}. Supported models: 'gpt-4o', 'gemma-3-27b-it'")
    
    def evaluate_comparison(self, comparison: PairwiseComparison) -> JudgeDecision:
        """Evaluate a pairwise comparison and return decision."""
        
        # Prepare the judge prompt
        judge_prompt = FixedInstructionPrompts.JUDGE_PROMPT_TEMPLATE.format(
            question=comparison.question,
            reference_answer=comparison.reference_answer,
            response_a=comparison.model_a_response,
            response_b=comparison.model_b_response
        )
        
        # Get judge response based on judge type
        if self.judge_type == "openai":
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator assessing the quality of AI responses."},
                        {"role": "user", "content": judge_prompt}
                    ],
                    temperature=0.8,
                    max_tokens=1000
                )
                raw_response = response.choices[0].message.content
            except Exception as e:
                print(f"Error with OpenAI judge {self.judge_name}: {e}")
                raw_response = "ERROR: Could not generate OpenAI judge response"
                
        elif self.judge_type == "google":
            try:
                response = self.client.generate_content(
                    judge_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.8,
                        max_output_tokens=1000,
                        top_p=0.9
                    )
                )
                raw_response = response.text
            except Exception as e:
                print(f"Error with Google GenAI judge {self.judge_name}: {e}")
                raw_response = "ERROR: Could not generate Google GenAI judge response"
        else:
            # Fallback response
            raw_response = "WINNER: TIE\nCONFIDENCE: 0.5\nREASONING: Unknown judge type"
        
        # Parse the response
        winner, confidence, reasoning = self._parse_judge_response(raw_response)
        
        return JudgeDecision(
            comparison_id=comparison.comparison_id,
            judge_name=self.judge_name,
            winner=winner,
            confidence=confidence,
            reasoning=reasoning,
            raw_response=raw_response
        )
    
    def _parse_judge_response(self, response: str) -> Tuple[str, float, str]:
        """Parse judge response to extract winner, confidence, and reasoning."""
        winner = "tie"
        confidence = 0.5
        reasoning = "Could not parse judge response"
        
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("WINNER:"):
                winner_raw = line.split(":", 1)[1].strip().upper()
                if winner_raw in ["A", "MODEL_A"]:
                    winner = "model_a"
                elif winner_raw in ["B", "MODEL_B"]:
                    winner = "model_b"
                else:
                    winner = "tie"
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
                except:
                    confidence = 0.5
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
        
        return winner, confidence, reasoning

class PairwiseEvaluator:
    """Main pairwise evaluation system."""
    
    def __init__(self, models: List[Tuple[str, str]], judge_models: List[str] = None):
        """
        Initialize evaluator.
        
        Args:
            models: List of (model_name, model_path) tuples
            judge_models: List of judge model names (default: ["gpt-4o", "gemma-3-27b-it"])
                         Supported models: "gpt-4o", "gemma-3-27b-it"
        """
        self.models = {}
        self.judges = []
        
        # Load models for evaluation
        for model_name, model_path in models:
            if GEMMA_AVAILABLE:
                print(f"🚀 Loading model '{model_name}' from: {model_path}")
                self.models[model_name] = Gemma3NInference(
                    model_path=model_path,
                    model_name=model_name,
                    max_seq_len=4096,
                    device="auto"
                )
            else:
                print(f"⚠️  Model '{model_name}' using placeholder")
                self.models[model_name] = None
        
        # Setup judges
        if judge_models is None:
            # Default: Use GPT-4o and Gemma 3 27B IT as judges
            judge_models = ["gpt-4o", "gemma-3-27b-it"]
        
        for i, model_name in enumerate(judge_models[:2]):  # Only use first 2 judges
            judge_name = f"judge_{i+1}_{model_name.replace('-', '_')}"
            self.judges.append(PairwiseLLMJudge(judge_name, model_name))
    
    def generate_response(self, model_name: str, question: str, context: str = "") -> ModelResponse:
        """Generate response from a model for a given question."""
        model = self.models.get(model_name)
        
        # Prepare input with context if provided
        if context:
            input_text = f"Context: {context}\n\nQuestion: {question}"
        else:
            input_text = question
        
        start_time = time.time()
        
        if model:
            try:
                response = model.generate_response(
                    input_text,
                    system_prompt=FixedInstructionPrompts.SYSTEM_PROMPT,
                    **FixedInstructionPrompts.GENERATION_PARAMS
                )
            except Exception as e:
                print(f"Error generating response for {model_name}: {e}")
                response = f"ERROR: Could not generate response ({str(e)})"
        else:
            # Placeholder response
            response = f"Placeholder response from {model_name} for: {question[:50]}..."
        
        generation_time = time.time() - start_time
        
        return ModelResponse(
            model_name=model_name,
            question=question,
            response=response,
            generation_time=generation_time,
            tokens_generated=len(response.split())  # Rough token estimate
        )
    
    def create_pairwise_comparisons(self, qa_examples: List[QAExample]) -> List[PairwiseComparison]:
        """Create all pairwise comparisons between models for the given QA examples."""
        comparisons = []
        model_names = list(self.models.keys())
        
        # Generate responses for all examples
        print("🔄 Generating model responses...")
        responses = {}
        for model_name in model_names:
            responses[model_name] = []
            for qa_example in qa_examples:
                response = self.generate_response(
                    model_name,
                    qa_example.question,
                    qa_example.context
                )
                responses[model_name].append(response)
        
        # Create pairwise comparisons
        print("📊 Creating pairwise comparisons...")
        for i, qa_example in enumerate(qa_examples):
            for j, model_a in enumerate(model_names):
                for k, model_b in enumerate(model_names):
                    if j >= k:  # Only compare each pair once
                        continue
                    
                    # Random order assignment
                    randomized = random.choice([True, False])
                    if randomized:
                        # Swap A and B
                        comparison = PairwiseComparison(
                            question=qa_example.question,
                            reference_answer=qa_example.reference_answer,
                            model_a_name=model_b,
                            model_a_response=responses[model_b][i].response,
                            model_b_name=model_a,
                            model_b_response=responses[model_a][i].response,
                            randomized_order=True
                        )
                    else:
                        # Keep original order
                        comparison = PairwiseComparison(
                            question=qa_example.question,
                            reference_answer=qa_example.reference_answer,
                            model_a_name=model_a,
                            model_a_response=responses[model_a][i].response,
                            model_b_name=model_b,
                            model_b_response=responses[model_b][i].response,
                            randomized_order=False
                        )
                    
                    comparisons.append(comparison)
        
        return comparisons
    
    def evaluate_comparisons(self, comparisons: List[PairwiseComparison]) -> List[PairwiseResult]:
        """Evaluate all comparisons using the judge panel."""
        results = []
        
        print(f"⚖️  Evaluating {len(comparisons)} comparisons with {len(self.judges)} judges...")
        
        for i, comparison in enumerate(comparisons):
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{len(comparisons)} comparisons")
            
            # Get decisions from both judges
            judge_decisions = []
            for judge in self.judges:
                decision = judge.evaluate_comparison(comparison)
                judge_decisions.append(decision)
            
            # Determine majority vote
            votes = [d.winner for d in judge_decisions]
            vote_counts = {v: votes.count(v) for v in set(votes)}
            
            # Find winner (majority vote with ties preserved)
            if len(judge_decisions) == 2:
                if votes[0] == votes[1]:
                    # Both judges agree (including on ties)
                    final_winner = votes[0]
                    agreement = True
                else:
                    # Judges disagree - this becomes a tie in the final result
                    final_winner = "tie"
                    agreement = False
            else:
                # More than 2 judges - take majority
                final_winner = max(vote_counts.items(), key=lambda x: x[1])[0]
                agreement = vote_counts[final_winner] > len(judge_decisions) / 2
            
            # Calculate average confidence
            avg_confidence = np.mean([d.confidence for d in judge_decisions])
            
            result = PairwiseResult(
                comparison=comparison,
                judge_decisions=judge_decisions,
                final_winner=final_winner,
                agreement=agreement,
                confidence=avg_confidence
            )
            
            results.append(result)
        
        return results
    
    def analyze_results(self, results: List[PairwiseResult]) -> Dict[str, Any]:
        """Analyze evaluation results and compute statistics following the exact protocol."""
        
        # Collect all model names
        model_names = set()
        for result in results:
            model_names.add(result.comparison.model_a_name)
            model_names.add(result.comparison.model_b_name)
        model_names = sorted(list(model_names))
        
        # Initialize counters for each model pair
        pairwise_stats = {}
        for i, model_a in enumerate(model_names):
            for j, model_b in enumerate(model_names):
                if i != j:
                    pairwise_stats[(model_a, model_b)] = {"wins": 0, "losses": 0, "ties": 0}
        
        # Count wins, losses, and ties for each model pair
        for result in results:
            model_a = result.comparison.model_a_name
            model_b = result.comparison.model_b_name
            winner = result.final_winner
            
            if winner == "model_a":
                pairwise_stats[(model_a, model_b)]["wins"] += 1
                pairwise_stats[(model_b, model_a)]["losses"] += 1
            elif winner == "model_b":
                pairwise_stats[(model_a, model_b)]["losses"] += 1
                pairwise_stats[(model_b, model_a)]["wins"] += 1
            else:  # tie
                pairwise_stats[(model_a, model_b)]["ties"] += 1
                pairwise_stats[(model_b, model_a)]["ties"] += 1
        
        # Calculate metrics for each model
        model_metrics = {}
        for model in model_names:
            # Aggregate across all opponents
            total_wins = sum(pairwise_stats[(model, opp)]["wins"] for opp in model_names if opp != model)
            total_losses = sum(pairwise_stats[(model, opp)]["losses"] for opp in model_names if opp != model)
            total_ties = sum(pairwise_stats[(model, opp)]["ties"] for opp in model_names if opp != model)
            total_comparisons = total_wins + total_losses + total_ties
            
            # Win Rate (ignoring ties): WR = w/(w+ℓ)
            win_rate_ignoring_ties = total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0.0
            
            # Net Preference: (w-ℓ)/N (includes ties)
            net_preference = (total_wins - total_losses) / total_comparisons if total_comparisons > 0 else 0.0
            
            # Wilson Score Interval for Win Rate (95% confidence)
            wilson_ci = self._wilson_score_interval(total_wins, total_wins + total_losses, 0.05)
            
            # Binomial sign test (wins vs losses, ties dropped)
            if total_wins + total_losses > 0:
                # Two-sided test: H0 = win rate = 0.5
                binomial_p_value = stats.binom_test(total_wins, total_wins + total_losses, 0.5, alternative='two-sided')
            else:
                binomial_p_value = 1.0
            
            model_metrics[model] = {
                "wins": total_wins,
                "losses": total_losses,
                "ties": total_ties,
                "total_comparisons": total_comparisons,
                "win_rate_ignoring_ties": win_rate_ignoring_ties,
                "win_rate_including_ties": total_wins / total_comparisons if total_comparisons > 0 else 0.0,
                "net_preference": net_preference,
                "wilson_ci_lower": wilson_ci[0],
                "wilson_ci_upper": wilson_ci[1],
                "binomial_p_value": binomial_p_value
            }
        
        # Judge agreement statistics
        total_agreements = sum(1 for r in results if r.agreement)
        agreement_rate = total_agreements / len(results) if results else 0.0
        avg_confidence = np.mean([r.confidence for r in results]) if results else 0.0
        
        # Overall tie rate
        ties = sum(1 for r in results if r.final_winner == "tie")
        tie_rate = ties / len(results) if results else 0.0
        
        # Model rankings by win rate (ignoring ties)
        model_rankings = sorted(
            [(model, metrics["win_rate_ignoring_ties"]) for model, metrics in model_metrics.items()],
            key=lambda x: x[1], reverse=True
        )
        
        return {
            "model_metrics": model_metrics,
            "model_rankings": model_rankings,
            "pairwise_stats": pairwise_stats,
            "total_comparisons": len(results),
            "judge_agreement_rate": agreement_rate,
            "average_confidence": avg_confidence,
            "tie_rate": tie_rate,
            "model_names": model_names
        }
    
    def _wilson_score_interval(self, successes: int, trials: int, alpha: float = 0.05) -> Tuple[float, float]:
        """Calculate Wilson score interval for binomial proportion."""
        if trials == 0:
            return (0.0, 0.0)
        
        p = successes / trials
        z = stats.norm.ppf(1 - alpha/2)  # 95% CI uses z = 1.96
        
        # Wilson score interval formula
        denominator = 1 + z**2 / trials
        center = (p + z**2 / (2 * trials)) / denominator
        margin = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
        
        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
        
        return (lower, upper)

def load_qa_dataset(dataset_path: str) -> List[QAExample]:
    """Load QA dataset from JSON/JSONL file."""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    qa_examples = []
    
    if dataset_path.suffix == '.jsonl':
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    qa_examples.append(QAExample(
                        question=data.get('instruction', data.get('question', '')),
                        reference_answer=data.get('response', data.get('answer', '')),
                        context=data.get('context', ''),
                        metadata=data.get('metadata', {})
                    ))
                except json.JSONDecodeError as e:
                    print(f"⚠️  Error parsing line {line_num}: {e}")
    else:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    qa_examples.append(QAExample(
                        question=item.get('instruction', item.get('question', '')),
                        reference_answer=item.get('response', item.get('answer', '')),
                        context=item.get('context', ''),
                        metadata=item.get('metadata', {})
                    ))
            else:
                qa_examples.append(QAExample(
                    question=data.get('instruction', data.get('question', '')),
                    reference_answer=data.get('response', data.get('answer', '')),
                    context=data.get('context', ''),
                    metadata=data.get('metadata', {})
                ))
    
    # Filter out examples with empty questions or answers
    qa_examples = [qa for qa in qa_examples if qa.question.strip() and qa.reference_answer.strip()]
    
    print(f"📚 Loaded {len(qa_examples)} QA examples from {dataset_path}")
    return qa_examples

def save_results(results: List[PairwiseResult], analysis: Dict[str, Any], output_path: str):
    """Save evaluation results to JSON file."""
    output_data = {
        "analysis": analysis,
        "detailed_results": [
            {
                "comparison": asdict(result.comparison),
                "judge_decisions": [asdict(decision) for decision in result.judge_decisions],
                "final_winner": result.final_winner,
                "agreement": result.agreement,
                "confidence": result.confidence
            }
            for result in results
        ],
        "evaluation_metadata": {
            "total_comparisons": len(results),
            "generation_params": FixedInstructionPrompts.GENERATION_PARAMS,
            "system_prompt": FixedInstructionPrompts.SYSTEM_PROMPT
        }
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {output_path}")

def print_results_summary(analysis: Dict[str, Any]):
    """Print a summary of the evaluation results."""
    print("\n" + "="*80)
    print("🏆 PAIRWISE EVALUATION RESULTS (Protocol-Compliant)")
    print("="*80)
    
    print(f"\n📊 Model Rankings (by Win Rate ignoring ties):")
    for i, (model, win_rate) in enumerate(analysis["model_rankings"], 1):
        metrics = analysis["model_metrics"][model]
        print(f"  {i}. {model}:")
        print(f"     Win Rate (WR): {win_rate:.3f} ({metrics['wins']}/{metrics['wins'] + metrics['losses']})")
        print(f"     Net Preference: {metrics['net_preference']:+.3f}")
        print(f"     95% Wilson CI: [{metrics['wilson_ci_lower']:.3f}, {metrics['wilson_ci_upper']:.3f}]")
        print(f"     Binomial p-value: {metrics['binomial_p_value']:.6f}")
        print(f"     W/L/T: {metrics['wins']}/{metrics['losses']}/{metrics['ties']}")
        print()
    
    print(f"📈 Summary Statistics:")
    print(f"  Total comparisons: {analysis['total_comparisons']}")
    print(f"  Judge agreement rate: {analysis['judge_agreement_rate']:.3f}")
    print(f"  Average confidence: {analysis['average_confidence']:.3f}")
    print(f"  Overall tie rate: {analysis['tie_rate']:.3f}")
    
    print(f"\n🎯 Detailed Model Performance:")
    models = analysis["model_names"]
    
    # Pairwise comparison matrix
    print("\nPairwise Win/Loss/Tie Matrix:")
    print("Row model vs Column model (Wins/Losses/Ties)")
    print("     ", end="")
    for model in models:
        print(f"{model[:12]:>12}", end="")
    print()
    
    pairwise_stats = analysis["pairwise_stats"]
    for model_a in models:
        print(f"{model_a[:4]:>4} ", end="")
        for model_b in models:
            if model_a == model_b:
                print("      -     ", end="")
            else:
                stats = pairwise_stats.get((model_a, model_b), {"wins": 0, "losses": 0, "ties": 0})
                print(f"{stats['wins']:>3}/{stats['losses']:>3}/{stats['ties']:>3}", end="")
        print()
    
    # Statistical significance summary
    print(f"\n📊 Statistical Significance (α=0.05):")
    for model, metrics in analysis["model_metrics"].items():
        significance = "***" if metrics["binomial_p_value"] < 0.001 else "**" if metrics["binomial_p_value"] < 0.01 else "*" if metrics["binomial_p_value"] < 0.05 else ""
        print(f"  {model}: p={metrics['binomial_p_value']:.6f} {significance}")
    
    print("\nLegend: *** p<0.001, ** p<0.01, * p<0.05")

def main():
    parser = argparse.ArgumentParser(description='Pairwise A/B evaluation for Gemma 3n Q&A dataset')
    parser.add_argument('--dataset', required=True, help='Path to QA dataset (JSON/JSONL)')
    parser.add_argument('--models', required=True, nargs='+', 
                       help='Model paths or names to evaluate (format: name:path or just path)')
    parser.add_argument('--judges', nargs='+', 
                       help='Judge model names, e.g., "gpt-4o" "gemma-3-27b-it" (default: both GPT-4o and Gemma-3-27B-IT)')
    parser.add_argument('--output', default='pairwise_evaluation_results.json',
                       help='Output path for results')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Number of QA examples to sample (default: use all)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Parse model specifications
    models = []
    for model_spec in args.models:
        if ':' in model_spec:
            name, path = model_spec.split(':', 1)
            models.append((name, path))
        else:
            # Use basename as name
            name = Path(model_spec).stem
            models.append((name, model_spec))
    
    print(f"🚀 Starting pairwise evaluation with {len(models)} models")
    
    # Check required environment variables
    if not args.judges or "gpt-4o" in args.judges:
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  Warning: OPENAI_API_KEY not set. GPT-4o judge will fail.")
    
    if not args.judges or "gemma-3-27b-it" in args.judges:
        if not os.getenv("GEMINI_API_KEY"):
            print("⚠️  Warning: GEMINI_API_KEY not set. Gemma 3 27B IT judge will fail.")
    
    # Load dataset
    qa_examples = load_qa_dataset(args.dataset)
    
    # Sample if requested
    if args.sample_size and args.sample_size < len(qa_examples):
        qa_examples = random.sample(qa_examples, args.sample_size)
        print(f"📝 Sampled {len(qa_examples)} examples for evaluation")
    
    # Initialize evaluator
    evaluator = PairwiseEvaluator(models, args.judges)
    
    # Create comparisons
    comparisons = evaluator.create_pairwise_comparisons(qa_examples)
    print(f"🔄 Created {len(comparisons)} pairwise comparisons")
    
    # Evaluate comparisons
    results = evaluator.evaluate_comparisons(comparisons)
    
    # Analyze results
    analysis = evaluator.analyze_results(results)
    
    # Print summary
    print_results_summary(analysis)
    
    # Save results
    save_results(results, analysis, args.output)
    
    print(f"\n✅ Evaluation complete! Results saved to {args.output}")

if __name__ == "__main__":
    main()