#!/usr/bin/env python3
"""
Intelligent Training System with Comprehensive Model and Dataset Classification

This system automatically detects model families and dataset types, then applies
optimal training configurations based on the specific combination detected.

ENHANCED Model Categories (Merged from train.py):
- OpenAI GPT-OSS: Official OpenAI GPT-OSS models with Mxfp4Config, harmony format, MoE expert targeting
- Mistral Family: Enhanced detection for Nemo (12B), Large (123B), Mixtral MoE, standard 7B variants
- Gemma Family: Gemma3 models with eager attention, conservative hyperparameters, BF16 precision
- Phi Family: Microsoft Phi models with ultra-long context support (131k tokens)
- GPT Family: Standard GPT models (non-OSS) with appropriate architectures
- Generic Decoder: Fallback for unknown architectures with auto-detection

ENHANCED Dataset Categories:
- Math/Reasoning: GSM8K, Orca-Math, mathematical word problems, logical reasoning
- Coding: Comprehensive detection for OpenCodeInstruct, rStar-Coder, CodeAlpaca, WizardCoder, MagiCoder
- Conversation: UltraChat, multi-turn dialogue, chat-based interactions
- General Instruction: Standard instruction-response pairs

ADVANCED Features:
- Official OpenAI GPT-OSS training methodology with SFTTrainer and harmony format
- Mistral AI official configurations with variant-specific optimizations
- Gemma3 repository-compliant training with conservative hyperparameters  
- Comprehensive LoRA configurations with MoE expert targeting for GPT-OSS
- Safe tokenization with vocabulary bounds checking
- Model-specific precision, attention, and sequence length handling
- Intelligent repository name cleaning and validation
"""

print("=== INTELLIGENT TRAINING SYSTEM STARTING ===")
print("Initializing imports...")

import os
import sys
import re
from typing import Dict, Any, Tuple, Optional
sys.stdout.flush()

try:
    import torch
    print("✅ PyTorch imported")
    sys.stdout.flush()
except Exception as e:
    print(f"❌ Failed to import torch: {e}")
    sys.stdout.flush()
    exit(1)

try:
    from datasets import load_dataset
    print("✅ Datasets imported")
    sys.stdout.flush()
except Exception as e:
    print(f"❌ Failed to import datasets: {e}")
    sys.stdout.flush()
    exit(1)

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        Mxfp4Config,
    )
    print("✅ Transformers imported")
    sys.stdout.flush()
except Exception as e:
    print(f"❌ Failed to import transformers: {e}")
    sys.stdout.flush()
    exit(1)

try:
    from trl import SFTConfig, SFTTrainer
    print("✅ TRL imported")
    sys.stdout.flush()
except Exception as e:
    print(f"❌ Failed to import trl: {e}")
    print("⚠️  Please install TRL: pip install 'trl>=0.20.0'")
    sys.stdout.flush()
    exit(1)

try:
    from peft import LoraConfig, get_peft_model, PeftModel
    print("✅ PEFT imported")
    sys.stdout.flush()
except Exception as e:
    print(f"❌ Failed to import peft: {e}")
    sys.stdout.flush()
    exit(1)

print("✅ All imports completed successfully!")
sys.stdout.flush()


class ModelClassifier:
    """Intelligent model family detection and configuration"""
    
    @staticmethod
    def classify_model(model_id: str) -> Dict[str, Any]:
        """
        Classify model into family and return optimized configuration
        
        Args:
            model_id: HuggingFace model identifier
            
        Returns:
            Dictionary containing model classification and config
        """
        model_lower = model_id.lower()
        
        # Mistral Family Detection (Enhanced from train.py)
        if any(term in model_lower for term in ['mistral', 'codestral']):
            # Detect MoE (Mixtral) models
            is_mixtral = any(x in model_lower for x in ['mixtral', '8x7b', '8x22b'])
            
            # Detect specific Mistral variants
            max_seq_length = 32768  # Default
            learning_rate = 2e-4    # Default
            
            if 'nemo' in model_lower or '12b' in model_lower:
                max_seq_length = 16384  # Official: Nemo seq_len <= 16384
                model_variant = 'nemo_12b'
            elif 'large' in model_lower or '123b' in model_lower:
                max_seq_length = 8192   # Official: Large v2 seq_len <= 8192
                learning_rate = 1e-6    # Official: Much lower LR for Large v2
                model_variant = 'large_123b'
            elif is_mixtral:
                model_variant = 'mixtral_moe'
            else:
                model_variant = 'standard_7b'
            
            return {
                'family': 'mistral',
                'variant': model_variant,
                'is_moe': is_mixtral,
                'precision': torch.bfloat16,
                'attention': 'sdpa',  # Mistral supports SDPA
                'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                'chat_template': 'mistral_instruct',
                'special_tokens': {
                    'bos_token': '<s>',
                    'eos_token': '</s>',
                    'pad_token': '</s>'
                },
                'max_context': max_seq_length,
                'learning_rate': learning_rate,
                'lora_config': {'r': 16, 'alpha': 32, 'dropout': 0.05},
                'optimization_level': 'very_high',
                'requires_vocab_check': True,
                'training_config': 'mistral_official'
            }
        
        # Gemma Family Detection (Enhanced from train.py)
        elif any(term in model_lower for term in ['gemma', 'google/gemma']):
            return {
                'family': 'gemma',
                'precision': torch.bfloat16,  # BF16 required for Gemma3
                'attention': 'eager',  # CRITICAL: Gemma3 requires eager attention
                'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                'chat_template': 'gemma',
                'special_tokens': {
                    'bos_token': '<bos>',
                    'eos_token': '<eos>',
                    'pad_token': '<eos>'
                },
                'max_context': 8192,
                'learning_rate': 1e-5,  # CRITICAL: Much lower LR than default (from Gemma3 script)
                'lora_config': {'r': 16, 'alpha': 32, 'dropout': 0.05},
                'optimization_level': 'high',
                'training_config': 'gemma3_official',
                'special_params': {
                    'adam_beta2': 0.95,     # Gemma3-specific Adam beta2
                    'weight_decay': 0.1,    # High weight decay
                    'warmup_ratio': 0.03    # 3% warmup
                }
            }
        
        # OpenAI GPT-OSS Family Detection (From train.py)
        elif any(term in model_lower for term in ['gpt-oss', 'gpt_oss', 'openai/gpt-oss', 'openai/gpt_oss']):
            return {
                'family': 'gpt_oss',
                'precision': torch.bfloat16,
                'attention': 'eager',  # Official: eager (not sdpa)
                'quantization_config': Mxfp4Config(dequantize=True),  # Official: Mxfp4Config
                'use_cache': False,  # Critical for training
                'device_map': 'auto',
                'target_modules': 'all-linear',  # Official: target all linear layers
                'target_parameters': [  # CRITICAL: target_parameters for specific MoE expert layers
                    "7.mlp.experts.gate_up_proj",    # Layer 7 expert projections
                    "7.mlp.experts.down_proj",
                    "15.mlp.experts.gate_up_proj",   # Layer 15 expert projections  
                    "15.mlp.experts.down_proj",
                    "23.mlp.experts.gate_up_proj",   # Layer 23 expert projections
                    "23.mlp.experts.down_proj",
                ],
                'chat_template': 'harmony',  # GPT-OSS uses harmony format
                'special_tokens': {
                    'bos_token': None,
                    'eos_token': '<|im_end|>',
                    'pad_token': '<|im_end|>'
                },
                'max_context': 8192,  # GPT-OSS supports long context
                'learning_rate': 2e-4,  # Official: 2e-4 (not 5e-5)
                'lora_config': {'r': 8, 'alpha': 16},  # Official: r=8, alpha=16
                'optimization_level': 'official_openai',
                'training_config': 'gpt_oss_official',
                'use_sft_trainer': True,  # Use SFTTrainer for OpenAI approach
                'training_args_class': 'SFTConfig'  # Use SFTConfig instead of TrainingArguments
            }
        
        # Phi Family Detection (Enhanced from train.py) 
        elif any(term in model_lower for term in ['phi', 'microsoft/phi']):
            return {
                'family': 'phi',
                'precision': 'auto',  # Phi models handle dtype automatically
                'attention': 'sdpa',
                'target_modules': ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
                'chat_template': 'phi_instruct',
                'special_tokens': {
                    'bos_token': '<|endoftext|>',
                    'eos_token': '<|endoftext|>',
                    'pad_token': '<|endoftext|>'
                },
                'max_context': 131072,  # Phi-3.5 supports very long context
                'learning_rate': 2e-4,  # Can handle slightly higher learning rates
                'lora_config': {'r': 16, 'alpha': 32, 'dropout': 0.05},
                'optimization_level': 'ultra_high'
            }
        
        # GPT Family Detection (Standard GPT models, not GPT-OSS)
        elif 'gpt' in model_lower and 'gpt_oss' not in model_lower and 'gpt-oss' not in model_lower:
            return {
                'family': 'gpt',
                'precision': torch.float16,
                'attention': 'sdpa',
                'target_modules': ["c_attn", "c_proj", "c_fc"],  # GPT architecture
                'chat_template': 'generic',
                'special_tokens': {
                    'bos_token': None,
                    'eos_token': '<|endoftext|>',
                    'pad_token': '<|endoftext|>'
                },
                'max_context': 4096,
                'learning_rate': 2e-4,
                'lora_config': {'r': 16, 'alpha': 32, 'dropout': 0.05},
                'optimization_level': 'standard'
            }
        
        # Generic Decoder (Fallback)
        else:
            print(f"⚠️  Unknown model family for {model_id}, using generic configuration")
            return {
                'family': 'generic_decoder',
                'precision': torch.float16,
                'attention': 'sdpa',
                'target_modules': None,  # Will auto-detect
                'chat_template': 'generic',
                'special_tokens': {
                    'bos_token': None,
                    'eos_token': None,
                    'pad_token': None
                },
                'max_context': 4096,
                'learning_rate': 2e-4,
                'lora_config': {'r': 16, 'alpha': 32, 'dropout': 0.05},
                'optimization_level': 'standard'
            }


class DatasetClassifier:
    """Intelligent dataset type detection and configuration"""
    
    @staticmethod
    def classify_dataset(dataset_id: str, dataset_subset: Optional[str] = None) -> Dict[str, Any]:
        """
        Classify dataset into type and return optimized configuration
        
        Args:
            dataset_id: HuggingFace dataset identifier
            dataset_subset: Optional dataset subset/split
            
        Returns:
            Dictionary containing dataset classification and config
        """
        dataset_lower = dataset_id.lower()
        subset_lower = (dataset_subset or "").lower()
        
        # Math/Reasoning Dataset Detection
        math_patterns = [
            'gsm8k', 'orca-math', 'math-word-problems', 'mathematical',
            'reasoning', 'grade-school-math', 'math_qa', 'aqua_rat',
            'mathqa', 'math-problems', 'arithmetic', 'algebra',
            'geometry', 'calculus', 'statistics'
        ]
        
        if any(pattern in dataset_lower for pattern in math_patterns):
            return {
                'type': 'math_reasoning',
                'sample_size': 15000,  # Larger for math complexity
                'max_length': 2048,    # Math problems can be verbose
                'formatting_strategy': 'step_by_step',
                'training_epochs': 3,  # More epochs for reasoning
                'learning_rate': 5e-5, # Lower LR for stable math learning
                'batch_size': 1,       # Complex problems need careful attention
                'gradient_accumulation': 8,
                'optimization_focus': 'reasoning_accuracy'
            }
        
        # Coding Dataset Detection (Enhanced from train.py)
        coding_patterns = [
            # OpenCodeInstruct variations
            'opencodeinstruct', 'open-code-instruct', 'open_code_instruct',
            # Organization/dataset format for OpenCodeInstruct
            '/opencodeinstruct', '/open-code-instruct', '/open_code_instruct',
            # Specific known OpenCodeInstruct repositories  
            'nvidia/opencodeinstruct', 'sahilchiddarwar/opencodeinstruct', 'm-a-p/opencodeinstruct',
            # Other high-quality coding datasets
            'code_alpaca', 'codealpaca', 'wizardcoder', 'wizard-coder', 'evol-codealpaca',
            'magicoder', 'magic-coder', 
            # rStar-Coder dataset
            'rstar-coder', 'microsoft/rstar-coder',
            # Additional coding datasets
            'code-instruct', 'coding-instruct', 'programming-instruct',
            'programming', 'coding', 'python', 'javascript', 'java', 'cpp', 'software'
        ]
        
        if any(pattern in dataset_lower for pattern in coding_patterns):
            # Determine specific dataset type for optimal configuration
            dataset_type = "generic-coding"
            sample_size = 5000  # Default
            
            if any(term in dataset_lower for term in ["opencodeinstruct", "open-code-instruct", "open_code_instruct"]):
                dataset_type = "opencodeinstruct"
                sample_size = 10000  # Large sample for comprehensive dataset
            elif any(term in dataset_lower for term in ["rstar-coder", "microsoft/rstar-coder"]):
                dataset_type = "rstar-coder"
                sample_size = 8000   # Good sample for competitive programming
            elif any(term in dataset_lower for term in ["code_alpaca", "codealpaca", "evol-codealpaca"]):
                dataset_type = "code-alpaca"
                sample_size = 6000   # Medium sample for specialized datasets
            elif any(term in dataset_lower for term in ["wizardcoder", "wizard-coder"]):
                dataset_type = "wizardcoder"
                sample_size = 6000   # Medium sample for specialized datasets
            elif any(term in dataset_lower for term in ["magicoder", "magic-coder"]):
                dataset_type = "magicoder"  
                sample_size = 6000   # Medium sample for specialized datasets
                
            return {
                'type': 'coding',
                'subtype': dataset_type,
                'sample_size': sample_size,
                'max_length': 4096,    # Longer sequences for coding tasks
                'formatting_strategy': 'code_instruction',
                'training_epochs': 2,  # Sufficient for code patterns
                'learning_rate': 1e-4, # Moderate LR for code learning  
                'batch_size': 1,       # Complex code needs attention
                'gradient_accumulation': 6,
                'optimization_focus': 'code_generation'
            }
        
        # Conversation Dataset Detection
        conversation_patterns = [
            'ultrachat', 'ultra-chat', 'conversation', 'dialogue',
            'chat', 'multi-turn', 'assistant', 'helpfulness',
            'harmless', 'honest', 'anthropic', 'sharegpt'
        ]
        
        if any(pattern in dataset_lower for pattern in conversation_patterns):
            return {
                'type': 'conversation',
                'sample_size': 20000,  # Large for conversation diversity
                'max_length': 3072,    # Multi-turn needs more context
                'formatting_strategy': 'multi_turn_chat',
                'training_epochs': 2,  # Good for conversation flow
                'learning_rate': 2e-4, # Standard LR for conversations
                'batch_size': 2,       # Can handle slightly larger batches
                'gradient_accumulation': 4,
                'optimization_focus': 'helpfulness_safety'
            }
        
        # General Instruction (Fallback)
        else:
            return {
                'type': 'general_instruction',
                'sample_size': 10000,  # Standard size
                'max_length': 2048,    # Standard length
                'formatting_strategy': 'instruction_response',
                'training_epochs': 1,  # Single epoch for general
                'learning_rate': 2e-4, # Standard learning rate
                'batch_size': 2,       # Standard batch size
                'gradient_accumulation': 4,
                'optimization_focus': 'general_capability'
            }


class IntelligentFormatter:
    """Dynamic prompt formatting based on model+dataset combination"""
    
    @staticmethod
    def format_example(example: Dict[str, Any], model_config: Dict[str, Any], 
                      dataset_config: Dict[str, Any]) -> Dict[str, str]:
        """
        Format training example based on model family and dataset type
        
        Args:
            example: Raw dataset example
            model_config: Model family configuration
            dataset_config: Dataset type configuration
            
        Returns:
            Formatted example with 'text' field
        """
        model_family = model_config['family']
        dataset_type = dataset_config['type']
        
        # Extract content from various possible field names
        instruction = ""
        response = ""
        
        # Try different field combinations for various dataset structures
        if 'instruction' in example and 'response' in example:
            instruction = example['instruction']
            response = example['response']
        elif 'question' in example and 'answer' in example:
            instruction = example['question']
            response = example['answer']
        elif 'problem' in example and 'solution' in example:
            instruction = example['problem']
            response = example['solution']
        elif 'prompt' in example and 'completion' in example:
            instruction = example['prompt']
            response = example['completion']
        elif 'messages' in example:
            # Handle conversation format
            messages = example['messages']
            if len(messages) >= 2:
                instruction = messages[0].get('content', '')
                response = messages[1].get('content', '')
        elif 'text' in example:
            # Single text field - split or use as is
            return {"text": example['text']}
        else:
            # Fallback: convert entire example to string
            return {"text": str(example)}
        
        # Apply model-specific formatting
        if model_family == 'gpt_oss':
            # GPT-OSS models REQUIRE harmony response format
            # Use the tokenizer's chat template which automatically applies harmony format
            
            # Prepare messages in OpenAI chat format
            if instruction and response:
                messages = [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response}
                ]
            else:
                # Fallback: convert entire example to string
                messages = [
                    {"role": "user", "content": str(example)},
                    {"role": "assistant", "content": "I understand."}
                ]
            
            # Apply chat template (automatically uses harmony format for GPT-OSS)
            try:
                # This would use the tokenizer's chat template, but since we don't have tokenizer here,
                # we'll use a basic harmony format approximation
                formatted_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
            except Exception as e:
                # Fallback to simple format
                formatted_text = f"User: {instruction}\n\nAssistant: {response}"
                
        elif model_family == 'mistral':
            if dataset_type == 'math_reasoning':
                # Enhanced math formatting for Mistral
                formatted_text = f"[INST] {instruction}\n\nPlease solve this step by step, showing your work clearly. [/INST] {response}"
            elif dataset_type == 'coding':
                # Enhanced coding formatting for Mistral
                if 'code' in instruction.lower() or 'programming' in instruction.lower():
                    formatted_text = f"[INST] {instruction}\n\nPlease provide a complete and well-commented solution. [/INST] {response}"
                else:
                    formatted_text = f"[INST] {instruction} [/INST] {response}"
            else:
                # Standard Mistral formatting
                formatted_text = f"[INST] {instruction} [/INST] {response}"
                
        elif model_family == 'gemma':
            if dataset_type == 'math_reasoning':
                # Gemma math formatting
                formatted_text = f"<bos><start_of_turn>user\n{instruction}\nPlease explain your reasoning step by step.<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"
            elif dataset_type == 'coding':
                # Gemma coding formatting  
                formatted_text = f"<bos><start_of_turn>user\n{instruction}\nPlease provide clean, documented code.<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"
            else:
                # Standard Gemma formatting
                formatted_text = f"<bos><start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"
                
        elif model_family == 'phi':
            if dataset_type == 'math_reasoning':
                # Phi math formatting
                formatted_text = f"<|system|>\nYou are a helpful math tutor. Solve problems step by step.<|end|>\n<|user|>\n{instruction}<|end|>\n<|assistant|>\n{response}<|end|>"
            elif dataset_type == 'coding':
                # Phi coding formatting
                formatted_text = f"<|system|>\nYou are an expert programmer. Write clean, efficient code.<|end|>\n<|user|>\n{instruction}<|end|>\n<|assistant|>\n{response}<|end|>"
            else:
                # Standard Phi formatting
                formatted_text = f"<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\n{instruction}<|end|>\n<|assistant|>\n{response}<|end|>"
                
        else:
            # Generic formatting
            formatted_text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        
        return {"text": formatted_text}


class IntelligentTrainer:
    """Main training orchestrator with intelligent configuration"""
    
    def __init__(self):
        self.model_classifier = ModelClassifier()
        self.dataset_classifier = DatasetClassifier()
        self.formatter = IntelligentFormatter()
    
    def validate_and_sanitize_repo_name(self, repo_name: str) -> str:
        """
        Validate and sanitize HuggingFace repository name
        
        Args:
            repo_name: Original repository name
            
        Returns:
            Sanitized repository name in correct format
        """
        if not repo_name:
            raise ValueError("Repository name cannot be empty")
        
        # Remove any leading/trailing whitespace
        repo_name = repo_name.strip()
        
        # Count number of slashes
        slash_count = repo_name.count('/')
        
        if slash_count == 0:
            # Just a repo name, keep as is
            return repo_name
        elif slash_count == 1:
            # Correct format: username/repo_name
            return repo_name
        else:
            # Multiple slashes - need to sanitize
            parts = repo_name.split('/')
            
            # Take the first part (username) and last part (repo_name)
            username = parts[0]
            repo_base = parts[-1]
            
            # Combine middle parts with the repo name if they contain meaningful info
            if len(parts) > 2:
                middle_parts = parts[1:-1]
                # Filter out common model family names that shouldn't be in repo name
                excluded_terms = {'google', 'microsoft', 'meta', 'facebook', 'openai', 'anthropic', 'huggingface'}
                meaningful_parts = [part for part in middle_parts if part.lower() not in excluded_terms]
                
                if meaningful_parts:
                    # Combine meaningful parts with repo name
                    repo_base = '-'.join(meaningful_parts + [repo_base])
            
            sanitized = f"{username}/{repo_base}"
            
            print(f"🔧 Sanitized repository name:")
            print(f"   Original: {repo_name}")
            print(f"   Issue: Contains {slash_count} slashes (max allowed: 1)")
            print(f"   Fixed: {sanitized}")
            
            return sanitized
    
    def clean_repo_name(self, repo_name: str) -> str:
        """
        Clean repository name to valid HuggingFace format
        
        Examples:
            hamzafaisal/google/gemma-3-1b-it-lora -> hamzafaisal/gemma-3-1b-it-lora
            user/microsoft/phi-3-mini-lora -> user/phi-3-mini-lora
            username/meta/llama-2-7b-lora -> username/llama-2-7b-lora
        
        Args:
            repo_name: Original repository name
            
        Returns:
            Cleaned repository name in correct format (username/repo_name)
        """
        if not repo_name:
            raise ValueError("Repository name cannot be empty")
        
        # Remove leading/trailing whitespace
        repo_name = repo_name.strip()
        
        # Split by slashes
        parts = repo_name.split('/')
        
        if len(parts) == 1:
            # Just repo name, return as is
            return repo_name
        elif len(parts) == 2:
            # Already correct format: username/repo_name
            return repo_name
        else:
            # Multiple slashes - clean it up
            username = parts[0]
            
            # Common organization names to remove from middle
            organizations_to_remove = {
                'google', 'microsoft', 'meta', 'facebook', 'openai', 
                'anthropic', 'huggingface', 'mistralai', 'mistral',
                'nvidia', 'apple', 'ibm', 'amazon', 'salesforce'
            }
            
            # Process middle parts and final part
            middle_and_final = parts[1:]
            
            # Remove organization names but keep meaningful parts
            cleaned_parts = []
            for part in middle_and_final:
                if part.lower() not in organizations_to_remove:
                    cleaned_parts.append(part)
            
            # If we removed everything, use the last part
            if not cleaned_parts:
                cleaned_parts = [parts[-1]]
            
            # Join remaining parts with hyphens
            repo_name_clean = '-'.join(cleaned_parts)
            
            cleaned_repo = f"{username}/{repo_name_clean}"
            
            print(f"🧹 Cleaned repository name:")
            print(f"   Original: {repo_name}")
            print(f"   Cleaned:  {cleaned_repo}")
            print(f"   Removed:  {', '.join(set(parts[1:]) - set(cleaned_parts))}")
            
            return cleaned_repo
    
    def train(self):
        """Execute intelligent training pipeline"""
        print("\n=== INTELLIGENT TRAINING PIPELINE ===")
        print("🧠 Analyzing model and dataset for optimal configuration...")
        
        # Read environment variables
        hf_token = os.getenv("HF_TOKEN")
        base_model_id = os.getenv("BASE_MODEL_ID")
        dataset_id = os.getenv("DATASET_ID")
        dataset_subset = os.getenv("DATASET_SUBSET", None)
        lora_model_repo = os.getenv("LORA_MODEL_REPO")
        
        if not all([hf_token, base_model_id, dataset_id, lora_model_repo]):
            print("❌ Missing required environment variables")
            exit(1)
        
        print(f"📊 Model: {base_model_id}")
        print(f"📊 Dataset: {dataset_id}")
        if dataset_subset:
            print(f"📊 Dataset Subset: {dataset_subset}")
        
        # Classify model and dataset
        print("\n🔍 CLASSIFICATION PHASE")
        model_config = self.model_classifier.classify_model(base_model_id)
        dataset_config = self.dataset_classifier.classify_dataset(dataset_id, dataset_subset)
        
        print(f"🏷️  Model Family: {model_config['family'].upper()}")
        print(f"🏷️  Dataset Type: {dataset_config['type'].upper()}")
        print(f"🎯 Optimization Focus: {dataset_config['optimization_focus']}")
        
        # Load model with intelligent configuration
        print(f"\n⚙️  LOADING MODEL WITH {model_config['family'].upper()} OPTIMIZATIONS")
        model_kwargs = {
            "token": hf_token,
            "device_map": model_config.get('device_map', "cuda"),
            "trust_remote_code": True,
            "resume_download": True,
        }
        
        # Apply model family specific settings
        if model_config['precision'] != 'auto':
            model_kwargs["torch_dtype"] = model_config['precision']
        if model_config['attention']:
            model_kwargs["attn_implementation"] = model_config['attention']
            
        # Special configurations for specific model families
        if model_config['family'] == 'gpt_oss':
            # GPT-OSS specific configuration
            model_kwargs.update({
                "quantization_config": model_config['quantization_config'],
                "use_cache": model_config['use_cache'],
                "device_map": model_config['device_map'],
            })
            print(f"🚀 Loading GPT-OSS with OFFICIAL OpenAI configuration")
            print("--> Features: Mxfp4Config quantization, eager attention, use_cache=False")
            print("--> Following OpenAI cookbook guide specifications")
        elif model_config['family'] == 'mistral':
            if model_config.get('is_moe', False):
                print("🔥 MISTRAL MoE (MIXTRAL) MODEL DETECTED")
                print("⚠️  MoE Training Notice: Mistral AI reports higher performance variance with MoE models")
                print("⚠️  Recommendation: Run multiple training instances with different seeds for best results")
                print(f"--> Loading Mixtral MoE model with BF16 precision and SDPA attention")
            else:
                print(f"--> Loading Mistral model with BF16 precision and SDPA attention")
        elif model_config['family'] == 'gemma':
            print(f"--> Loading Gemma3 model with BF16 precision and EAGER attention")
            print("--> Note: Gemma3 models require eager attention for optimal training")
            
        print(f"🔧 Loading with precision: {model_config['precision']}")
        print(f"🔧 Attention implementation: {model_config['attention']}")
        
        model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
        
        # Configure tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token, trust_remote_code=True)
        
        # Set special tokens based on model family
        special_tokens = model_config['special_tokens']
        if tokenizer.pad_token is None:
            if special_tokens['pad_token']:
                tokenizer.pad_token = special_tokens['pad_token']
            else:
                tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✅ Model loaded with {model_config['optimization_level']} optimization level")
        
        # Resize embeddings and configure LoRA
        model.resize_token_embeddings(len(tokenizer))
        
        # Intelligent LoRA configuration
        print(f"\n🎛️  CONFIGURING LoRA FOR {model_config['family'].upper()}")
        
        target_modules = model_config['target_modules']
        if target_modules is None:
            # Auto-detect for unknown models
            print("🔍 Auto-detecting LoRA target modules...")
            all_linear_names = []
            for name, module in model.named_modules():
                if "Linear" in str(type(module)):
                    all_linear_names.append(name.split('.')[-1])
            target_modules = list(set(all_linear_names))
            print(f"📡 Detected modules: {target_modules}")
        
        # Get LoRA configuration from model config
        lora_config_dict = model_config.get('lora_config', {'r': 16, 'alpha': 32, 'dropout': 0.05})
        lora_rank = lora_config_dict['r']
        lora_alpha = lora_config_dict['alpha']
        lora_dropout = lora_config_dict.get('dropout', 0.05)
        
        # Override for specific combinations
        if model_config['family'] == 'phi' and dataset_config['type'] == 'math_reasoning':
            lora_rank = 32  # Higher rank for complex reasoning
            lora_alpha = 64
        elif dataset_config['type'] == 'coding':
            lora_rank = 24  # Medium-high rank for code
            lora_alpha = 48
        
        print(f"🎯 LoRA rank: {lora_rank}")
        print(f"🎯 LoRA alpha: {lora_alpha}")
        print(f"🎯 Target modules: {target_modules}")
        
        # CRITICAL FIX: Add embedding and LM head layers to modules_to_save
        # This prevents CUDA device-side assert errors when using LoRA with resized embeddings
        embedding_layer_names = []
        lm_head_names = []
        
        for name, module in model.named_modules():
            if "embed" in name.lower() and ("token" in name.lower() or "word" in name.lower()):
                embedding_layer_names.append(name)
            elif "lm_head" in name.lower() or ("head" in name.lower() and "embed" not in name.lower()):
                lm_head_names.append(name)
        
        print(f"--> Found embedding layers: {embedding_layer_names}")
        print(f"--> Found LM head layers: {lm_head_names}")
        
        # Determine modules to save based on model architecture
        modules_to_save = []
        if embedding_layer_names:
            modules_to_save.extend(embedding_layer_names)
        if lm_head_names:
            modules_to_save.extend(lm_head_names)
        
        # Fallback for common naming patterns
        if not modules_to_save:
            common_embed_names = ["embed_tokens", "wte", "token_embedding", "word_embeddings"]
            common_head_names = ["lm_head", "head", "classifier"]
            modules_to_save = common_embed_names + common_head_names
            print(f"--> Using fallback modules_to_save: {modules_to_save}")
        else:
            print(f"--> Using detected modules_to_save: {modules_to_save}")
        
        # Configure LoRA based on model family
        if model_config['family'] == 'gpt_oss':
            # OFFICIAL OpenAI configuration for GPT-OSS (prefer native target_parameters if available)
            try:
                from peft import __version__ as peft_version
                from packaging.version import parse as vparse
                supports_target_parameters = vparse(peft_version) >= vparse("0.17.0")
            except Exception:
                supports_target_parameters = False

            if supports_target_parameters and model_config.get('target_parameters'):
                lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=[],
                    target_parameters=model_config['target_parameters'],
                    task_type="CAUSAL_LM",
                    modules_to_save=modules_to_save,
                )
                print(f"--> GPT-OSS LoRA config using PEFT target_parameters (>=0.17.0)")
            else:
                try:
                    layer_indices = sorted({int(p.split('.')[0]) for p in model_config.get('target_parameters', [])})
                except Exception:
                    layer_indices = []

                layers_pattern = None
                try:
                    if hasattr(model, "model") and hasattr(model.model, "layers"):
                        layers_pattern = "model.layers"
                    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                        layers_pattern = "transformer.h"
                except Exception:
                    layers_pattern = None

                lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=["gate_up_proj", "down_proj"],
                    task_type="CAUSAL_LM",
                    modules_to_save=modules_to_save,
                    layers_to_transform=layer_indices if layer_indices else None,
                    layers_pattern=layers_pattern,
                )
                print(f"--> GPT-OSS LoRA config using layers_to_transform fallback, layers={layer_indices}, pattern={layers_pattern}")
        else:
            # Standard LoRA configuration for other models
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                task_type="CAUSAL_LM",
                modules_to_save=modules_to_save,
            )
            print(f"--> Standard LoRA config: r={lora_rank}, alpha={lora_alpha}")
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Load and format dataset
        print(f"\n📂 LOADING {dataset_config['type'].upper()} DATASET")
        dataset = load_dataset(dataset_id, name=dataset_subset, split="train")
        print(f"📊 Original size: {len(dataset):,} examples")
        
        # Apply intelligent sampling
        sample_size = min(len(dataset), dataset_config['sample_size'])
        dataset = dataset.select(range(sample_size))
        print(f"📊 Using: {len(dataset):,} examples (optimized for {dataset_config['type']})")
        
        # Apply intelligent formatting
        print(f"🎨 Applying {model_config['family']}-{dataset_config['type']} formatting...")
        
        def format_function(example):
            return self.formatter.format_example(example, model_config, dataset_config)
        
        dataset = dataset.map(format_function)
        
        # Tokenization with intelligent max length
        max_length = min(dataset_config['max_length'], model_config['max_context'])
        print(f"📏 Max sequence length: {max_length}")
        
        def tokenize_function(examples):
            tokens = tokenizer(examples["text"], truncation=True, padding=False, max_length=max_length)
            
            # Safety: ensure all tokens are within bounds
            actual_vocab_size = model.get_input_embeddings().weight.size(0)
            for key in tokens:
                if key == 'input_ids':
                    tokens[key] = [[min(max(token_id, 0), actual_vocab_size-1) for token_id in seq] for seq in tokens[key]]
            
            return tokens
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        
        # Create intelligent training arguments with model-specific optimizations
        print(f"\n🏋️  CONFIGURING TRAINING FOR {dataset_config['optimization_focus'].upper()}")
        
        # Base configuration from dataset type
        batch_size = dataset_config['batch_size']
        grad_accumulation = dataset_config['gradient_accumulation']
        base_learning_rate = dataset_config['learning_rate']
        epochs = dataset_config['training_epochs']
        
        # Override learning rate with model-specific values if available
        learning_rate = model_config.get('learning_rate', base_learning_rate)
        
        # Configure precision based on model family
        use_bf16 = model_config['precision'] == torch.bfloat16
        use_fp16 = model_config['precision'] == torch.float16
        
        # Model-specific training configuration
        if model_config['family'] == 'gpt_oss' and model_config.get('use_sft_trainer', False):
            # OFFICIAL OpenAI GPT-OSS configuration using TRL
            print("🚀 APPLYING OFFICIAL GPT-OSS TRAINING CONFIGURATION")
            print("--> Using OpenAI cookbook guide specifications")
            
            training_args = SFTConfig(
                output_dir="./output",
                learning_rate=learning_rate,              # Official: 2e-4 (not 5e-5)
                gradient_checkpointing=True,
                num_train_epochs=1,                      # Official: 1 epoch (not 2)
                logging_steps=1,
                per_device_train_batch_size=4,           # Official: 4 (not 1)
                gradient_accumulation_steps=4,           # Official: 4 (not 16)
                max_seq_length=min(model_config['max_context'], 2048),  # Use correct SFTConfig arg
                warmup_ratio=0.03,                       # Official: warmup_ratio (not warmup_steps)
                lr_scheduler_type="cosine_with_min_lr",  # Official scheduler
                lr_scheduler_kwargs={"min_lr_rate": 0.1},  # Official scheduler config
                bf16=use_bf16,
                fp16=use_fp16,
                dataloader_drop_last=True,
                remove_unused_columns=False,
                eval_strategy="no",
                optim="adamw_torch",
                push_to_hub=False,
            )
            print(f"--> OFFICIAL config: LR={learning_rate}, batch=4, epochs=1, warmup_ratio=0.03")
            
        elif model_config['family'] == 'mistral':
            print("🔥 APPLYING MISTRAL AI OFFICIAL TRAINING CONFIGURATION")
            print("--> Using Mistral AI fine-tuning methodology")
            
            # Get Mistral-specific parameters
            max_seq_length = model_config['max_context']
            
            # Mistral-specific messages
            if model_config.get('variant') == 'nemo_12b':
                print(f"--> Mistral Nemo (12B) detected: max_seq_length={max_seq_length}")
            elif model_config.get('variant') == 'large_123b':
                print(f"--> Mistral Large v2 (123B) detected: max_seq_length={max_seq_length}, lr={learning_rate}")
            elif model_config.get('is_moe', False):
                print(f"--> Mixtral MoE detected: max_seq_length={max_seq_length}")
                print("⚠️  MoE Variance: Consider running multiple seeds for optimal results")
            else:
                print(f"--> Standard Mistral (7B) detected: max_seq_length={max_seq_length}")
            
            training_args = TrainingArguments(
                output_dir="./output",
                learning_rate=learning_rate,             # Model-specific learning rate
                per_device_train_batch_size=1,          # Mistral recommendation for memory efficiency
                gradient_accumulation_steps=4,          # Effective batch size = 4
                num_train_epochs=3,                     # Standard for fine-tuning
                warmup_ratio=0.1,                       # 10% warmup (standard)
                weight_decay=0.1,                       # Mistral repository default
                lr_scheduler_type="cosine",             # Cosine scheduler
                logging_steps=5,
                save_strategy="steps",
                save_steps=500,                         # Mistral repository default
                eval_strategy="no",
                optim="adamw_torch",
                bf16=use_bf16,                          # BF16 for Mistral models
                fp16=use_fp16,
                gradient_checkpointing=True,            # Memory optimization
                dataloader_drop_last=True,
                remove_unused_columns=False,
                max_grad_norm=1.0,
                # Note: sequence length is enforced during tokenization, not a TrainingArguments parameter
            )
            
        elif model_config['family'] == 'gemma':
            print("🔥 APPLYING GEMMA3-OPTIMIZED TRAINING PARAMETERS")
            print("--> Using Gemma3 repository specifications for optimal results")
            
            # Get Gemma-specific parameters
            special_params = model_config.get('special_params', {})
            
            training_args = TrainingArguments(
                output_dir="./output",
                learning_rate=learning_rate,            # CRITICAL: Much lower LR than default
                per_device_train_batch_size=1,          # Small batch size
                gradient_accumulation_steps=4,          # Adjust for effective batch size
                num_train_epochs=1,                     # Single epoch
                warmup_ratio=special_params.get('warmup_ratio', 0.03),  # 3% warmup
                weight_decay=special_params.get('weight_decay', 0.1),   # High weight decay
                adam_beta2=special_params.get('adam_beta2', 0.95),      # Gemma3-specific Adam beta2
                lr_scheduler_type="cosine",             # Cosine scheduler
                logging_steps=5,
                save_strategy="steps",
                save_steps=200,
                eval_strategy="no",
                optim="adamw_torch",
                bf16=use_bf16,                          # BF16 precision
                fp16=use_fp16,
                gradient_checkpointing=True,            # Memory optimization
                dataloader_drop_last=True,
                remove_unused_columns=False,
                max_grad_norm=1.0,
            )
            print(f"--> GEMMA3 config: LR={learning_rate} (20x lower), warmup_ratio=0.03, adam_beta2=0.95")
            
        else:
            # Standard training arguments for other models
            training_args = TrainingArguments(
                output_dir="./output",
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_accumulation,
                num_train_epochs=epochs,
                learning_rate=learning_rate,
                warmup_steps=200,
                logging_steps=5,
                save_strategy="steps",
                save_steps=500,
                eval_strategy="no",
                optim="adamw_torch",
                lr_scheduler_type="cosine",
                bf16=use_bf16,
                fp16=use_fp16,
                gradient_checkpointing=True,
                dataloader_drop_last=True,
                remove_unused_columns=False,
                max_grad_norm=1.0,
            )
        
        print(f"🎯 Effective batch size: {batch_size * grad_accumulation}")
        print(f"🎯 Learning rate: {learning_rate}")
        print(f"🎯 Training epochs: {epochs}")
        print(f"🎯 Precision: {'BF16' if use_bf16 else 'FP16' if use_fp16 else 'FP32'}")
        
        # Safe data collator
        class SafeDataCollator(DataCollatorForLanguageModeling):
            def __init__(self, tokenizer, mlm=False):
                super().__init__(tokenizer=tokenizer, mlm=mlm)
                self.vocab_size = model.get_input_embeddings().weight.size(0)
                
            def __call__(self, features):
                batch = super().__call__(features)
                if 'input_ids' in batch:
                    batch['input_ids'] = torch.clamp(batch['input_ids'], 0, self.vocab_size - 1)
                if 'labels' in batch:
                    valid_mask = batch['labels'] != -100
                    batch['labels'] = torch.where(
                        valid_mask, 
                        torch.clamp(batch['labels'], 0, self.vocab_size - 1),
                        batch['labels']
                    )
                return batch
        
        data_collator = SafeDataCollator(tokenizer=tokenizer, mlm=False)
        
        # Initialize trainer based on model type
        if model_config['family'] == 'gpt_oss' and model_config.get('use_sft_trainer', False):
            # OFFICIAL OpenAI approach: Use SFTTrainer with automatic harmony format handling
            print("🚀 Using SFTTrainer for GPT-OSS (Official OpenAI approach)")
            print("--> SFTTrainer handles harmony format automatically via processing_class")
            
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,  # Use original dataset, not tokenized
                processing_class=tokenizer,  # SFTTrainer handles tokenization + harmony format
            )
        else:
            # Standard trainer for other models
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
        
        # Training summary
        print(f"\n🚀 TRAINING SUMMARY")
        print(f"   Model Family: {model_config['family'].upper()}")
        print(f"   Dataset Type: {dataset_config['type'].upper()}")
        print(f"   Optimization: {dataset_config['optimization_focus']}")
        print(f"   Samples: {len(tokenized_dataset):,}")
        print(f"   Max Length: {max_length}")
        print(f"   LoRA Rank: {lora_rank}")
        print(f"   Effective Batch: {batch_size * grad_accumulation}")
        print(f"   Learning Rate: {learning_rate}")
        print(f"   Epochs: {epochs}")
        
        # Start training
        print(f"\n🔥 STARTING INTELLIGENT TRAINING...")
        trainer.train()
        print("✅ Training completed successfully!")
        
        # Save and merge model with repository validation
        print(f"\n💾 Preparing to merge and upload full model...")
        
        # Clean and validate repository name
        cleaned_repo = self.clean_repo_name(lora_model_repo)
        
        if cleaned_repo != lora_model_repo:
            print(f"⚠️  Repository name was cleaned")
        
        print(f"🔀 Merging LoRA weights with base model...")
        
        # Merge the LoRA weights with the base model
        try:
            # Get the base model again for merging
            print(f"📥 Loading base model for merging: {base_model_id}")
            base_model_for_merge = AutoModelForCausalLM.from_pretrained(
                base_model_id, 
                token=hf_token,
                device_map="cpu",  # Load on CPU for merging to save GPU memory
                torch_dtype=model_config['precision'] if model_config['precision'] != 'auto' else torch.float16,
                trust_remote_code=True
            )
            
            # Move the trained model to CPU for merging
            print("🔄 Moving trained model to CPU for merging...")
            model = model.to("cpu")
            
            # Merge the adapter weights into the base model
            print("🔀 Merging adapter weights into base model...")
            merged_model = model.merge_and_unload()
            
            print(f"💾 Uploading merged model to {cleaned_repo}...")
            merged_model.push_to_hub(cleaned_repo, token=hf_token)
            
            # Also upload the tokenizer
            print("💾 Uploading tokenizer...")
            tokenizer.push_to_hub(cleaned_repo, token=hf_token)
            
            print("✅ Full merged model uploaded successfully!")
            print(f"🎯 Repository: https://huggingface.co/{cleaned_repo}")
            
        except Exception as merge_error:
            print(f"❌ Error during model merging: {str(merge_error)}")
            print("🔄 Falling back to adapter-only upload...")
            
            # Fallback: upload adapter weights only
            model.push_to_hub(cleaned_repo, token=hf_token)
            print("⚠️  Uploaded LoRA adapters only (merge failed)")
        
        print("\n🎉 INTELLIGENT TRAINING PIPELINE COMPLETED!")
        print("✅ Training completed successfully!")
        print("📊 Status: COMPLETED")
        print("🔄 Return code: 0")


def main():
    """Main entry point"""
    try:
        # CUDA checks
        if not torch.cuda.is_available():
            print("❌ CUDA not available!")
            exit(1)
        
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()
        
        # Set environment variables for optimization
        os.environ["HF_HUB_VERBOSITY"] = "info"
        os.environ["TRANSFORMERS_VERBOSITY"] = "info"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        os.environ["WANDB_DISABLED"] = "true"
        
        # Initialize and run intelligent trainer
        intelligent_trainer = IntelligentTrainer()
        intelligent_trainer.train()
        
        # Training completed successfully
        print("\n" + "="*60)
        print("🎉 SUCCESS: INTELLIGENT TRAINING COMPLETED!")
        print("="*60)
        print("✅ All training stages completed successfully")
        print("✅ Model uploaded to HuggingFace Hub")
        print("✅ Exit code: 0 (Success)")
        print("="*60)
        
        # Explicitly exit with success code
        sys.exit(0)
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ TRAINING FAILED!")
        print("="*60)
        print(f"❌ Error: {str(e)}")
        print("❌ Exit code: 1 (Failure)")
        print("="*60)
        
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
        # Explicitly exit with failure code
        sys.exit(1)


if __name__ == "__main__":
    main()