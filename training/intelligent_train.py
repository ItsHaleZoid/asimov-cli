#!/usr/bin/env python3
"""
Intelligent Training System with Dynamic Model and Dataset Classification

This system automatically detects model families and dataset types, then applies
optimal training configurations based on the specific combination detected.

Model Categories:
- Gemma Family: Google's Gemma models (gemma-2-2b-it, etc.)
- Mistral Family: Mistral AI models (mistral-7b-instruct, codestral, etc.) 
- Phi Family: Microsoft Phi models (phi-3.5-mini-instruct, etc.)
- Generic Decoder: Fallback for unknown architectures

Dataset Categories:
- Math/Reasoning: Mathematical word problems, logical reasoning tasks
- Coding: Programming instruction datasets, code generation tasks
- Conversation: Multi-turn dialogue, chat-based interactions
- General Instruction: Standard instruction-response pairs

The system applies optimized hyperparameters, formatting, and training strategies
based on the detected model+dataset combination.
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
    )
    print("✅ Transformers imported")
    sys.stdout.flush()
except Exception as e:
    print(f"❌ Failed to import transformers: {e}")
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
        
        # Gemma Family Detection
        if any(term in model_lower for term in ['gemma', 'google/gemma']):
            return {
                'family': 'gemma',
                'precision': torch.bfloat16,
                'attention': 'sdpa',
                'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", 
                                 "gate_proj", "up_proj", "down_proj"],
                'chat_template': 'gemma',
                'special_tokens': {
                    'bos_token': '<bos>',
                    'eos_token': '<eos>',
                    'pad_token': '<eos>'
                },
                'max_context': 8192,
                'optimization_level': 'high'
            }
        
        # Mistral Family Detection
        elif any(term in model_lower for term in ['mistral', 'codestral']):
            return {
                'family': 'mistral',
                'precision': torch.bfloat16,
                'attention': 'sdpa',
                'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                'chat_template': 'mistral_instruct',
                'special_tokens': {
                    'bos_token': '<s>',
                    'eos_token': '</s>',
                    'pad_token': '</s>'
                },
                'max_context': 32768,
                'optimization_level': 'very_high'
            }
        
        # Phi Family Detection  
        elif any(term in model_lower for term in ['phi', 'microsoft/phi']):
            return {
                'family': 'phi',
                'precision': 'auto',  # Phi models handle dtype automatically
                'attention': 'sdpa',
                'target_modules': ["q_proj", "k_proj", "v_proj", "dense",
                                 "fc1", "fc2"],
                'chat_template': 'phi_instruct',
                'special_tokens': {
                    'bos_token': '<|endoftext|>',
                    'eos_token': '<|endoftext|>',
                    'pad_token': '<|endoftext|>'
                },
                'max_context': 131072,  # Phi-3.5 supports very long context
                'optimization_level': 'ultra_high'
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
        
        # Coding Dataset Detection
        coding_patterns = [
            'opencodeinstruct', 'open-code-instruct', 'code_alpaca',
            'codealpaca', 'wizardcoder', 'wizard-coder', 'magicoder',
            'magic-coder', 'programming', 'code-instruct', 'coding',
            'python', 'javascript', 'java', 'cpp', 'software'
        ]
        
        if any(pattern in dataset_lower for pattern in coding_patterns):
            return {
                'type': 'coding',
                'sample_size': 10,  # Good coverage for coding patterns
                'max_length': 4096,    # Code can be lengthy
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
        if model_family == 'mistral':
            if dataset_type == 'math_reasoning':
                # Enhanced math formatting for Mistral
                formatted_text = f"[INST] {instruction}\n\nPlease solve this step by step, showing your work clearly. [/INST] {response}"
            elif dataset_type == 'coding':
                # Enhanced coding formatting for Mistral
                formatted_text = f"[INST] {instruction}\n\nPlease provide a complete, well-commented solution. [/INST] {response}"
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
            "device_map": "cuda",
            "trust_remote_code": True,
        }
        
        # Apply model family specific settings
        if model_config['precision'] != 'auto':
            model_kwargs["torch_dtype"] = model_config['precision']
        if model_config['attention']:
            model_kwargs["attn_implementation"] = model_config['attention']
        
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
        
        # Configure LoRA rank based on model family and dataset type
        lora_rank = 16  # Default
        if model_config['family'] == 'phi' and dataset_config['type'] == 'math_reasoning':
            lora_rank = 32  # Higher rank for complex reasoning
        elif dataset_config['type'] == 'coding':
            lora_rank = 24  # Medium-high rank for code
        
        print(f"🎯 LoRA rank: {lora_rank}")
        print(f"🎯 Target modules: {target_modules}")
        
        # Detect modules to save
        modules_to_save = []
        for name, module in model.named_modules():
            if any(term in name.lower() for term in ["embed", "lm_head", "head"]):
                modules_to_save.append(name)
        
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.05,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save,
        )
        
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
        
        # Create intelligent training arguments
        print(f"\n🏋️  CONFIGURING TRAINING FOR {dataset_config['optimization_focus'].upper()}")
        
        # Base configuration from dataset type
        batch_size = dataset_config['batch_size']
        grad_accumulation = dataset_config['gradient_accumulation']
        learning_rate = dataset_config['learning_rate']
        epochs = dataset_config['training_epochs']
        
        # Adjust based on model family
        if model_config['family'] == 'phi':
            # Phi models can handle slightly higher learning rates
            learning_rate *= 1.2
        elif model_config['family'] == 'gemma':
            # Gemma benefits from slightly lower learning rates
            learning_rate *= 0.8
        
        # Configure precision based on model family
        use_bf16 = model_config['precision'] == torch.bfloat16
        use_fp16 = model_config['precision'] == torch.float16
        
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
        
        # Initialize trainer
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