print("Starting imports...")
import os
print("✅ os imported")
import sys
print("✅ sys imported")
sys.stdout.flush()

try:
    import torch
    print("✅ torch imported")
    sys.stdout.flush()
except Exception as e:
    print(f"❌ Failed to import torch: {e}")
    sys.stdout.flush()
    exit(1)

try:
    from datasets import load_dataset
    print("✅ datasets imported")
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
    )
    print("✅ transformers imported")
    sys.stdout.flush()
except Exception as e:
    print(f"❌ Failed to import transformers: {e}")
    sys.stdout.flush()
    exit(1)

try:
    from peft import LoraConfig, get_peft_model
    print("✅ peft imported")
    sys.stdout.flush()
except Exception as e:
    print(f"❌ Failed to import peft: {e}")
    sys.stdout.flush()
    exit(1)

print("All imports completed successfully!")
sys.stdout.flush()

def main():
    print("=== TRAINING SCRIPT DEBUG START ===")
    print("--- Training Script Started ---")
    print("Python version:", __import__('sys').version)
    print("Current working directory:", os.getcwd())
    print("Files in current directory:", os.listdir('.'))
    
    # Check all environment variables
    print("=== ENVIRONMENT VARIABLES ===")
    env_vars = ['HF_TOKEN', 'BASE_MODEL_ID', 'DATASET_ID', 'LORA_MODEL_REPO', 'LORA_TARGET_MODULES']
    for var in env_vars:
        value = os.getenv(var)
        print(f"{var}: {value if value else 'NOT SET'}")
    print("=== END ENVIRONMENT VARIABLES ===")
    
    import sys
    sys.stdout.flush()  # Force flush output
    
    # Force CUDA usage
    print("Checking CUDA availability...")
    sys.stdout.flush()
    if not torch.cuda.is_available():
        print("❌ CUDA is not available! This script requires CUDA.")
        print("❌ TRAINING FAILED: No CUDA device found")
        sys.stdout.flush()
        exit(1)
    
    print(f"✅ CUDA is available. Using device: {torch.cuda.get_device_name()}")
    sys.stdout.flush()
    torch.cuda.empty_cache()  # Clear CUDA cache
    print("✅ CUDA cache cleared")
    sys.stdout.flush()
    
    # Enable verbose HF Hub logging
    os.environ["HF_HUB_VERBOSITY"] = "info"
    os.environ["TRANSFORMERS_VERBOSITY"] = "info"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # --- Read configuration from environment variables ---
    hf_token = os.getenv("HF_TOKEN")
    base_model_id = os.getenv("BASE_MODEL_ID")
    dataset_id = os.getenv("DATASET_ID")
    dataset_subset = os.getenv("DATASET_SUBSET", None)
    lora_model_repo = os.getenv("LORA_MODEL_REPO")
  
    if not all([hf_token, base_model_id, dataset_id, lora_model_repo]):
        print("❌ Error: Missing one or more required environment variables.")
        print(f"HF_TOKEN: {'✅' if hf_token else '❌'}")
        print(f"BASE_MODEL_ID: {'✅' if base_model_id else '❌'}")
        print(f"DATASET_ID: {'✅' if dataset_id else '❌'}")
        print(f"LORA_MODEL_REPO: {'✅' if lora_model_repo else '❌'}")
        print("❌ TRAINING FAILED: Missing environment variables")
        exit(1)

    print(f"Base Model: {base_model_id}")
    print(f"Dataset: {dataset_id}")
    if dataset_subset:
        print(f"Dataset Subset: {dataset_subset}")
    print(f"Output Repo: {lora_model_repo}")

    # --- Load Model and Tokenizer with retry logic ---
    print("Loading model with network retry logic...")
    import time
    max_retries = 3
    retry_delay = 30
    
    model = None
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} to load model...")
            # Model-specific loading configuration
            model_name_lower = base_model_id.lower()
            
            # Configure model loading parameters based on model family
            model_kwargs = {
                "token": hf_token,
                "device_map": "cuda",
                "resume_download": True,
                "trust_remote_code": True,
            }
            
            # Mistral models prefer BF16 according to documentation
            if 'mistral' in model_name_lower or 'codestral' in model_name_lower:
                model_kwargs["torch_dtype"] = torch.bfloat16
                model_kwargs["attn_implementation"] = "sdpa"  # Mistral supports SDPA
                print(f"--> Loading Mistral model with BF16 precision and SDPA attention")
            else:
                # Default to FP16 for other models
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["attn_implementation"] = "sdpa"
                print(f"--> Loading model with FP16 precision and SDPA attention")
            
            model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
            print("✅ Model loaded successfully!")
            break
        except Exception as e:
            print(f"❌ Model loading attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("❌ All model loading attempts failed!")
                print("❌ TRAINING FAILED: Could not load model")
                exit(1)
    
    # Model-specific tokenizer configuration
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token, trust_remote_code=True)
    
    # Set pad token appropriately for different model families
    if tokenizer.pad_token is None:
        if 'mistral' in model_name_lower or 'codestral' in model_name_lower:
            # Mistral models typically use EOS as pad token
            tokenizer.pad_token = tokenizer.eos_token
            print(f"--> Set Mistral pad token to EOS: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
        else:
            # Default behavior for other models
            tokenizer.pad_token = tokenizer.eos_token
            print(f"--> Set pad token to EOS: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    
    # Verify tokenizer configuration
    print(f"✅ Tokenizer loaded successfully!")
    print(f"--> Vocabulary size: {len(tokenizer)}")
    print(f"--> Special tokens: BOS={tokenizer.bos_token}, EOS={tokenizer.eos_token}, PAD={tokenizer.pad_token}")

    # --- Crucial Step: Resize model embeddings to match tokenizer ---
    # This prevents CUDA errors from out-of-bounds token IDs.
    model.resize_token_embeddings(len(tokenizer))

    # --- Configure LoRA (PEFT) with Model-Specific Target Modules ---
    print("=== Configuring LoRA Target Modules ===")
    
    lora_target_modules_env = os.getenv("LORA_TARGET_MODULES")
    if lora_target_modules_env:
        target_modules = lora_target_modules_env.split(',')
        print(f"--> Using target modules from environment variable: {target_modules}")
    else:
        print("--> LORA_TARGET_MODULES not set. Auto-detecting based on model architecture...")
        
        # Model-specific LoRA target modules based on architecture
        model_name_lower = base_model_id.lower()
        
        if 'mistral' in model_name_lower or 'codestral' in model_name_lower:
            # Mistral architecture: attention and MLP layers
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            print(f"--> Using Mistral-specific target modules: {target_modules}")
        elif 'gemma' in model_name_lower:
            # Gemma architecture: similar to Mistral
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            print(f"--> Using Gemma-specific target modules: {target_modules}")
        elif 'gpt' in model_name_lower:
            # GPT architecture: attention and MLP layers
            target_modules = ["c_attn", "c_proj", "c_fc"]
            print(f"--> Using GPT-specific target modules: {target_modules}")
        else:
            # Generic detection for unknown architectures
            print("--> Unknown architecture, detecting all linear layers...")
            all_linear_layer_names = []
            for name, module in model.named_modules():
                if "Linear" in str(type(module)):
                    all_linear_layer_names.append(name.split('.')[-1])
            target_modules = list(set(all_linear_layer_names))
            print(f"    Found unique Linear layer names: {target_modules}")
    
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
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save,  # CRITICAL: Include embedding and LM head
    )
    
    print(f"\nApplying LoRA with {len(target_modules)} target modules...")
    model = get_peft_model(model, lora_config)
    
    print("\n=== PARAMETER ANALYSIS ===")
    model.print_trainable_parameters()

    # --- Load and Prepare Dataset with Smart Configuration ---
    print("=== DATASET LOADING AND CONFIGURATION ===")
    
    # Detect if this is a coding dataset for Mistral models
    # Handle various possible dataset naming patterns
    dataset_id_lower = dataset_id.lower()
    
    # Check for OpenCodeInstruct and related coding datasets
    is_coding_dataset = any([
        # OpenCodeInstruct variations
        "opencodeinstruct" in dataset_id_lower,
        "open-code-instruct" in dataset_id_lower,
        "open_code_instruct" in dataset_id_lower,
        # Handle organization/dataset format for OpenCodeInstruct
        dataset_id_lower.endswith("/opencodeinstruct"),
        dataset_id_lower.endswith("/open-code-instruct"),
        dataset_id_lower.endswith("/open_code_instruct"),
        # Specific known OpenCodeInstruct repositories
        "nvidia/opencodeinstruct" in dataset_id_lower,
        "sahilchiddarwar/opencodeinstruct" in dataset_id_lower,
        "m-a-p/opencodeinstruct" in dataset_id_lower,
        # Other high-quality coding datasets
        "code_alpaca" in dataset_id_lower,
        "codealpaca" in dataset_id_lower,
        "wizardcoder" in dataset_id_lower,
        "wizard-coder" in dataset_id_lower,
        "evol-codealpaca" in dataset_id_lower,
        "magicoder" in dataset_id_lower,
        "magic-coder" in dataset_id_lower,
        # rStar-Coder dataset
        "rstar-coder" in dataset_id_lower,
        "microsoft/rstar-coder" in dataset_id_lower,
        # Additional coding datasets
        "code-instruct" in dataset_id_lower,
        "coding-instruct" in dataset_id_lower,
        "programming-instruct" in dataset_id_lower,
    ])
    
    is_mistral_model = 'mistral' in base_model_id.lower() or 'codestral' in base_model_id.lower()
    
    # Determine specific dataset type for optimal configuration
    dataset_type = "unknown"
    if any(term in dataset_id_lower for term in ["opencodeinstruct", "open-code-instruct", "open_code_instruct"]):
        dataset_type = "opencodeinstruct"
    elif any(term in dataset_id_lower for term in ["rstar-coder", "microsoft/rstar-coder"]):
        dataset_type = "rstar-coder"
    elif any(term in dataset_id_lower for term in ["code_alpaca", "codealpaca", "evol-codealpaca"]):
        dataset_type = "code-alpaca"
    elif any(term in dataset_id_lower for term in ["wizardcoder", "wizard-coder"]):
        dataset_type = "wizardcoder"
    elif any(term in dataset_id_lower for term in ["magicoder", "magic-coder"]):
        dataset_type = "magicoder"
    elif is_coding_dataset:
        dataset_type = "generic-coding"
    
    print(f"--> Dataset detection: '{dataset_id}' -> Coding dataset: {is_coding_dataset} (type: {dataset_type})")
    print(f"--> Model detection: '{base_model_id}' -> Mistral: {is_mistral_model}")
    
    if is_coding_dataset and is_mistral_model:
        print(f"🎯 DETECTED: {dataset_type.upper()} coding dataset with Mistral model")
        print("--> Applying optimized configuration for coding fine-tuning")
        
        # Load the coding dataset
        dataset = load_dataset(dataset_id, split="train")
        print(f"✅ Loaded {dataset_type} dataset: {len(dataset):,} examples")
        
        # Adjust sample size based on dataset type
        if dataset_type == "opencodeinstruct":
            sample_size = min(len(dataset), 10000)  # Large sample for comprehensive dataset
        elif dataset_type == "rstar-coder":
            sample_size = min(len(dataset), 8000)   # Good sample for competitive programming
        elif dataset_type in ["code-alpaca", "wizardcoder", "magicoder"]:
            sample_size = min(len(dataset), 6000)   # Medium sample for specialized datasets
        else:
            sample_size = min(len(dataset), 5000)   # Standard sample for generic coding
            
        dataset = dataset.select(range(sample_size))
        print(f"--> Using {len(dataset):,} examples for comprehensive coding training")
        print(f"--> Dataset optimization level: {dataset_type}")
        
        # Enable coding optimization
        CODING_OPTIMIZED = True
    else:
        # Standard dataset loading
        dataset = load_dataset(dataset_id, name=dataset_subset, split="train")
        print(f"Original dataset size: {len(dataset):,} examples")
        sample_size = min(len(dataset), 5000)  # Standard sample size
        dataset = dataset.select(range(sample_size))
        print(f"Using a sample of {len(dataset):,} examples for training.")
        CODING_OPTIMIZED = False
    
    def format_prompt(example):
        # Detect model family and apply appropriate formatting
        model_name_lower = base_model_id.lower()
        
        if 'mistral' in model_name_lower or 'codestral' in model_name_lower:
            # Mistral models use [INST] / [/INST] instruction format
            
            # Special handling for coding dataset structure
            if is_coding_dataset:
                # Coding datasets have various field names depending on the source
                # Handle OpenCodeInstruct, rStar-Coder, CodeAlpaca, WizardCoder, etc.
                instruction_text = ""
                response_text = ""
                
                # Try different field combinations for various coding datasets
                if 'instruction' in example and 'response' in example:
                    instruction_text = example['instruction']
                    response_text = example['response']
                elif 'question' in example and 'solution' in example:
                    instruction_text = example['question']
                    response_text = example['solution']
                elif 'problem' in example and 'solution' in example:
                    instruction_text = example['problem']
                    response_text = example['solution']
                elif 'prompt' in example and 'completion' in example:
                    instruction_text = example['prompt']
                    response_text = example['completion']
                elif 'text' in example:
                    # Fallback for single text field
                    return {"text": f"[INST] {example['text']} [/INST]"}
                else:
                    # Last resort: convert to string
                    return {"text": f"[INST] {str(example)} [/INST]"}
                
                # Enhanced formatting for coding tasks
                if 'code' in instruction_text.lower() or 'programming' in instruction_text.lower():
                    formatted_text = f"[INST] {instruction_text}\n\nPlease provide a complete and well-commented solution. [/INST] {response_text}"
                else:
                    formatted_text = f"[INST] {instruction_text} [/INST] {response_text}"
                
                return {"text": formatted_text}
            
            # Standard Mistral formatting for other datasets
            elif 'instruction' in example and 'response' in example:
                return {"text": f"[INST] {example['instruction']} [/INST] {example['response']}"}
            elif 'text' in example:
                # For plain text, wrap in instruction format for better training
                return {"text": f"[INST] {example['text']} [/INST]"}
            else:
                return {"text": f"[INST] {str(example)} [/INST]"}
        else:
            # Default format for other models (Gemma, GPT, etc.)
            if 'instruction' in example and 'response' in example:
                return {"text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"}
            elif 'text' in example:
                return {"text": example['text']}
            else:
                return {"text": str(example)}
    
    dataset = dataset.map(format_prompt)
    
    def tokenize_function(examples):
        # Adjust max_length for coding tasks with coding datasets
        if is_coding_dataset and is_mistral_model:
            # Longer sequences for coding tasks (Mistral supports 32k context)
            max_length = 4096  # Increased for coding examples
            print(f"--> Using extended max_length={max_length} for coding tasks")
        else:
            max_length = 2048  # Standard length
        
        tokens = tokenizer(examples["text"], truncation=True, padding=False, max_length=max_length)
        # Ensure all token IDs are within vocabulary bounds
        # Use the actual embedding layer size, not config (which may be stale)
        actual_vocab_size = model.get_input_embeddings().weight.size(0)
        print(f"DEBUG: Tokenizer vocab size: {len(tokenizer)}, Model vocab size: {actual_vocab_size}")
        
        for key in tokens:
            if key == 'input_ids':
                # Clamp token IDs to valid range
                tokens[key] = [[min(max(token_id, 0), actual_vocab_size-1) for token_id in seq] for seq in tokens[key]]
                
                # Additional safety check
                for seq in tokens[key]:
                    for token_id in seq:
                        if token_id >= actual_vocab_size:
                            print(f"ERROR: Token ID {token_id} >= vocab size {actual_vocab_size}")
                        if token_id < 0:
                            print(f"ERROR: Token ID {token_id} is negative")
        return tokens
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    # CRITICAL: Comprehensive token validation before training
    print("\n=== COMPREHENSIVE TOKEN VALIDATION ===")
    actual_vocab_size = model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)
    config_vocab_size = model.config.vocab_size
    
    print(f"Model config vocab size: {config_vocab_size}")
    print(f"Tokenizer vocab size: {tokenizer_vocab_size}")
    print(f"Actual embedding layer size: {actual_vocab_size}")
    
    # Check for mismatch that could cause issues
    if config_vocab_size != actual_vocab_size:
        print(f"⚠️  WARNING: Config vocab size ({config_vocab_size}) != actual embedding size ({actual_vocab_size})")
    
    # Validate ALL tokens in the dataset
    print("Validating all tokens in the dataset...")
    invalid_token_count = 0
    total_tokens = 0
    max_token_found = -1
    min_token_found = float('inf')
    
    for i, example in enumerate(tokenized_dataset):
        if i < 5:  # Check first 5 examples in detail
            print(f"Example {i} input_ids: {example['input_ids'][:10]}...")  # Show first 10 tokens
        
        for token_id in example['input_ids']:
            total_tokens += 1
            max_token_found = max(max_token_found, token_id)
            min_token_found = min(min_token_found, token_id)
            
            if token_id >= actual_vocab_size or token_id < 0:
                invalid_token_count += 1
                if invalid_token_count <= 10:  # Show first 10 invalid tokens
                    print(f"❌ INVALID TOKEN: {token_id} (valid range: 0-{actual_vocab_size-1})")
    
    print(f"Token validation results:")
    print(f"  Total tokens checked: {total_tokens}")
    print(f"  Invalid tokens found: {invalid_token_count}")
    print(f"  Min token ID: {min_token_found}")
    print(f"  Max token ID: {max_token_found}")
    print(f"  Valid range: 0 to {actual_vocab_size-1}")
    
    if invalid_token_count > 0:
        print(f"❌ CRITICAL: Found {invalid_token_count} invalid tokens. Training will fail!")
        print("❌ TRAINING ABORTED: Invalid token IDs detected")
        exit(1)
    else:
        print("✅ All tokens are within valid range")
    
    # Additional safety: Check special tokens
    print(f"\nSpecial tokens check:")
    print(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else 'None'})")
    print(f"  UNK token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id if hasattr(tokenizer, 'unk_token_id') else 'None'})")
    
    # Check if special token IDs are valid
    special_tokens = [tokenizer.pad_token_id, tokenizer.eos_token_id]
    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
        special_tokens.append(tokenizer.bos_token_id)
    if hasattr(tokenizer, 'unk_token_id') and tokenizer.unk_token_id is not None:
        special_tokens.append(tokenizer.unk_token_id)
    
    for token_id in special_tokens:
        if token_id is not None and (token_id >= actual_vocab_size or token_id < 0):
            print(f"❌ CRITICAL: Special token ID {token_id} is out of bounds!")
            print("❌ TRAINING ABORTED: Invalid special token ID")
            exit(1)
    
    print("✅ All special tokens are within valid range")
    print("=== TOKEN VALIDATION COMPLETE ===\n")

    # --- Define Training Arguments with OpenCodeInstruct Optimization ---
    if CODING_OPTIMIZED:
        print("🔧 APPLYING CODING-OPTIMIZED TRAINING PARAMETERS")
        print("--> Enhanced configuration for OpenCodeInstruct + Mistral")
        
        # Optimized parameters for coding tasks
        training_args = TrainingArguments(
            output_dir="./output",
            per_device_train_batch_size=1,  # Reduced due to longer sequences
            gradient_accumulation_steps=8,  # Increased to maintain effective batch size
            num_train_epochs=2,  # More epochs for coding proficiency
            learning_rate=1e-4,  # Slightly lower LR for coding stability
            warmup_steps=200,  # More warmup for complex coding tasks
            logging_steps=5,  # More frequent logging for coding progress
            save_strategy="steps",
            save_steps=500,  # Save checkpoints more frequently
            eval_strategy="no",  # Disable eval for faster training
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            bf16=True,  # BF16 for Mistral
            fp16=False,
            gradient_checkpointing=True,  # Enable for memory efficiency with longer sequences
            dataloader_drop_last=True,  # Consistent batch sizes
            remove_unused_columns=False,
            max_grad_norm=1.0,  # Gradient clipping for stability
        )
        print(f"--> Batch size: {training_args.per_device_train_batch_size} (effective: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps})")
        print(f"--> Learning rate: {training_args.learning_rate}")
        print(f"--> Epochs: {training_args.num_train_epochs}")
    else:
        # Standard training arguments
        training_args = TrainingArguments(
            output_dir="./output",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=2e-4,
            warmup_steps=100,
            logging_steps=10,
            save_strategy="epoch",
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            bf16=True if 'mistral' in base_model_id.lower() or 'codestral' in base_model_id.lower() else False,
            fp16=False if 'mistral' in base_model_id.lower() or 'codestral' in base_model_id.lower() else True,
            gradient_checkpointing=False, # Disabled to fix grad_fn error
        )

    # --- Create Trainer and Start Training ---
    from transformers import DataCollatorForLanguageModeling
    
    # Custom data collator to ensure all token IDs are valid
    class SafeDataCollatorForCausalLM(DataCollatorForLanguageModeling):
        def __init__(self, tokenizer, mlm=False):
            super().__init__(tokenizer=tokenizer, mlm=mlm)
            self.vocab_size = model.get_input_embeddings().weight.size(0)
            
        def __call__(self, features):
            # Process the batch normally first
            batch = super().__call__(features)
            
            # Ensure all input_ids and labels are within bounds
            if 'input_ids' in batch:
                batch['input_ids'] = torch.clamp(batch['input_ids'], 0, self.vocab_size - 1)
            
            if 'labels' in batch:
                # For labels, we need to preserve -100 (ignore index) but clamp others
                valid_mask = batch['labels'] != -100
                batch['labels'] = torch.where(
                    valid_mask, 
                    torch.clamp(batch['labels'], 0, self.vocab_size - 1),
                    batch['labels']
                )
            
            return batch
    
    data_collator = SafeDataCollatorForCausalLM(tokenizer=tokenizer, mlm=False)
    print(f"✅ Using safe data collator with vocab size bounds: 0-{model.get_input_embeddings().weight.size(0)-1}")
    
    # Additional logging for OpenCodeInstruct optimization
    if CODING_OPTIMIZED:
        print("\n🎯 CODING OPTIMIZATION SUMMARY:")
        print(f"   Dataset: {dataset_type.upper()} ({len(dataset):,} samples)")
        print(f"   Model: Mistral with BF16 precision")
        print(f"   Max sequence length: 4096 tokens")
        print(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        print(f"   Total training steps: ~{len(dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")
        print("   Optimized for: Code generation, debugging, and programming tasks")
        print("   Features: Extended sequences, enhanced prompting, coding-specific hyperparameters\n")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    print("\n--- Starting model training... ---")
    trainer.train()
    print("--- Training complete. ---")

    # --- Save and Upload Model ---
    print("\n--- Uploading LoRA adapter to Hugging Face Hub... ---")
    model.push_to_hub(lora_model_repo, token=hf_token)
    print(f"--- Successfully uploaded to {lora_model_repo} ---")

if __name__ == "__main__":
    try:
        main()
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    except Exception as e:
        print(f"❌ TRAINING FAILED WITH EXCEPTION: {str(e)}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        exit(1)