import os
import sys
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from lora_instructions import get_target_modules, infer_target_modules_from_model

def main():
    # Ensure output is unbuffered for real-time logging
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    # Create a separate training log if it doesn't exist
    log_file = "/app/training_python.log"
    try:
        with open(log_file, "w") as f:
            f.write(f"=== Python Training Script Started at {os.popen('date').read().strip()} ===\n")
        print(f"Created training log at: {log_file}", flush=True)
    except Exception as e:
        print(f"Warning: Could not create log file {log_file}: {e}", flush=True)
    
    print("--- Training Script Started ---", flush=True)
    print(f"Python version: {sys.version}", flush=True)
    print(f"Working directory: {os.getcwd()}", flush=True)
    
    # Enable verbose HF Hub logging
    os.environ["HF_HUB_VERBOSITY"] = "info"
    os.environ["TRANSFORMERS_VERBOSITY"] = "info"
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # --- Read configuration from environment variables ---
    hf_token = os.getenv("HF_TOKEN")
    base_model_id = os.getenv("BASE_MODEL_ID")
    dataset_id = os.getenv("DATASET_ID")
    dataset_subset = os.getenv("DATASET_SUBSET", None)
    lora_model_repo = os.getenv("LORA_MODEL_REPO")
    lora_target_modules_env = os.getenv("LORA_TARGET_MODULES", None)
  
    if not all([hf_token, base_model_id, dataset_id, lora_model_repo]):
        print("Error: Missing one or more required environment variables.", flush=True)
        return

    print(f"Base Model: {base_model_id}", flush=True)
    print(f"Dataset: {dataset_id}", flush=True)
    if dataset_subset:
        print(f"Dataset Subset: {dataset_subset}", flush=True)
    print(f"Output Repo: {lora_model_repo}", flush=True)

    # --- Load Model and Tokenizer with retry logic ---
    print("Loading model with network retry logic...", flush=True)
    import time
    max_retries = 3
    retry_delay = 30
    
    model = None
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} to load model...", flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype="auto",
                device_map="cuda",
                token=hf_token,
                attn_implementation="sdpa",
                resume_download=True,
            )
            print("✅ Model loaded successfully!", flush=True)
            break
        except Exception as e:
            print(f"❌ Model loading attempt {attempt + 1} failed: {e}", flush=True)
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...", flush=True)
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("All model loading attempts failed!", flush=True)
                return
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded successfully!", flush=True)

    # --- Configure LoRA (PEFT) with architecture-aware target detection ---
    print("=== DIAGNOSING MODEL ARCHITECTURE FOR LORA ===")
    print("🔍 Starting target module detection process...")
    
    # Get model architecture name from config
    model_arch = model.config.architectures[0] if hasattr(model.config, 'architectures') and model.config.architectures else None
    print(f"📋 Detected model architecture: {model_arch}")
    
    target_modules = None
    detection_method = "unknown"
    
    # First priority: Use target modules from main.py if available
    if lora_target_modules_env:
        target_modules = lora_target_modules_env.split(",")
        detection_method = "env_variable"
        print(f"✅ Using target modules from environment variable!")
        print(f"🌐 Environment provided {len(target_modules)} target modules: {target_modules}")
    
    # Second priority: Try to get target modules from registry
    elif model_arch:
        print(f"🔎 Looking up '{model_arch}' in architecture registry...")
        target_modules = get_target_modules(model_arch, target_type="all")
        
        if target_modules != ["q_proj", "k_proj", "v_proj", "o_proj"]:  # Not fallback
            print(f"✅ Registry lookup successful!")
            print(f"📚 Found {len(target_modules)} target modules in registry: {target_modules}")
            detection_method = "registry"
        else:
            print(f"⚠️  Architecture not found in registry, received fallback modules")
            target_modules = None  # Force ML inference
    else:
        print("⚠️  No architecture detected in model config")
    
    # Third priority: If registry lookup failed or returned fallback, use ML-based inference
    if not target_modules:
        print("🤖 Using ML-based target module inference from model state dict...")
        print("📊 Analyzing model structure for optimal LoRA targets...")
        inferred_modules = infer_target_modules_from_model(model.state_dict())
        
        # Combine attention and mlp modules
        target_modules = []
        if "attention" in inferred_modules:
            target_modules.extend(inferred_modules["attention"])
            print(f"🎯 Found {len(inferred_modules['attention'])} attention modules: {inferred_modules['attention']}")
        if "mlp" in inferred_modules:
            target_modules.extend(inferred_modules["mlp"])
            print(f"🧠 Found {len(inferred_modules['mlp'])} MLP modules: {inferred_modules['mlp']}")
        
        if "unknown" in inferred_modules:
            print(f"❓ Found {len(inferred_modules['unknown'])} unknown modules: {inferred_modules['unknown']}")
        
        detection_method = "ml_inference"
        print(f"✅ ML inference complete! Total target modules found: {len(target_modules)}")
    
    # Final fallback: original logic if all else fails
    if not target_modules:
        print("⚠️  ML inference failed, falling back to linear layer detection...")
        all_linear_layer_names = []
        for name, module in model.named_modules():
            if "Linear" in str(type(module)):
                all_linear_layer_names.append(name.split('.')[-1])
        target_modules = list(set(all_linear_layer_names))
        detection_method = "linear_fallback"
        print(f"🔧 Fallback detection found {len(target_modules)} target modules: {target_modules}")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"🎯 FINAL TARGET MODULE SELECTION")
    print(f"{'='*60}")
    print(f"📊 Detection method: {detection_method.upper()}")
    print(f"🏗️  Model architecture: {model_arch or 'Unknown'}")
    print(f"📝 Selected target modules ({len(target_modules)} total):")
    for i, module in enumerate(target_modules, 1):
        print(f"   {i:2d}. {module}")
    print(f"{'='*60}")
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    
    print(f"⚙️  Creating LoRA configuration with {len(target_modules)} target modules...")
    print(f"🚀 Applying LoRA adapter to model...")
    model = get_peft_model(model, lora_config)
    
    print("\n=== PARAMETER ANALYSIS ===")
    model.print_trainable_parameters()

    # --- Load and Prepare Dataset ---
    dataset = load_dataset(dataset_id, name=dataset_subset, split="train")
    
    # Reduce dataset size for faster testing
    print(f"Original dataset size: {len(dataset):,} examples")
    sample_size = min(len(dataset), 5000) # Use up to 5000 examples
    dataset = dataset.select(range(sample_size))
    print(f"Using a sample of {len(dataset):,} examples for training.")
    
    def format_prompt(example):
        if 'instruction' in example and 'response' in example:
            return {"text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"}
        elif 'text' in example:
            return {"text": example['text']}
        else:
            return {"text": str(example)}
    
    dataset = dataset.map(format_prompt)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=False, max_length=2048)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # --- Define Training Arguments ---
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
        bf16=True,
        gradient_checkpointing=False, # Disabled to fix grad_fn error
    )

    # --- Create Trainer and Start Training ---
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    print("\n--- Starting model training... ---", flush=True)
    trainer.train()
    print("--- Training complete. ---", flush=True)

    # --- Save and Upload Model ---
    print("\n--- Uploading LoRA adapter to Hugging Face Hub... ---", flush=True)
    model.push_to_hub(lora_model_repo, token=hf_token)
    print(f"--- Successfully uploaded to {lora_model_repo} ---", flush=True)

if __name__ == "__main__":
    main()
