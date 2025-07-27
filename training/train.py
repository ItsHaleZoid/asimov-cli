import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

def main():
    print("--- Training Script Started ---")
    
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
  
    if not all([hf_token, base_model_id, dataset_id, lora_model_repo]):
        print("Error: Missing one or more required environment variables.")
        return

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
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype="auto",
                device_map="auto",
                token=hf_token,
                attn_implementation="sdpa",
                resume_download=True,
            )
            print("✅ Model loaded successfully!")
            break
        except Exception as e:
            print(f"❌ Model loading attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("All model loading attempts failed!")
                return
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded successfully!")

    # --- Configure LoRA (PEFT) ---
    print("=== Configuring LoRA Target Modules ===")
    
    lora_target_modules_env = os.getenv("LORA_TARGET_MODULES")
    if lora_target_modules_env:
        target_modules = lora_target_modules_env.split(',')
        print(f"--> Using target modules from environment variable: {target_modules}")
    else:
        print("--> LORA_TARGET_MODULES not set. Detecting all linear layers...")
        all_linear_layer_names = []
        for name, module in model.named_modules():
            if "Linear" in str(type(module)):
                all_linear_layer_names.append(name.split('.')[-1])
        target_modules = list(set(all_linear_layer_names))
        print(f"    Found unique Linear layer names: {target_modules}")
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    
    print(f"\nApplying LoRA with {len(target_modules)} target modules...")
    model = get_peft_model(model, lora_config)
    
    print("\n=== PARAMETER ANALYSIS ===")
    model.print_trainable_parameters()

    # --- Load and Prepare Dataset ---
    dataset = load_dataset(dataset_id, name=dataset_subset, split="train")
    
    # Use a larger portion of the dataset for training
    print(f"Original dataset size: {len(dataset):,} examples")
    sample_size = min(len(dataset), 5000) # Use up to 5000 examples for more comprehensive training
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
    print("\n--- Starting model training... ---")
    trainer.train()
    print("--- Training complete. ---")

    # --- Save and Upload Model ---
    print("\n--- Uploading LoRA adapter to Hugging Face Hub... ---")
    model.push_to_hub(lora_model_repo, token=hf_token)
    print(f"--- Successfully uploaded to {lora_model_repo} ---")

if __name__ == "__main__":
    main()