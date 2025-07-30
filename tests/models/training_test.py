import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoConfig
)
from peft import LoraConfig, get_peft_model




def main():
    print("--- Initialization Test Script Started ---")
    
    # --- 1. Read configuration from environment variables ---
    hf_token = os.getenv("HF_TOKEN", "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")
    base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    dataset_id = "microsoft/rStar-Coder"
    dataset_subset = "synthetic_sft"

    print(hf_token, base_model_id, dataset_id, dataset_subset)
    
    if not all([hf_token, base_model_id, dataset_id]):
        print("❌ FATAL: Missing one or more required configuration values (hf_token, base_model_id, dataset_id).")
        return

    print(f"✅ Base Model: {base_model_id}")
    print(f"✅ Dataset: {dataset_id}")
    if dataset_subset:
        print(f"✅ Dataset Subset: {dataset_subset}")

    try:
        # --- H100 Compatibility Setup ---
        # Check for CUDA availability and set appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            print(f"🔥 CUDA detected - using GPU acceleration on {torch.cuda.get_device_name(0)}")
            print(f"📊 Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("💻 Using CPU - CUDA not available")
        
        print(f"✅ Device: {device}")

        # --- 2. Load Blueprints (Config and Tokenizer) ---
        print("\n--> Step 1 of 5: Loading model config and tokenizer...")
        config = AutoConfig.from_pretrained(base_model_id, token=hf_token, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("    ✅ Config and Tokenizer loaded successfully.")

        # --- 3. Build "Hollow" Model and Apply LoRA ---
        print("\n--> Step 2 of 5: Building hollow model from config...")
        print("    📋 Model configuration details:")
        print(f"    ├── Model type: {config.model_type}")
        print(f"    ├── Architecture: {config.architectures}")
        print(f"    ├── Hidden size: {config.hidden_size}")
        print(f"    ├── Number of layers: {config.num_hidden_layers}")
        print(f"    ├── Number of attention heads: {config.num_attention_heads}")
        print(f"    ├── Vocabulary size: {config.vocab_size}")
        print(f"    └── Maximum position embeddings: {getattr(config, 'max_position_embeddings', 'Not specified')}")
        
        # Use optimal precision for H100 (supports bfloat16 natively)
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"    🎯 Using torch dtype: {torch_dtype}")
        
        print("    🏗️  Creating model from config (hollow model - no pretrained weights)...")
        model = AutoModelForCausalLM.from_config(
            config, 
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )
        
        print("    📊 Model structure summary:")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    ├── Total parameters: {total_params:,}")
        print(f"    ├── Trainable parameters: {trainable_params:,}")
        print(f"    └── Model size estimate: {total_params * 4 / 1024**3:.2f} GB (FP32)")
        
        print(f"    🚀 Moving model to device: {device}")
        model = model.to(device)
        
        if device == "cuda":
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"    📈 GPU memory after model loading:")
            print(f"    ├── Allocated: {memory_allocated:.2f} GB")
            print(f"    └── Reserved: {memory_reserved:.2f} GB")
        
        print("    ✅ Hollow model built successfully.")
        
        print("\n--> Step 3 of 5: Applying LoRA adapters...")
        # Auto-detect all linear layers for the test
        target_modules = [name.split('.')[-1] for name, module in model.named_modules() if "Linear" in str(type(module))]
        target_modules = list(set(target_modules))
        if "lm_head" in target_modules:
            target_modules.remove("lm_head")

        # Keep LoRA rank small for testing
        lora_config = LoraConfig(r=4, target_modules=target_modules, task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_config)
        print("    ✅ LoRA adapters applied successfully.")
        model.print_trainable_parameters()

        # --- 4. Load and Process a Data Sample ---
        print("\n--> Step 4 of 5: Loading cached data (if available)...")
        # Use cached data if available, don't download full dataset
        print("    📡 Using streaming mode to avoid large downloads...")
        dataset = load_dataset(dataset_id, name=dataset_subset, split="train", streaming=True)
        dataset = dataset.take(5)  # Keep sample size small for testing
        dataset = list(dataset)  # Convert to list for processing
        
        print(f"    📊 Loaded {len(dataset)} samples for testing")
        
        def format_prompt(example):
            if 'instruction' in example and 'response' in example:
                return {"text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"}
            else:
                return {"text": str(example)}
        
        from datasets import Dataset
        dataset = Dataset.from_list(dataset)  # Convert back to Dataset object
        # Keep max_length reasonable for testing
        tokenized_dataset = dataset.map(lambda ex: tokenizer(format_prompt(ex)["text"], truncation=True, max_length=128), batched=False)
        print("    ✅ Cached data sample loaded and processed successfully.")

        # --- 5. Initialize Trainer and Run a Single Step ---
        print("\n--> Step 5 of 5: Initializing Trainer and running minimal training steps...")
        
        # H100-optimized data collator
        def data_collator(batch):
            input_ids = []
            labels = []
            for item in batch:
                ids = item['input_ids']
                if isinstance(ids, list):
                    ids = torch.tensor(ids, dtype=torch.long)
                elif not isinstance(ids, torch.Tensor):
                    ids = torch.tensor(ids, dtype=torch.long)
                input_ids.append(ids)
                labels.append(ids.clone())
            
            # Pad sequences to same length
            max_len = max(len(ids) for ids in input_ids)
            padded_input_ids = []
            padded_labels = []
            
            for ids in input_ids:
                if len(ids) < max_len:
                    padding = torch.full((max_len - len(ids),), tokenizer.pad_token_id, dtype=torch.long)
                    padded_ids = torch.cat([ids, padding])
                else:
                    padded_ids = ids[:max_len]
                padded_input_ids.append(padded_ids)
                padded_labels.append(padded_ids.clone())
            
            return {
                'input_ids': torch.stack(padded_input_ids),
                'labels': torch.stack(padded_labels)
            }
        
        trainer = Trainer(
            model=model,
            train_dataset=tokenized_dataset,
            args=TrainingArguments(
                output_dir="./output",
                per_device_train_batch_size=1,
                max_steps=1, # Run for only 1 step to minimize training time
                save_steps=999999, # Disable saving to speed up test
                logging_steps=1,
                dataloader_num_workers=0, # Single-threaded for simplicity
                fp16=False,  # Use bfloat16 instead for H100
                bf16=device == "cuda",  # Enable bf16 for H100 optimization
                dataloader_pin_memory=device == "cuda",  # Optimize data loading for GPU
            ),
            data_collator=data_collator,
        )
        trainer.train()
        print("    ✅ Trainer initialized and completed minimal training successfully.")

        print("\n🎉 --- TEST RUN COMPLETED SUCCESSFULLY --- 🎉")
        print("Your configuration is valid and the training process can start.")

    except Exception as e:
        print(f"\n❌ --- TEST RUN FAILED --- ❌")
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()