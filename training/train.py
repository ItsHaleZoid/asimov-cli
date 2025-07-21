import os
from huggingface_hub import HfApi
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

def main():
    print("--- Training Script Started ---")
    
    # --- 1. Load environment variables from .env.variables file ---
  
    
    # --- 2. Read configuration from environment variables ---
    hf_token = "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"
    base_model_id = "deepseek-ai/DeepSeek-R1"
    dataset_id = "databricks/databricks-dolly-15k"
    lora_model_repo = "HaleZoid/my-deepseek-r1-lora"
  
   

    if not all([hf_token, base_model_id, dataset_id, lora_model_repo]):
        print("Error: Missing one or more required environment variables.")
        # Print which variables are missing for easier debugging
        if not hf_token: print("Missing: HF_TOKEN")
        if not base_model_id: print("Missing: BASE_MODEL_ID")
        if not dataset_id: print("Missing: DATASET_ID")
        if not lora_model_repo: print("Missing: LORA_MODEL_REPO")
        return

    print(f"Base Model: {base_model_id}")
    print(f"HF Token: {hf_token}")
    print(f"Dataset: {dataset_id}")
    print(f"Output Repo: {lora_model_repo}")

    # --- 3. Load Model and Tokenizer ---
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quantization_config,
        device_map="auto",
        token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    # --- 4. Configure LoRA (PEFT) ---
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # --- 5. Load and Prepare Dataset ---
    dataset = load_dataset(dataset_id, split="train")
    def format_prompt(example):
        return {"text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"}
    
    dataset = dataset.map(format_prompt)

    # --- 6. Define Training Arguments ---
    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=4,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch",
    )

    # --- 7. Create Trainer and Start Training ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=lambda data: {'input_ids': tokenizer([x['text'] for x in data], padding=True, return_tensors="pt").input_ids,
                                    'labels': tokenizer([x['text'] for x in data], padding=True, return_tensors="pt").input_ids},
    )
    print("\n--- Starting model training... ---")
    trainer.train()
    print("--- Training complete. ---")

    # --- 8. Save and Upload Model ---
    print("\n--- Uploading LoRA adapter to Hugging Face Hub... ---")
    model.push_to_hub(lora_model_repo, token=hf_token)
    print(f"--- Successfully uploaded to {lora_model_repo} ---")

if __name__ == "__main__":
    main()

