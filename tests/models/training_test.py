import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ.setdefault("RUST_LOG", "error")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


def classify_model(model_id: str):
    mid = model_id.lower()
    # Minimal classifier modeled after intelligent_train.py
    if any(t in mid for t in ["gpt-oss", "gpt_oss", "openai/gpt-oss", "openai/gpt_oss"]):
        return {
            "family": "gpt_oss",
            "precision": torch.bfloat16,
            "attention": "eager",
            "device_map": "auto",
            "target_modules": "all-linear",
            "target_parameters": [
                "7.mlp.experts.gate_up_proj",
                "7.mlp.experts.down_proj",
                "15.mlp.experts.gate_up_proj",
                "15.mlp.experts.down_proj",
            ],
            "lora": {"r": 8, "alpha": 16},
        }
    elif any(t in mid for t in ["mistral", "codestral"]):
        return {
            "family": "mistral",
            "precision": torch.bfloat16,
            "attention": "sdpa",
            "device_map": "auto",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
        }
    elif "gemma" in mid:
        return {
            "family": "gemma",
            "precision": torch.bfloat16,
            "attention": "eager",
            "device_map": "auto",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
        }
    else:
        return {
            "family": "generic",
            "precision": torch.float32,
            "attention": None,
            "device_map": "auto",
            "target_modules": None,
            "lora": {"r": 8, "alpha": 16, "dropout": 0.05},
        }


def main():
    print("=== TRAINING TEST (Modeled after intelligent_train) ===")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    base_model_id = os.environ.get("BASE_MODEL_ID", "sshleifer/tiny-gpt2")
    dataset_id = os.environ.get("DATASET_ID", "microsoft/rStar-Coder")
    dataset_subset = os.environ.get("DATASET_SUBSET", "synthetic_sft")

    print(f"Model: {base_model_id}")
    print(f"Dataset: {dataset_id} ({dataset_subset})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    try:
        # Tokenizer and model load (pretrained, like intelligent_train)
        model_cfg = classify_model(base_model_id)
        model_kwargs = {
            "token": hf_token,
            "trust_remote_code": True,
        }
        if model_cfg["precision"] != "auto":
            model_kwargs["torch_dtype"] = model_cfg["precision"]
        if model_cfg.get("attention"):
            model_kwargs["attn_implementation"] = model_cfg["attention"]

        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Resize embeddings before LoRA
        model.resize_token_embeddings(len(tokenizer))

        # Determine target modules
        target_modules = model_cfg["target_modules"]
        if target_modules is None:
            # Auto-detect all linear layer suffixes
            names = []
            for name, module in model.named_modules():
                if "Linear" in str(type(module)):
                    names.append(name.split(".")[-1])
            target_modules = list(set(names))

        # Determine modules_to_save (embeddings + heads)
        modules_to_save = []
        for name, _ in model.named_modules():
            low = name.lower()
            if ("embed" in low and ("token" in low or "word" in low)) or ("lm_head" in low and "embed" not in low):
                modules_to_save.append(name)
        if not modules_to_save:
            modules_to_save = ["embed_tokens", "wte", "token_embedding", "word_embeddings", "lm_head", "head", "classifier"]

        # PEFT version-aware MoE expert targeting like intelligent_train
        lora_rank = model_cfg["lora"]["r"]
        lora_alpha = model_cfg["lora"]["alpha"]
        lora_dropout = model_cfg["lora"].get("dropout", 0.0)

        if model_cfg["family"] == "gpt_oss":
            try:
                from peft import __version__ as peft_version
                from packaging.version import parse as vparse
                supports_target_parameters = vparse(peft_version) >= vparse("0.17.0")
            except Exception:
                supports_target_parameters = False

            if supports_target_parameters and model_cfg.get("target_parameters"):
                lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=[],
                    target_parameters=model_cfg["target_parameters"],
                    task_type="CAUSAL_LM",
                    modules_to_save=modules_to_save,
                )
            else:
                # Fallback: map expert parameter paths to layer indices
                try:
                    layer_indices = sorted({int(p.split(".")[0]) for p in model_cfg.get("target_parameters", [])})
                except Exception:
                    layer_indices = []
                layers_pattern = None
                if hasattr(model, "model") and hasattr(model.model, "layers"):
                    layers_pattern = "model.layers"
                elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                    layers_pattern = "transformer.h"
                lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=["gate_up_proj", "down_proj"],
                    task_type="CAUSAL_LM",
                    modules_to_save=modules_to_save,
                    layers_to_transform=layer_indices if layer_indices else None,
                    layers_pattern=layers_pattern,
                )
        else:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                task_type="CAUSAL_LM",
                modules_to_save=modules_to_save,
            )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Minimal dataset sample (streaming) modeled lightly after intelligent approach
        print("Loading tiny dataset sample (streaming)...")
        sds = load_dataset(dataset_id, name=dataset_subset, split="train", streaming=True)
        sample = list(sds.take(5))
        ds = Dataset.from_list(sample)

        def format_example(ex):
            if "instruction" in ex and "response" in ex:
                return {"text": f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['response']}"}
            return {"text": str(ex)}

        ds = ds.map(lambda ex: format_example(ex))
        tokenized = ds.map(lambda ex: tokenizer(ex["text"], truncation=True, max_length=128), batched=False)

        class SafeCollator:
            def __init__(self, tok):
                self.tok = tok
                self.pad_id = tok.pad_token_id
            def __call__(self, batch):
                input_ids = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
                max_len = max(len(x) for x in input_ids)
                padded = []
                labels = []
                for ids in input_ids:
                    if len(ids) < max_len:
                        pad = torch.full((max_len - len(ids),), self.pad_id, dtype=torch.long)
                        ids = torch.cat([ids, pad])
                    padded.append(ids)
                    labels.append(ids.clone())
                return {"input_ids": torch.stack(padded), "labels": torch.stack(labels)}

        trainer = Trainer(
            model=model,
            train_dataset=tokenized,
            args=TrainingArguments(
                output_dir="./output",
                per_device_train_batch_size=1,
                max_steps=1,
                save_steps=999999,
                logging_steps=1,
                dataloader_num_workers=0,
                fp16=False,
                bf16=(device == "cuda"),
                dataloader_pin_memory=(device == "cuda"),
            ),
            data_collator=SafeCollator(tokenizer),
        )
        trainer.train()
        print("✅ Minimal training step completed")
        print("🎉 TEST SUCCESS")

    except Exception as e:
        print("❌ TEST FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()