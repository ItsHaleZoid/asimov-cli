import yaml
import os
from transformers import AutoModelForCausalLM, AutoConfig
from huggingface_hub import login

def check_lora_targets():
    """Check potential LoRA target modules for a model."""
    # Hardcoded model ID and HF token from my_job.yaml
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    print("Logging in to Hugging Face...")
    login("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")

    print(f"Loading model architecture for: {model_id}")
    # Load only the config first
    config = AutoConfig.from_pretrained(model_id)
    # Load the model structure without weights using _fast_init=False and torch_dtype
    model = AutoModelForCausalLM.from_config(config)

    print("\n--- Model Structure ---")
    # Printing the model object gives you a readable view of all the layers
    print(model)

    print("\n--- Finding Potential LoRA Target Modules ---")
    # For LoRA, we are interested in the Linear layers.
    # Let's find all of them.
    linear_layer_names = set()
    for name, module in model.named_modules():
        if "Linear" in str(type(module)):
            # We get the last part of the name (e.g., "q_proj")
            layer_name = name.split('.')[-1]
            linear_layer_names.add(layer_name)

    print("Found the following unique Linear layer names:")
    for name in sorted(list(linear_layer_names)):
        print(f"- {name}")

if __name__ == "__main__":
    check_lora_targets()
