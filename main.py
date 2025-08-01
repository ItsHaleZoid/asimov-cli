import argparse
import yaml
import vast_manager
import sys
import json
from huggingface_hub import hf_hub_download
from training.lora_instructions import get_target_modules

def get_lora_target_modules_from_config(model_id, hf_token):
    """Analyzes the model config from Hugging Face to determine LoRA target modules."""
    print(f"--> Analyzing model architecture for {model_id}...")
    try:
        config_path = hf_hub_download(repo_id=model_id, filename="config.json", token=hf_token)
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        architectures = config.get("architectures", [])
        if not architectures:
            raise ValueError("Could not determine model architecture from config.json")
            
        model_type = architectures[0]
        print(f"    Found model architecture: {model_type}")

        target_modules = get_target_modules(model_type, target_type="all")
        
        if target_modules:
            print(f"--> Automatically determined target modules: {target_modules}")
            return ",".join(target_modules)
        else:
            print("--> Warning: Could not determine target modules. Using a default set.")
            return "q_proj,k_proj,v_proj,o_proj"

    except Exception as e:
        print(f"--> Error analyzing model config: {e}")
        print("--> Using a default set of target modules as a fallback.")
        return "q_proj,k_proj,v_proj,o_proj"

def test_target_modules(config):
    """Test function to check what target modules are detected for a model."""
    print("--- Testing LoRA Target Module Detection ---")
    
    lora_target_modules = get_lora_target_modules_from_config(config['base_model_id'], config['hf_token'])
    print(f"Final target modules string: {lora_target_modules}")
    print("--- Target Module Test Complete ---")

def run_tuning_job(config, callback=None):
    print("--- Starting Tuning Job ---")
    
    docker_image = config.get('docker_image')
    if not docker_image:
        print("Error: 'docker_image' not found in config file. Exiting.")
        sys.exit(1)
    
    print(f"--> Using Docker image: {docker_image}")
    print(f"--> Will upload local training script: training/train.py")
    
    if callback: callback("searching_gpu", 15, "Searching for available GPU...")
    
    instance_id, bid_price = vast_manager.search_cheapest_instance(
        gpu_name=config['gpu_name'],
        num_gpus=config['num_gpus']
    )
    if not instance_id:
        error_msg = "--- Job Failed: Could not find a suitable GPU. Exiting. ---"
        print(error_msg)
        if callback: callback("failed", 0, error_msg)
        return None
    
    if callback: callback("found_gpu", 20, f"Found instance {instance_id}")
    
    # Get LoRA target modules
    lora_target_modules = get_lora_target_modules_from_config(config['base_model_id'], config['hf_token'])
    if not lora_target_modules:
        error_msg = "--- Job Failed: Could not determine LoRA target modules. Exiting. ---"
        print(error_msg)
        if callback: callback("failed", 0, error_msg)
        return None

    env_vars = {
        "HF_TOKEN": config['hf_token'],
        "BASE_MODEL_ID": config['base_model_id'],
        "DATASET_ID": config['dataset_id'],
        "LORA_MODEL_REPO": config['lora_model_repo'],
        "LORA_TARGET_MODULES": lora_target_modules,
    }

    if config.get('max_train_steps'):
        env_vars["MAX_TRAIN_STEPS"] = str(config['max_train_steps'])
    
    if 'dataset_subset' in config and config['dataset_subset']:
        env_vars["DATASET_SUBSET"] = config['dataset_subset']
    print(f"--> Prepared environment variables for the job.")

    if callback: callback("creating_instance", 25, "Creating instance...")
    
    new_instance_id = vast_manager.create_instance(instance_id, docker_image, env_vars, bid_price)
    if not new_instance_id:
        error_msg = "--- Job Failed: Could not create the instance. Exiting. ---"
        print(error_msg)
        if callback: callback("failed", 0, error_msg)
        return None
    
    if callback: callback("instance_ready", 30, f"Instance {new_instance_id} created successfully")
        
    print("\n--- Job Started Successfully on Vast.ai ---")
    print("The training script will now run automatically inside the container.")
    print("You can monitor its progress with the command:")
    print(f"vastai ssh {new_instance_id} 'tail -f /app/training.log'")
    
    return new_instance_id


def main():
    parser = argparse.ArgumentParser(description="CLI for LoRA Fine-tuning on Vast.ai")
    parser.add_argument("--config", required=True, help="Path to the job configuration YAML file")
    parser.add_argument("--test-targets", action="store_true", help="Test LoRA target module detection without running the job")
    args = parser.parse_args()
# The command to run the script is:
# python main.py --config my_job.yaml --test-targets
# python main.py --config my_job.yaml

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"--> Loaded configuration from {args.config}")
        
        if args.test_targets:
            test_target_modules(config)
        else:
            run_tuning_job(config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config}'")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()