import argparse
import yaml
import vast_manager
import sys

def run_tuning_job(config):
    print("--- Starting Tuning Job ---")
    
    docker_image = config.get('docker_image')
    if not docker_image:
        print("Error: 'docker_image' not found in config file. Exiting.")
        sys.exit(1)
    
    print(f"--> Using Docker image: {docker_image}")
    print(f"--> Will upload local training script: training/train.py")

    instance_id, bid_price = vast_manager.search_cheapest_instance(
        gpu_name=config['gpu_name'],
        num_gpus=config['num_gpus']
    )
    if not instance_id:
        print("--- Job Failed: Could not find a suitable GPU. Exiting. ---")
        return

    env_vars = {
        "HF_TOKEN": config['hf_token'],
        "BASE_MODEL_ID": config['base_model_id'],
        "DATASET_ID": config['dataset_id'],
        "LORA_MODEL_REPO": config['lora_model_repo']
    }
    
    if 'dataset_subset' in config and config['dataset_subset']:
        env_vars["DATASET_SUBSET"] = config['dataset_subset']
    print(f"--> Prepared environment variables for the job.")

    new_instance_id = vast_manager.create_instance(instance_id, docker_image, env_vars, bid_price=bid_price)
    if not new_instance_id:
        print("--- Job Failed: Could not create the instance. Exiting. ---")
        return
        
    print("\n--- Job Started Successfully on Vast.ai ---")
    print("The training script will now run automatically inside the container.")
    print("You can monitor its progress with the command:")
    print(f"vastai ssh {new_instance_id} 'tail -f /app/training.log'")


def main():
    parser = argparse.ArgumentParser(description="CLI for LoRA Fine-tuning on Vast.ai")
    parser.add_argument("--config", required=True, help="Path to the job configuration YAML file")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"--> Loaded configuration from {args.config}")
        run_tuning_job(config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config}'")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()