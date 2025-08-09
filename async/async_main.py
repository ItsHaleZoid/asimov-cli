import asyncio
import yaml
import argparse
import os
import sys
import json
import async_vast_manager as vast
from huggingface_hub import hf_hub_download
from datetime import datetime

# Define LoRA target modules mapping locally to avoid import issues
def get_target_modules(model_type, target_type="all"):
    """Determines LoRA target modules based on model architecture."""
    target_modules_map = {
        "LlamaForCausalLM": {
            "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "mlp": ["gate_proj", "up_proj", "down_proj"]
        },
        "MistralForCausalLM": {
            "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "mlp": ["gate_proj", "up_proj", "down_proj"]
        },
        "Qwen2ForCausalLM": {
            "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "mlp": ["gate_proj", "up_proj", "down_proj"]
        },
        "GPTNeoXForCausalLM": {
            "all": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            "attention": ["query_key_value", "dense"],
            "mlp": ["dense_h_to_4h", "dense_4h_to_h"]
        },
        "GPT2LMHeadModel": {
            "all": ["c_attn", "c_proj", "c_fc"],
            "attention": ["c_attn", "c_proj"],
            "mlp": ["c_fc"]
        }
    }
    
    return target_modules_map.get(model_type, {}).get(target_type, [])

async def get_lora_target_modules_from_config(model_id, hf_token):
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

async def test_target_modules(config):
    """Test function to check what target modules are detected for a model."""
    print("--- Testing LoRA Target Module Detection ---")
    
    lora_target_modules = await get_lora_target_modules_from_config(config['base_model_id'], config['hf_token'])
    print(f"Final target modules string: {lora_target_modules}")
    print("--- Target Module Test Complete ---")

async def run_tuning_job(config, callback=None):
    print("--- Starting Async Tuning Job ---")
    
    docker_image = config.get('docker_image')
    if not docker_image:
        print("Error: 'docker_image' not found in config file. Exiting.")
        sys.exit(1)
    
    print(f"--> Using Docker image: {docker_image}")
    print(f"--> Will upload local training script: training/train.py")
    
    if callback: 
        if asyncio.iscoroutinefunction(callback):
            await callback("initializing", 5, "Initializing training job...")
        else:
            callback("initializing", 5, "Initializing training job...")
    
    # Note: Skipping individual search step, create_instance_with_fallback handles it
    
    # Get LoRA target modules
    if callback: 
        if asyncio.iscoroutinefunction(callback):
            await callback("loading_model", 22, f"Loading model configuration for {config['base_model_id']}...")
        else:
            callback("loading_model", 22, f"Loading model configuration for {config['base_model_id']}...")
    lora_target_modules = await get_lora_target_modules_from_config(config['base_model_id'], config['hf_token'])
    if not lora_target_modules:
        error_msg = "--- Job Failed: Could not determine LoRA target modules. Exiting. ---"
        print(error_msg)
        if callback: 
            if asyncio.iscoroutinefunction(callback):
                await callback("failed", 0, error_msg)
            else:
                callback("failed", 0, error_msg)
        return None

    if callback: 
        if asyncio.iscoroutinefunction(callback):
            await callback("loading_dataset", 24, f"Preparing dataset {config['dataset_id']}...")
        else:
            callback("loading_dataset", 24, f"Preparing dataset {config['dataset_id']}...")
    
    env_vars = {
        "HF_TOKEN": config['hf_token'],
        "BASE_MODEL_ID": config['base_model_id'],
        "DATASET_ID": config['dataset_id'],
        "LORA_MODEL_REPO": config['lora_model_repo'],
        "LORA_TARGET_MODULES": lora_target_modules,
        "DATASET_SUBSET": config['dataset_subset'] if 'dataset_subset' in config and config['dataset_subset'] else "default",
        # Reduce noisy Rust logs from accelerated download clients
        "RUST_LOG": "error",
        # Avoid Xet CAS path which can exhaust file descriptors on some hosts
        "HF_HUB_DISABLE_XET": "1",
        # Ensure hf_transfer is not implicitly enabled if installed
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
    }

    if config.get('max_train_steps'):
        env_vars["MAX_TRAIN_STEPS"] = str(config['max_train_steps'])
    
    # if 'dataset_subset' in config and config['dataset_subset']:
    #     env_vars["DATASET_SUBSET"] = config['dataset_subset']
    print(f"--> Prepared environment variables for the job.")

    max_failovers = int(config.get('max_failovers', 2))
    attempt_index = 0
    last_instance_id = None

    while attempt_index <= max_failovers:
        # Provision a new instance (has built-in fallback + SSH readiness checks)
        new_instance_id = await vast.create_instance_with_fallback(
            config['gpu_name'], docker_image, env_vars, callback=callback
        )
        if not new_instance_id:
            error_msg = "--- Job Failed: Could not create an instance (all fallbacks exhausted). ---"
            print(error_msg)
            if callback:
                if asyncio.iscoroutinefunction(callback):
                    await callback("failed", 0, error_msg)
                else:
                    callback("failed", 0, error_msg)
            return None

        last_instance_id = new_instance_id

        if callback:
            if asyncio.iscoroutinefunction(callback):
                await callback("training", 55, f"Training started on instance {new_instance_id}")
            else:
                callback("training", 55, f"Training started on instance {new_instance_id}")

        # Provide realistic progress updates while training runs
        if callback:
            if asyncio.iscoroutinefunction(callback):
                await callback("training", 60, f"Training started successfully on instance {new_instance_id}")
            else:
                callback("training", 60, f"Training started successfully on instance {new_instance_id}")

        # Wait a bit more for training to actually start
        await asyncio.sleep(5)
        if callback:
            if asyncio.iscoroutinefunction(callback):
                await callback("training", 65, "Training process is running. Monitor with SSH for detailed progress.")
            else:
                callback("training", 65, "Training process is running. Monitor with SSH for detailed progress.")

        # Final status - training is running but we can't monitor completion automatically
        if callback:
            if asyncio.iscoroutinefunction(callback):
                await callback("training", 70, f"Training is running on instance {new_instance_id}. Use 'vastai ssh {new_instance_id} tail -f /app/training.log' to monitor.")
            else:
                callback("training", 70, f"Training is running on instance {new_instance_id}. Use 'vastai ssh {new_instance_id} tail -f /app/training.log' to monitor.")

        print("\n--- Job Started Successfully on Vast.ai ---")
        print("The training script is now running automatically inside the container.")
        print("You can monitor its progress with the command:")
        print(f"vastai ssh {new_instance_id} 'tail -f /app/training.log'")
        print("\nNote: It may take a few minutes for the training.log file to appear as packages are installed first.")

        # NOW WAIT FOR ACTUAL TRAINING COMPLETION (with SSH/instance health detection)
        print("\n--- Monitoring Training Progress ---")
        result = await monitor_remote_training(new_instance_id, callback)

        if result == "completed":
            if callback:
                if asyncio.iscoroutinefunction(callback):
                    await callback("completed", 100, "Training completed successfully!")
                else:
                    callback("completed", 100, "Training completed successfully!")
            print("🎉 TRAINING COMPLETED SUCCESSFULLY!")

            # Destroy the instance immediately to save costs
            print("--> Destroying instance to save costs...")
            try:
                await vast.destroy_instance(new_instance_id)
                print("🔥 Instance destroyed successfully!")
            except Exception as e:
                print(f"⚠️  Warning: Failed to destroy instance {new_instance_id}: {e}")
                print("💡 You may need to destroy it manually to avoid costs.")

            return new_instance_id

        if result == "unhealthy":
            # SSH or instance health degraded - fail over to a new instance
            failover_msg = f"Instance {new_instance_id} became unreachable. Failing over to a new instance (attempt {attempt_index + 1}/{max_failovers})."
            print(f"⚠️  {failover_msg}")
            if callback:
                if asyncio.iscoroutinefunction(callback):
                    await callback("reprovision", 50, failover_msg)
                else:
                    callback("reprovision", 50, failover_msg)

            # Always try to destroy the bad instance to save cost
            try:
                await vast.destroy_instance(new_instance_id)
            except Exception:
                pass

            attempt_index += 1
            continue  # Provision a replacement and retry

        # Any other outcome is a real failure or timeout; clean up and exit
        if callback:
            if asyncio.iscoroutinefunction(callback):
                await callback("failed", 0, "Training failed or timed out")
            else:
                callback("failed", 0, "Training failed or timed out")
        print("❌ TRAINING FAILED OR TIMED OUT")

        print("--> Destroying instance to save costs...")
        try:
            await vast.destroy_instance(new_instance_id)
            print("🔥 Instance destroyed after failure!")
        except Exception as e:
            print(f"⚠️  Warning: Failed to destroy instance {new_instance_id}: {e}")
            print("💡 You may need to destroy it manually to avoid costs.")
        return None

    # If we exit the loop, all failover attempts were exhausted
    exhausted_msg = f"--- Job Failed: Exhausted {max_failovers} failover attempts due to SSH/health issues. ---"
    print(exhausted_msg)
    if callback:
        if asyncio.iscoroutinefunction(callback):
            await callback("failed", 0, exhausted_msg)
        else:
            callback("failed", 0, exhausted_msg)
    if last_instance_id:
        try:
            await vast.destroy_instance(last_instance_id)
        except Exception:
            pass
    return None

async def monitor_remote_training(instance_id, callback=None, timeout_minutes=30, max_consecutive_errors=6):
    """Monitor training progress on remote Vast.ai instance until completion.

    Returns one of: "completed", "failed", "timeout", "unhealthy".
    Unhealthy indicates repeated inability to reach the instance/logs, likely SSH failure.
    """
    print(f"--> Starting remote training monitoring for instance {instance_id}")
    
    start_time = asyncio.get_event_loop().time()
    timeout_seconds = timeout_minutes * 60
    last_log_position = 0
    
    consecutive_errors = 0

    while (asyncio.get_event_loop().time() - start_time) < timeout_seconds:
        try:
            # Get current training logs
            process = await asyncio.create_subprocess_exec(
                "vastai", "execute", str(instance_id), "cat /app/training.log",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logs = stdout.decode('utf-8')
                consecutive_errors = 0  # reset on success
                
                # Check for completion indicators
                completion_patterns = [
                    "🎉 TRAINING COMPLETED SUCCESSFULLY!",
                    "🎉 SUCCESS: INTELLIGENT TRAINING COMPLETED!",
                    "🎉 INTELLIGENT TRAINING PIPELINE COMPLETED!",
                    "✅ All training stages completed successfully"
                ]
                
                failure_patterns = [
                    "❌ TRAINING FAILED",
                    "TRAINING FAILED WITH EXCEPTION"
                ]
                
                # Check for successful completion
                for pattern in completion_patterns:
                    if pattern in logs:
                        print(f"--> Detected successful training completion: {pattern}")
                        return "completed"
                
                # Check for failure
                for pattern in failure_patterns:
                    if pattern in logs:
                        print(f"--> Detected training failure: {pattern}")
                        return "failed"
                
                # Parse progress and update status
                progress = parse_training_progress_from_logs(logs)
                if progress and callback:
                    if asyncio.iscoroutinefunction(callback):
                        await callback("training", progress, f"Training progress: {progress}%")
                    else:
                        callback("training", progress, f"Training progress: {progress}%")
                
                # Check for new log content to show we're still alive
                current_log_length = len(logs)
                if current_log_length > last_log_position:
                    last_log_position = current_log_length
                    print(f"    Training logs updated (size: {current_log_length} chars)")
                
            else:
                print(f"    Failed to get training logs: {stderr.decode()}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print(f"--> Marking instance {instance_id} as unhealthy after {consecutive_errors} consecutive errors")
                    return "unhealthy"
                
        except Exception as e:
            print(f"    Error monitoring training: {e}")
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                print(f"--> Marking instance {instance_id} as unhealthy after {consecutive_errors} consecutive errors")
                return "unhealthy"
        
        # Wait before next check
        await asyncio.sleep(30)  # Check every 30 seconds
    
    print(f"--> Training monitoring timeout after {timeout_minutes} minutes")
    return "timeout"

def parse_training_progress_from_logs(logs):
    """Extract training progress percentage from logs."""
    
    # Look for training progress indicators
    if "=== PARAMETER ANALYSIS ===" in logs:
        if "--- Starting model training... ---" in logs:
            if "--- Training complete. ---" in logs:
                if "--- Uploading LoRA adapter to Hugging Face Hub... ---" in logs:
                    if "--- Successfully uploaded to" in logs:
                        return 100
                    return 95
                return 85
            return 75
        return 65
    elif "Starting imports..." in logs:
        return 50
    
    return None

def file_based_callback(job_id, status, progress, message):
    """Callback function to write status updates to a file that the API server monitors"""
    try:
        import json
        from datetime import datetime
        
        status_file = f"jobs/{job_id}_status.json"
        status_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        # Write atomically by writing to temp file first
        temp_file = f"{status_file}.tmp"
        with open(temp_file, 'w') as f:
            json.dump(status_data, f)
        
        # Atomic rename
        import os
        os.rename(temp_file, status_file)
        
        print(f"[FILE] Status update: {status} ({progress}%) - {message}")
    except Exception as e:
        print(f"[FILE] Failed to write status: {e}")
        # Continue execution even if file write fails

async def main():
    parser = argparse.ArgumentParser(description="CLI for LoRA Fine-tuning on Vast.ai")
    parser.add_argument("--config", required=True, help="Path to the job configuration YAML file")
    parser.add_argument("--test-targets", action="store_true", help="Test LoRA target module detection without running the job")
    parser.add_argument("--job-id", help="Job ID for WebSocket status updates")
    args = parser.parse_args()
# The command to run the script is:
# python async_main.py --config my_job.yaml --test-targets
# python async_main.py --config my_job.yaml

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"--> Loaded configuration from {args.config}")
        
        if args.test_targets:
            await test_target_modules(config)
        else:
            # Create callback function if job_id is provided
            callback = None
            if args.job_id:
                callback = lambda status, progress, message: file_based_callback(args.job_id, status, progress, message)
            
            await run_tuning_job(config, callback)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config}'")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())