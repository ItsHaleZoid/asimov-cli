import subprocess
import json
import time
import os

def search_cheapest_instance(gpu_name="H100 NVL", num_gpus=1):
    """Searches for the cheapest available interruptible instance on Vast.ai."""
    print(f"--> Searching for the cheapest {num_gpus}x {gpu_name} interruptible instance...")
    
    # Start with the most restrictive search, then progressively relax criteria
    search_strategies = [
        # Strategy 1: Strict requirements
        {
            "criteria": f'num_gpus={num_gpus} gpu_name="{gpu_name}" rentable=true reliability>=0.95 disk_space>=100',
            "description": "strict criteria"
        },
        # Strategy 2: Relax verification and reliability
        {
            "criteria": f'num_gpus={num_gpus} gpu_name="{gpu_name}" rentable=true reliability>=0.8 disk_space>=100',
            "description": "relaxed verification"
        },
        # Strategy 3: Just disk space and basic requirements
        {
            "criteria": f'num_gpus={num_gpus} gpu_name="{gpu_name}" rentable=true disk_space>=100',
            "description": "minimal disk space"
        },
        # Strategy 4: Only GPU and rentable requirements
        {
            "criteria": f'num_gpus={num_gpus} gpu_name="{gpu_name}" rentable=true',
            "description": "basic requirements only"
        },
        # Strategy 5: Target the specific cheap H100 NVL instances from the image
        {
            "criteria": f'num_gpus={num_gpus} gpu_name="H100 NVL" rentable=true dph_total<=0.5',
            "description": "ultra-cheap H100 NVL instances (under $0.50/hr)"
        },
        # Strategy 6: Broader price range for H100 NVL instances
        {
            "criteria": f'num_gpus={num_gpus} gpu_name="H100 NVL" rentable=true dph_total<=1.0',
            "description": "cheap H100 NVL instances (under $1.00/hr)"
        },
        # Strategy 7: Fuzzy GPU name matching (in case exact name doesn't work)
        {
            "criteria": f'num_gpus={num_gpus} gpu_name~="{gpu_name.split()[0]}" rentable=true',
            "description": "fuzzy GPU name matching"
        },
        # Strategy 8: Alternative GPU names that might be cheaper
        {
            "criteria": f'num_gpus={num_gpus} (gpu_name="RTX 4090" OR gpu_name="RTX 3090" OR gpu_name="A100" OR gpu_name="V100") rentable=true',
            "description": "alternative high-end GPUs"
        },
        # Strategy 9: Cast wider net with any modern GPU
        {
            "criteria": f'num_gpus={num_gpus} (gpu_name~="RTX" OR gpu_name~="A100" OR gpu_name~="H100" OR gpu_name~="V100") rentable=true',
            "description": "any modern GPU"
        }
    ]
    
    for strategy in search_strategies:
        print(f"--> Trying {strategy['description']}...")
        
        command = [
            "vastai", "search", "offers",
            strategy["criteria"],
            "--interruptible",
            "--order", "dph_total asc",
            "--raw"
        ]
        
        print(f"    Running command: {' '.join(command)}")
        try:
            result = subprocess.run(
                command,
                capture_output=True, text=True, check=True
            )
            
            if result.stdout.strip():
                instances = json.loads(result.stdout)
                if instances:
                    # Sort by price to get the absolute cheapest
                    instances = sorted(instances, key=lambda x: float(x.get('dph_total', float('inf'))))
                    
                    # Show top 3 cheapest options
                    print(f"--> Found {len(instances)} instances with {strategy['description']}")
                    for i, instance in enumerate(instances[:3]):
                        gpu_info = f"{instance.get('gpu_name', 'Unknown')} x{instance.get('num_gpus', '?')}"
                        reliability = instance.get('reliability', 0)
                        print(f"    Option {i+1}: Instance {instance['id']} - ${instance['dph_total']}/hr - {gpu_info} (reliability: {reliability:.2f})")
                    
                    cheapest = instances[0]
                    recommended_price = float(cheapest.get('min_bid', cheapest['dph_total']))
                    bid_price = recommended_price * 1.80  # Add 50% buffer
                    print(f"--> Selected cheapest: Instance {cheapest['id']}")
                    print(f"    Recommended price: ${recommended_price:.4f}/hr")
                    print(f"    Our bid: ${bid_price:.4f}/hr (recommended + 50%)")
                    return cheapest['id'], bid_price
                    
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"    Search failed: {e}")
            continue
    
    print("--> No suitable instances found with any search strategy.")
    return None, None

def create_instance(instance_id, docker_image, env_vars, bid_price, local_train_script="training/train.py"):
    """Creates a Vast.ai instance and uploads local training script."""
    print(f"--> Attempting to create instance {instance_id} with bid ${bid_price:.4f}/hr...")
    
    existing_instances = get_running_instances()
    if existing_instances:
        print(f"--> Found {len(existing_instances)} existing instance(s). Keeping them running...")
        print(f"    Existing instances: {existing_instances}")
        # Note: Not destroying existing instances to preserve running workloads

    # Create instance - Vast.ai doesn't support -e flags, we'll set env vars via SSH later
    command = [
        "vastai", "create", "instance", str(instance_id),
        "--image", docker_image,
        "--disk", "50",
        "--bid", str(bid_price),
        "--raw"
    ]
             
    print(f"    Running command: {' '.join(command)}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        instance_info = json.loads(result.stdout)
        if instance_info.get("success"):
            new_id = instance_info['new_contract']
            print(f"--> Successfully created instance {new_id}. Waiting for it to start...")
            
            # Wait for instance to be running
            if wait_for_instance_ready(new_id):
                print(f"--> Instance {new_id} is ready!")
                
                # Upload local training script (force overwrite)
                if upload_file_to_instance(new_id, local_train_script, "/app/train.py"):
                    # Upload the intelligent training script (force overwrite)
                    if upload_file_to_instance(new_id, "training/intelligent_train.py", "/app/intelligent_train.py"):
                        # Also upload the lora_instructions.py file (force overwrite)
                        if upload_file_to_instance(new_id, "training/lora_instructions.py", "/app/lora_instructions.py"):
                            # Set environment variables and start training
                            print(f"--> DEBUG: Environment variables being set:")
                            for key, value in env_vars.items():
                                print(f"    {key}: {value[:20] + '...' if len(str(value)) > 20 else value}")
                            
                            env_exports = " && ".join([f"export {key}='{value}'" for key, value in env_vars.items()])
                            print(f"--> DEBUG: Full export command: {env_exports}")
                            
                            start_training_command = f"cd /app && {env_exports} && python intelligent_train.py"
                            print(f"--> DEBUG: Full training command: {start_training_command}")
                            if execute_command_on_instance(new_id, start_training_command):
                                print(f"--> Training started successfully on instance {new_id}!")
                                return new_id
                            else:
                                print(f"--> Failed to start training on instance {new_id}")
                                return None
                        else:
                            print(f"--> Failed to upload lora_instructions.py to instance {new_id}")
                            return None
                    else:
                        print(f"--> Failed to upload intelligent_train.py to instance {new_id}")
                        return None
                else:
                    print(f"--> Failed to upload training script to instance {new_id}")
                    return None
            else:
                print(f"--> Instance {new_id} failed to start properly.")
                return None
        else:
            print(f"--> Failed to create instance. Response: {instance_info}")
            return None
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"--> Error creating instance: {e}")
        if hasattr(e, 'stderr'):
            print(f"    Vast.ai CLI error: {e.stderr}")
        return None

def get_running_instances():
    """Gets a list of currently running instances."""
    try:
        result = subprocess.run(
            ["vastai", "show", "instances", "--raw"],
            capture_output=True, text=True, check=True
        )
        
        if not result.stdout.strip():
            return []
            
        instances = json.loads(result.stdout)
        running_instances = [instance['id'] for instance in instances if instance.get('actual_status') == 'running']
        return running_instances
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"--> Error getting running instances: {e}")
        return []

def get_instance_ssh_info(instance_id):
    """Gets SSH connection info for an instance."""
    try:
        result = subprocess.run(
            ["vastai", "show", "instance", str(instance_id), "--raw"],
            capture_output=True, text=True, check=True
        )
        instance_info = json.loads(result.stdout)
        
        ssh_host = instance_info.get('ssh_host')
        ssh_port = instance_info.get('ssh_port')
        
        if ssh_host and ssh_port:
            return ssh_host, ssh_port
        return None, None
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"--> Error getting SSH info: {e}")
        return None, None

def upload_file_to_instance(instance_id, local_file_path, remote_path="/app/"):
    """Uploads a local file to the instance via SCP, overwriting if exists."""
    ssh_host, ssh_port = get_instance_ssh_info(instance_id)
    if not ssh_host or not ssh_port:
        print(f"--> Could not get SSH info for instance {instance_id}")
        return False
    
    print(f"--> Uploading {local_file_path} to instance {instance_id} (overwriting if exists)...")
    
    try:
        # First ensure the remote directory exists
        mkdir_command = [
            "ssh", "-p", str(ssh_port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            f"root@{ssh_host}",
            f"mkdir -p {os.path.dirname(remote_path) if os.path.dirname(remote_path) else '/app'}"
        ]
        subprocess.run(mkdir_command, capture_output=True, text=True, check=True)
        
        # Force upload with overwrite
        scp_command = [
            "scp", "-P", str(ssh_port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            local_file_path,
            f"root@{ssh_host}:{remote_path}"
        ]
        
        subprocess.run(scp_command, capture_output=True, text=True, check=True)
        print(f"--> Successfully uploaded {os.path.basename(local_file_path)} (overwritten)")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"--> Error uploading file: {e}")
        if e.stderr:
            print(f"    SCP error: {e.stderr}")
        return False

def execute_command_on_instance(instance_id, command):
    """Executes a command on the instance via SSH."""
    ssh_host, ssh_port = get_instance_ssh_info(instance_id)
    if not ssh_host or not ssh_port:
        print(f"--> Could not get SSH info for instance {instance_id}")
        return False
    
    print(f"--> Starting training on instance {instance_id}...")
    
    try:
        # Create a simple startup script and run it
        startup_script = f'''#!/bin/bash
echo "=== Training startup script started ===" >> /app/training.log 2>&1
cd /app
echo "Working directory: $(pwd)" >> /app/training.log 2>&1
echo "Files in /app:" >> /app/training.log 2>&1
ls -la /app >> /app/training.log 2>&1
echo "Environment variables:" >> /app/training.log 2>&1
env | grep -E "(HF_TOKEN|BASE_MODEL|DATASET|LORA)" >> /app/training.log 2>&1
echo "=== Installing additional packages ===" >> /app/training.log 2>&1
python -m pip install protobuf einops sentencepiece accelerate bitsandbytes deepspeed xformers >> /app/training.log 2>&1
export WANDB_DISABLED="true"
echo "=== Starting Python training ===" >> /app/training.log 2>&1
{command} >> /app/training.log 2>&1 &
echo "Training process started with PID: $!" >> /app/training.log 2>&1

'''
        
        ssh_command = [
            "ssh", "-p", str(ssh_port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            f"root@{ssh_host}",
            startup_script
        ]
        
        result = subprocess.run(ssh_command, capture_output=True, text=True, timeout=20)
        
        if result.returncode == 0:
            print(f"--> Training startup script executed successfully")
            print(f"--> Training is now running in background")
        else:
            print(f"--> Error executing startup script: {result.stderr}")
            
        print(f"--> Monitor progress with:")
        print(f"   vastai ssh {instance_id} 'tail -f /app/training.log'")
        print(f"   or check: vastai ssh {instance_id} 'cat /app/training.log'")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"--> Startup script sent (timeout expected)")
        print(f"--> Check logs: vastai ssh {instance_id} 'tail -f /app/training.log'")
        return True
        
    except Exception as e:
        print(f"--> Error executing command: {e}")
        return False

def wait_for_ssh_ready(instance_id, timeout=180):
    """Waits for SSH service to be ready on the instance."""
    print(f"--> Waiting for SSH service to be ready on instance {instance_id}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        ssh_host, ssh_port = get_instance_ssh_info(instance_id)
        if not ssh_host or not ssh_port:
            print(f"    No SSH info available yet...")
            time.sleep(10)
            continue
            
        try:
            # Try a simple SSH connection test
            test_command = [
                "ssh", "-p", str(ssh_port),
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ConnectTimeout=10",
                f"root@{ssh_host}",
                "echo 'SSH ready'"
            ]
            
            result = subprocess.run(test_command, capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                print(f"--> SSH service is ready on instance {instance_id}!")
                return True
            else:
                print(f"    SSH not ready yet (exit code: {result.returncode})")
                
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"    SSH connection test failed: {type(e).__name__}")
            
        time.sleep(15)
    
    print(f"--> Timeout waiting for SSH service on instance {instance_id}")
    return False

def wait_for_instance_ready(instance_id, timeout=300):
    """Waits for instance to be ready and running."""
    print(f"--> Waiting for instance {instance_id} to be ready...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            result = subprocess.run(
                ["vastai", "show", "instance", str(instance_id), "--raw"],
                capture_output=True, text=True, check=True
            )
            instance_info = json.loads(result.stdout)
            
            if instance_info.get('actual_status') == 'running':
                print(f"--> Instance {instance_id} is now running!")
                # Additional wait for SSH to be ready
                if wait_for_ssh_ready(instance_id):
                    return True
                else:
                    return False
                
            print(f"    Current status: {instance_info.get('actual_status', 'unknown')}")
            time.sleep(10)
            
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"    Error checking instance status: {e}")
            time.sleep(10)
    
    print(f"--> Timeout waiting for instance {instance_id} to be ready")
    return False

def destroy_instance(instance_id):
    """Destroys a Vast.ai instance."""
    print(f"--> Destroying instance {instance_id}...")
    try:
        subprocess.run(
            ["vastai", "destroy", "instance", str(instance_id)],
            check=True
        )
        print("--> Instance destroyed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"--> Error destroying instance: {e}")