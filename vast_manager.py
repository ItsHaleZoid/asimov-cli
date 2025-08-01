import subprocess
import json
import time
import os
import threading

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
            "criteria": f'num_gpus={num_gpus} gpu_name="{gpu_name}" rentable=true disk_space>=100',
            "description": "basic requirements only"
        },
        # Strategy 5: Target the specific cheap H100 NVL instances from the image
        {
            "criteria": f'num_gpus={num_gpus} gpu_name="H100 NVL" rentable=true dph_total<=0.5 disk_space>=100',
            "description": "ultra-cheap H100 NVL instances (under $0.50/hr)"
        },
        # Strategy 6: Broader price range for H100 NVL instances
        {
            "criteria": f'num_gpus={num_gpus} gpu_name="H100 NVL" rentable=true dph_total<=1.0 disk_space>=100',
            "description": "cheap H100 NVL instances (under $1.00/hr)"
        },
        # Strategy 7: Fuzzy GPU name matching (in case exact name doesn't work)
        {
            "criteria": f'num_gpus={num_gpus} gpu_name~="{gpu_name.split()[0]}" rentable=true',
            "description": "fuzzy GPU name matching"
        },
        # Strategy 8: Alternative GPU names that might be cheaper
        {
            "criteria": f'num_gpus={num_gpus} (gpu_name="RTX 4090" OR gpu_name="RTX 3090" OR gpu_name="A100" OR gpu_name="V100") rentable=true disk_space>=100',
            "description": "alternative high-end GPUs"
        },
        # Strategy 9: Cast wider net with any modern GPU
        {
            "criteria": f'num_gpus={num_gpus} (gpu_name~="RTX" OR gpu_name~="A100" OR gpu_name~="H100" OR gpu_name~="V100") rentable=true disk_space>=100',
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
                    bid_price = recommended_price * 1.50  # Add 50% buffer
                    print(f"--> Selected cheapest: Instance {cheapest['id']}")
                    print(f"    Recommended price: ${recommended_price:.4f}/hr")
                    print(f"    Our bid: ${bid_price:.4f}/hr (recommended + 50%)")
                    return cheapest['id'], bid_price
                    
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"    Search failed: {e}")
            continue
    
    print("--> No suitable instances found with any search strategy.")
    return None, None

def schedule_auto_shutdown(instance_id, timeout_minutes=10):
    """Schedules automatic shutdown of instance after specified timeout."""
    def shutdown_timer():
        print(f"--> Auto-shutdown timer started: Instance {instance_id} will be destroyed in {timeout_minutes} minute(s)")
        time.sleep(timeout_minutes * 60)
        print(f"--> Auto-shutdown triggered for instance {instance_id}")
        destroy_instance(instance_id)
    
    timer_thread = threading.Thread(target=shutdown_timer, daemon=True)
    timer_thread.start()
    return timer_thread

def create_instance(instance_id, docker_image, env_vars, bid_price, local_train_script="training/train.py"):
    """Creates a Vast.ai instance by placing a bid and uploads local training script."""
    print(f"--> Attempting to create instance {instance_id} with bid ${bid_price:.4f}/hr...")
    
    existing_instances = get_running_instances()
    if existing_instances:
        print(f"--> Found {len(existing_instances)} existing instance(s). Destroying them first...")
        for existing_id in existing_instances:
            destroy_instance(existing_id)

    # Create instance with bid - Always start with 100 GB disk space
    command = [
        "vastai", "create", "instance", str(instance_id),
        "--image", docker_image,
        "--disk", "250",
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
            
            # Schedule auto-shutdown after 3 minutes
            schedule_auto_shutdown(new_id, timeout_minutes=3)
            
            # Wait for instance to be running
            if wait_for_instance_ready(new_id):
                print(f"--> Instance {new_id} is ready!")
                
                # Upload local training script (force overwrite)
                if upload_file_to_instance(new_id, local_train_script, "/app/train.py"):
                    # Also upload the lora_instructions.py file (force overwrite)
                    if upload_file_to_instance(new_id, "training/lora_instructions.py", "/app/lora_instructions.py"):
                        # Upload the training_test.py file as well
                        if upload_file_to_instance(new_id, "tests/models/training_test.py", "/app/training_test.py"):
                            # Set environment variables but don't start training automatically
                            env_exports = " && ".join([f"export {key}={value}" for key, value in env_vars.items()])
                            setup_command = f"cd /app && {env_exports} && echo 'Environment variables set successfully'"
                            if execute_command_on_instance(new_id, setup_command):
                                print(f"--> Instance {new_id} is ready for training!")
                                print(f"--> To manually start training, run: vastai ssh {new_id} 'cd /app && python train.py'")
                                print(f"--> WARNING: Instance will auto-shutdown in 3 minutes for testing purposes")
                                return new_id
                            else:
                                print(f"--> Failed to setup environment on instance {new_id}")
                                return None
                        else:
                            print(f"--> Failed to upload training_test.py to instance {new_id}")
                            return None
                    else:
                        print(f"--> Failed to upload lora_instructions.py to instance {new_id}")
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

def create_instance_with_progress(instance_id, docker_image, env_vars, bid_price, local_train_script="training/train.py"):
    """Enhanced create_instance with progress reporting"""
    print(f"--> Creating instance {instance_id} with bid ${bid_price:.4f}/hr...")
    
    # Check and destroy existing instances
    existing_instances = get_running_instances()
    if existing_instances:
        print(f"--> Found {len(existing_instances)} existing instance(s). Destroying them first...")
        for existing_id in existing_instances:
            destroy_instance(existing_id)

    # Create instance with bid
    command = [
        "vastai", "create", "instance", str(instance_id),
        "--image", docker_image,
        "--disk", "250",
        "--bid", str(bid_price),
        "--raw"
    ]
             
    print(f"    Running command: {' '.join(command)}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        instance_info = json.loads(result.stdout)
        if instance_info.get("success"):
            new_id = instance_info['new_contract']
            print(f"--> Instance {new_id} created successfully! Waiting for it to start...")
            
            # Schedule auto-shutdown after 3 minutes
            schedule_auto_shutdown(new_id, timeout_minutes=3)
            
            # Wait for instance to be running with progress updates
            if wait_for_instance_ready_with_progress(new_id):
                print(f"--> Instance {new_id} is ready!")
                
                # Upload files with progress
                print("--> Uploading training scripts...")
                if upload_file_to_instance(new_id, local_train_script, "/app/train.py"):
                    print("    ✓ train.py uploaded")
                    if upload_file_to_instance(new_id, "training/lora_instructions.py", "/app/lora_instructions.py"):
                        print("    ✓ lora_instructions.py uploaded")
                        if upload_file_to_instance(new_id, "tests/models/training_test.py", "/app/training_test.py"):
                            print("    ✓ training_test.py uploaded")
                            
                            # Set environment variables and start training
                            print("--> Starting training process...")
                            env_exports = " && ".join([f"export {key}={value}" for key, value in env_vars.items()])
                            start_command = f"cd /app && {env_exports} && python train.py 2>&1 | tee training.log"
                            
                            if execute_command_on_instance(new_id, start_command):
                                print(f"--> Training started on instance {new_id}")
                                print(f"--> WARNING: Instance will auto-shutdown in 3 minutes for testing purposes")
                                return new_id
            
            print(f"--> Failed to properly setup instance {new_id}")
            return None
        else:
            print(f"--> Failed to create instance. Response: {instance_info}")
            return None
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"--> Error creating instance: {e}")
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

def get_instance_ssh_info(instance_id, retries=5, delay=5):
    """Gets SSH connection info for an instance with retries."""
    for attempt in range(retries):
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
            
            print(f"    SSH info not available yet. Retrying in {delay}s... (Attempt {attempt + 1}/{retries})")
            time.sleep(delay)

        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"--> Error getting SSH info (Attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return None, None
    
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
    
    print(f"--> Executing command on instance {instance_id}...")
    
    try:
        # Create a simple startup script and run it



       



        startup_script = f'''#!/bin/bash
echo "=== Setup script started ===" >> /app/setup.log 2>&1
cd /app
echo "Working directory: $(pwd)" >> /app/setup.log 2>&1
echo "Files in /app:" >> /app/setup.log 2>&1
ls -la /app >> /app/setup.log 2>&1
echo "Environment variables:" >> /app/setup.log 2>&1
env | grep -E "(HF_TOKEN|BASE_MODEL|DATASET|LORA)" >> /app/setup.log 2>&1
echo "=== Installing additional packages ===" >> /app/setup.log 2>&1
python -m pip install --no-cache-dir git+https://github.com/huggingface/transformers.git tiktoken blobfile>> /app/setup.log 2>&1
export WANDB_DISABLED="true"
echo "=== Executing command ===" >> /app/setup.log 2>&1
{command} >> /app/setup.log 2>&1
echo "Command execution completed" >> /app/setup.log 2>&1

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
            print(f"--> Command executed successfully")
        else:
            print(f"--> Error executing command: {result.stderr}")
            
        print(f"--> Check logs with:")
        print(f"   vastai ssh {instance_id} 'cat /app/setup.log'")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"--> Command sent (timeout expected)")
        print(f"--> Check logs: vastai ssh {instance_id} 'cat /app/setup.log'")
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
                    print(f"--> SSH service failed to become ready for instance {instance_id}")
                    return False
                
            print(f"    Current status: {instance_info.get('actual_status', 'unknown')}")
            time.sleep(10)
            
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"    Error checking instance status: {e}")
            time.sleep(10)
    
    print(f"--> Timeout waiting for instance {instance_id} to be ready")
    return False

def wait_for_instance_ready_with_progress(instance_id, timeout=300):
    """Enhanced wait function with progress updates"""
    print(f"--> Waiting for instance {instance_id} to be ready...")
    start_time = time.time()
    last_status = None
    
    while time.time() - start_time < timeout:
        try:
            result = subprocess.run(
                ["vastai", "show", "instance", str(instance_id), "--raw"],
                capture_output=True, text=True, check=True
            )
            instance_info = json.loads(result.stdout)
            current_status = instance_info.get('actual_status', 'unknown')
            
            if current_status != last_status:
                print(f"    Instance status: {current_status}")
                last_status = current_status
            
            if current_status == 'running':
                print(f"--> Instance {instance_id} is now running!")
                # Additional wait for SSH to be ready
                if wait_for_ssh_ready(instance_id):
                    return True
                else:
                    print(f"--> SSH service failed to become ready for instance {instance_id}")
                    return False
                    
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