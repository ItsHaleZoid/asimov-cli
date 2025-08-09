import asyncio
import json
import os
import threading
import time
from typing import List, Dict, Any, Tuple, Optional

async def search_multiple_instances(gpu_name: str = "H100 NVL", num_gpus: int = 1, callback=None) -> List[Tuple[int, float]]:
    """Asynchronously searches for multiple available instances on Vast.ai, returns list of (instance_id, bid_price)."""
    print(f"--> Searching for multiple {num_gpus}x {gpu_name} interruptible instances...")
    if callback: 
        if asyncio.iscoroutinefunction(callback):
            await callback("searching_gpu", 15, f"Searching for {num_gpus}x {gpu_name} instances...")
        else:
            callback("searching_gpu", 15, f"Searching for {num_gpus}x {gpu_name} instances...")
    
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
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0 and stdout.strip():
                instances = json.loads(stdout)
                if instances:
                    # Sort by price to get the absolute cheapest
                    instances = sorted(instances, key=lambda x: float(x.get('dph_total', float('inf'))))
                    
                    # Show top options found
                    print(f"--> Found {len(instances)} instances with {strategy['description']}")
                    for i, instance in enumerate(instances[:5]):  # Show top 5
                        gpu_info = f"{instance.get('gpu_name', 'Unknown')} x{instance.get('num_gpus', '?')}"
                        reliability = instance.get('reliability', 0)
                        print(f"    Option {i+1}: Instance {instance['id']} - ${instance['dph_total']}/hr - {gpu_info} (reliability: {reliability:.2f})")
                    
                    # Prepare the list of (instance_id, bid_price) tuples
                    instance_list = []
                    for instance in instances[:5]:  # Return top 5 options
                        recommended_price = float(instance.get('min_bid', instance['dph_total']))
                        bid_price = recommended_price * 1.20  # Add 20% buffer
                        instance_list.append((instance['id'], bid_price))
                    
                    if callback: 
                        if asyncio.iscoroutinefunction(callback):
                            await callback("found_gpu", 20, f"Found {len(instance_list)} instances to try")
                        else:
                            callback("found_gpu", 20, f"Found {len(instance_list)} instances to try")
                    
                    return instance_list
                    
        except (json.JSONDecodeError, Exception) as e:
            print(f"    Search failed: {e}")
            continue
    
    print("--> No suitable instances found with any search strategy.")
    return []

async def search_cheapest_instance(gpu_name: str = "H100 NVL", num_gpus: int = 1, callback=None) -> Tuple[Optional[int], Optional[float]]:
    """Asynchronously searches for the cheapest available interruptible instance on Vast.ai."""
    print(f"--> Searching for the cheapest {num_gpus}x {gpu_name} interruptible instance...")
    if callback: 
        if asyncio.iscoroutinefunction(callback):
            await callback("searching_gpu", 15, f"Searching for {num_gpus}x {gpu_name} instances...")
        else:
            callback("searching_gpu", 15, f"Searching for {num_gpus}x {gpu_name} instances...")
    
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
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0 and stdout.strip():
                instances = json.loads(stdout)
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
                    bid_price = recommended_price * 0.2  # Add 80% buffer
                    print(f"--> Selected cheapest: Instance {cheapest['id']}")
                    print(f"    Recommended price: ${recommended_price:.4f}/hr")
                    print(f"    Our bid: ${bid_price:.4f}/hr (recommended + 80%)")
                    if callback: 
                        if asyncio.iscoroutinefunction(callback):
                            await callback("found_gpu", 20, f"Found instance {cheapest['id']} at ${bid_price:.4f}/hr")
                        else:
                            callback("found_gpu", 20, f"Found instance {cheapest['id']} at ${bid_price:.4f}/hr")
                    return cheapest['id'], bid_price
                    
        except (json.JSONDecodeError, Exception) as e:
            print(f"    Search failed: {e}")
            continue
    
    print("--> No suitable instances found with any search strategy.")
    return None, None

async def search_cheapest_on_demand_instance(gpu_name: str = "H100 NVL", num_gpus: int = 1, callback=None) -> Tuple[Optional[int], Optional[float]]:
    """Asynchronously searches for the cheapest available on-demand instance on Vast.ai."""
    print(f"--> Searching for the cheapest {num_gpus}x {gpu_name} on-demand instance...")
    if callback: 
        if asyncio.iscoroutinefunction(callback):
            await callback("searching_gpu", 15, f"Searching for {num_gpus}x {gpu_name} on-demand instances...")
        else:
            callback("searching_gpu", 15, f"Searching for {num_gpus}x {gpu_name} on-demand instances...")
    
    # Search strategies for on-demand instances (without --interruptible flag)
    search_strategies = [
        # Strategy 1: Strict requirements
        {
            "criteria": f'num_gpus={num_gpus} gpu_name="{gpu_name}" rentable=true reliability>=0.95 disk_space>=100',
            "description": "strict criteria (on-demand)"
        },
        # Strategy 2: Relax verification and reliability
        {
            "criteria": f'num_gpus={num_gpus} gpu_name="{gpu_name}" rentable=true reliability>=0.8 disk_space>=100',
            "description": "relaxed verification (on-demand)"
        },
        # Strategy 3: Just disk space and basic requirements
        {
            "criteria": f'num_gpus={num_gpus} gpu_name="{gpu_name}" rentable=true disk_space>=100',
            "description": "minimal disk space (on-demand)"
        },
        # Strategy 4: Only GPU and rentable requirements
        {
            "criteria": f'num_gpus={num_gpus} gpu_name="{gpu_name}" rentable=true',
            "description": "basic requirements only (on-demand)"
        },
        # Strategy 5: Alternative GPU names that might be cheaper
        {
            "criteria": f'num_gpus={num_gpus} (gpu_name="RTX 4090" OR gpu_name="RTX 3090" OR gpu_name="A100" OR gpu_name="V100") rentable=true',
            "description": "alternative high-end GPUs (on-demand)"
        },
        # Strategy 6: Cast wider net with any modern GPU
        {
            "criteria": f'num_gpus={num_gpus} (gpu_name~="RTX" OR gpu_name~="A100" OR gpu_name~="H100" OR gpu_name~="V100") rentable=true',
            "description": "any modern GPU (on-demand)"
        }
    ]
    
    for strategy in search_strategies:
        print(f"--> Trying {strategy['description']}...")
        
        command = [
            "vastai", "search", "offers",
            strategy["criteria"],
            # NOTE: No --interruptible flag for on-demand instances
            "--order", "dph_total asc",
            "--raw"
        ]
        
        print(f"    Running command: {' '.join(command)}")
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0 and stdout.strip():
                instances = json.loads(stdout)
                if instances:
                    # Sort by price to get the absolute cheapest
                    instances = sorted(instances, key=lambda x: float(x.get('dph_total', float('inf'))))
                    
                    # Show top 3 cheapest options
                    print(f"--> Found {len(instances)} on-demand instances with {strategy['description']}")
                    for i, instance in enumerate(instances[:3]):
                        gpu_info = f"{instance.get('gpu_name', 'Unknown')} x{instance.get('num_gpus', '?')}"
                        reliability = instance.get('reliability', 0)
                        print(f"    Option {i+1}: Instance {instance['id']} - ${instance['dph_total']}/hr - {gpu_info} (reliability: {reliability:.2f})")
                    
                    cheapest = instances[0]
                    # For on-demand instances, use the exact price (no bidding)
                    price = float(cheapest['dph_total'])
                    print(f"--> Selected cheapest on-demand: Instance {cheapest['id']}")
                    print(f"    On-demand price: ${price:.4f}/hr")
                    if callback: 
                        if asyncio.iscoroutinefunction(callback):
                            await callback("found_gpu", 20, f"Found on-demand instance {cheapest['id']} at ${price:.4f}/hr")
                        else:
                            callback("found_gpu", 20, f"Found on-demand instance {cheapest['id']} at ${price:.4f}/hr")
                    return cheapest['id'], price
                    
        except (json.JSONDecodeError, Exception) as e:
            print(f"    Search failed: {e}")
            continue
    
    print("--> No suitable on-demand instances found with any search strategy.")
    return None, None

def schedule_auto_shutdown(instance_id: int, timeout_minutes: int = 100) -> threading.Thread:
    """Schedules automatic shutdown of instance after specified timeout."""
    def shutdown_timer():
        print(f"--> Auto-shutdown timer started: Instance {instance_id} will be destroyed in {timeout_minutes} minute(s)")
        time.sleep(timeout_minutes * 60)
        print(f"--> Auto-shutdown triggered for instance {instance_id}")
        # Run the async destroy_instance in a new event loop since we're in a thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(destroy_instance(instance_id))
        loop.close()
    
    timer_thread = threading.Thread(target=shutdown_timer, daemon=True)
    timer_thread.start()
    return timer_thread

async def create_instance_with_fallback(gpu_name: str, docker_image: str, env_vars: Dict[str, Any], local_train_script: str = "training/train.py", callback=None) -> Optional[int]:
    """Tries to create an instance with fallback to other instances, including on-demand if interruptible fails."""
    print(f"--> Starting instance creation with fallback for {gpu_name}...")
    
    # Phase 1: Try interruptible instances first (cheaper)
    print("--> Phase 1: Attempting interruptible instances...")
    instances = await search_multiple_instances(gpu_name=gpu_name, callback=callback)
    if instances:
        # Try up to 3 interruptible instances
        max_attempts = min(3, len(instances))
        for attempt, (instance_id, bid_price) in enumerate(instances[:max_attempts]):
            print(f"--> Interruptible attempt {attempt + 1}/{max_attempts}: Trying instance {instance_id}")
            
            result = await create_instance(instance_id, docker_image, env_vars, bid_price, local_train_script, callback)
            if result:
                print(f"--> Success with interruptible instance {result}")
                return result
                
            if attempt < max_attempts - 1:
                print(f"--> Instance {instance_id} failed, trying next interruptible option...")
        
        print(f"--> All {max_attempts} interruptible attempts failed")
    else:
        print("--> No interruptible instances found")
    
    # Phase 2: Fallback to on-demand instances if interruptible failed
    print("--> Phase 2: Falling back to on-demand instances...")
    if callback: 
        if asyncio.iscoroutinefunction(callback):
            await callback("searching_gpu", 15, "Interruptible instances failed, trying on-demand...")
        else:
            callback("searching_gpu", 15, "Interruptible instances failed, trying on-demand...")
    
    instance_id, price = await search_cheapest_on_demand_instance(gpu_name=gpu_name, callback=callback)
    if instance_id:
        print(f"--> Trying on-demand instance {instance_id} at ${price:.4f}/hr")
        result = await create_instance(instance_id, docker_image, env_vars, price, local_train_script, callback)
        if result:
            print(f"--> Success with on-demand instance {result}")
            return result
        else:
            print(f"--> On-demand instance {instance_id} also failed")
    else:
        print("--> No on-demand instances found")
    
    print("--> All fallback attempts failed (both interruptible and on-demand)")
    return None

async def create_instance(instance_id: int, docker_image: str, env_vars: Dict[str, Any], bid_price: float, local_train_script: str = "training/train.py", callback=None) -> Optional[int]:
    """Creates a Vast.ai instance and uploads local training script."""
    print(f"--> Attempting to create instance {instance_id} with bid ${bid_price:.4f}/hr...")
    if callback: 
        if asyncio.iscoroutinefunction(callback):
            await callback("creating_instance", 25, f"Creating instance {instance_id} with bid ${bid_price:.4f}/hr...")
        else:
            callback("creating_instance", 25, f"Creating instance {instance_id} with bid ${bid_price:.4f}/hr...")
    
    existing_instances = await get_running_instances()
    if existing_instances:
        print(f"--> Found {len(existing_instances)} existing instance(s). Keeping them running...")
        print(f"    Existing instances: {existing_instances}")
        # Note: Not destroying existing instances to preserve running workloads

    # Create instance - Vast.ai doesn't support -e flags, we'll set env vars via SSH later
    command = [
        "vastai", "create", "instance", str(instance_id),
        "--image", docker_image,
        "--disk", "250",
        "--bid", str(bid_price),
        "--raw"
    ]
             
    print(f"    Running command: {' '.join(command)}")
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            if not stdout.strip():
                print(f"--> Error: Empty response from vastai create command")
                print(f"    stderr: {stderr.decode() if stderr else 'No error output'}")
                return None
            
            try:
                instance_info = json.loads(stdout)
            except json.JSONDecodeError as je:
                print(f"--> Error parsing JSON response: {je}")
                print(f"    Raw stdout: {stdout.decode()}")
                print(f"    stderr: {stderr.decode() if stderr else 'No error output'}")
                return None
                
            if instance_info.get("success"):
                new_id = instance_info['new_contract']
                print(f"--> Successfully created instance {new_id}. Waiting for it to start...")
                if callback: 
                    if asyncio.iscoroutinefunction(callback):
                        await callback("instance_ready", 30, f"Instance {new_id} created, waiting for startup...")
                    else:
                        callback("instance_ready", 30, f"Instance {new_id} created, waiting for startup...")
                
                # Auto-shutdown disabled to preserve running workloads
                # schedule_auto_shutdown(new_id, timeout_minutes=10)
                
                # Wait for instance to be running
                if await wait_for_instance_ready_with_progress(new_id, callback=callback):
                    print(f"--> Instance {new_id} is ready!")
                    if callback: 
                        if asyncio.iscoroutinefunction(callback):
                            await callback("instance_ready", 45, f"Instance {new_id} is fully ready for training")
                        else:
                            callback("instance_ready", 45, f"Instance {new_id} is fully ready for training")
                    
                    # Upload local training script (force overwrite)
                    if callback: 
                        if asyncio.iscoroutinefunction(callback):
                            await callback("uploading_script", 50, "Uploading training scripts to instance...")
                        else:
                            callback("uploading_script", 50, "Uploading training scripts to instance...")
                    if await upload_file_to_instance(new_id, local_train_script, "/app/train.py"):
                        # Upload the intelligent training script (force overwrite)
                        if await upload_file_to_instance(new_id, "training/intelligent_train.py", "/app/intelligent_train.py"):
                            # Also upload the lora_instructions.py file (force overwrite)
                            if await upload_file_to_instance(new_id, "training/lora_instructions.py", "/app/lora_instructions.py"):
                                # Set environment variables and start training
                                env_exports = " && ".join([f"export {key}='{value}'" for key, value in env_vars.items()])
                                # Raise ulimit for open files to mitigate 'Too many open files' during large downloads
                                ulimit_cmd = "ulimit -n 65535 || true"
                                start_training_command = f"cd /app && {ulimit_cmd} && {env_exports} && python intelligent_train.py"
                                if await execute_command_on_instance(new_id, start_training_command):
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
                # Check for specific failure reasons
                if 'error' in instance_info:
                    print(f"    Error details: {instance_info['error']}")
                if 'new_contract' in instance_info:
                    print(f"    Contract ID: {instance_info['new_contract']}")
                    
                # Possible reasons for failure:
                print("    Possible reasons:")
                print("    - Instance already rented by someone else")
                print("    - Bid price too low")
                print("    - Host requirements not met")
                print("    - Vast.ai API rate limiting")
                return None
        else:
            print(f"--> Error creating instance: {stderr.decode()}")
            return None
    except Exception as e:
        print(f"--> Error creating instance: {e}")
        return None

async def get_running_instances(callback=None) -> List[int]:
    """Gets a list of currently running instances."""
    if callback: 
        if asyncio.iscoroutinefunction(callback):
            await callback("instance_ready", 35, "Checking for running instances...")
        else:
            callback("instance_ready", 35, "Checking for running instances...")
    try:
        process = await asyncio.create_subprocess_exec(
            "vastai", "show", "instances", "--raw",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0 and stdout.strip():
            instances = json.loads(stdout)
            running_instances = [instance['id'] for instance in instances if instance.get('actual_status') == 'running']
            return running_instances
    except Exception as e:
        print(f"--> Error getting running instances: {e}")
    return []

async def get_instance_ssh_info(instance_id, retries=5, delay=5):
    """Gets SSH connection info for an instance with retries."""
    for attempt in range(retries):
        try:
            process = await asyncio.create_subprocess_exec(
                "vastai", "show", "instance", str(instance_id), "--raw",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0 and stdout:
                instance_info = json.loads(stdout)
                
                ssh_host = instance_info.get('ssh_host')
                ssh_port = instance_info.get('ssh_port')
                
                if ssh_host and ssh_port:
                    return ssh_host, ssh_port
            
            print(f"    SSH info not available yet. Retrying in {delay}s... (Attempt {attempt + 1}/{retries})")
            await asyncio.sleep(delay)

        except Exception as e:
            print(f"--> Error getting SSH info (Attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay)
            else:
                return None, None
    
    return None, None

async def upload_file_to_instance(instance_id, local_file_path, remote_path="/app/"):
    """Uploads a local file to the instance via SCP, overwriting if exists."""
    ssh_host, ssh_port = await get_instance_ssh_info(instance_id)
    if not ssh_host or not ssh_port:
        print(f"--> Could not get SSH info for instance {instance_id}")
        return False
    
    print(f"--> Uploading {local_file_path} to instance {instance_id} (overwriting if exists)...")
    
    try:
        # First ensure the remote directory exists
        mkdir_process = await asyncio.create_subprocess_exec(
            "ssh", "-p", str(ssh_port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            f"root@{ssh_host}",
            f"mkdir -p {os.path.dirname(remote_path) if os.path.dirname(remote_path) else '/app'}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await mkdir_process.communicate()
        
        # Force upload with overwrite
        scp_process = await asyncio.create_subprocess_exec(
            "scp", "-P", str(ssh_port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            local_file_path,
            f"root@{ssh_host}:{remote_path}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await scp_process.communicate()
        
        if scp_process.returncode == 0:
            print(f"--> Successfully uploaded {os.path.basename(local_file_path)} (overwritten)")
            return True
        else:
            print(f"--> Error uploading file: {stderr.decode()}")
            return False
        
    except Exception as e:
        print(f"--> Error uploading file: {e}")
        return False

async def execute_command_on_instance(instance_id, command):
    """Executes a command on the instance via SSH."""
    ssh_host, ssh_port = await get_instance_ssh_info(instance_id)
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
env | grep -E "(HF_TOKEN|BASE_MODEL|DATASET|LORA|DATASET_SUBSET)" >> /app/training.log 2>&1
echo "=== Installing additional packages ===" >> /app/training.log 2>&1
python -m pip install --upgrade transformers protobuf einops sentencepiece trl peft accelerate bitsandbytes deepspeed xformers  >> /app/training.log 2>&1
export WANDB_DISABLED="true"
echo "=== Starting Python training ===" >> /app/training.log 2>&1
{command} >> /app/training.log 2>&1 &
echo "Training process started with PID: $!" >> /app/training.log 2>&1

'''
        
        ssh_process = await asyncio.create_subprocess_exec(
            "ssh", "-p", str(ssh_port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            f"root@{ssh_host}",
            startup_script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(ssh_process.communicate(), timeout=20)
            
            if ssh_process.returncode == 0:
                print(f"--> Training startup script executed successfully")
                print(f"--> Training is now running in background")
            else:
                print(f"--> Error executing startup script: {stderr.decode()}")
                
            print(f"--> Monitor progress with:")
            print(f"   vastai ssh {instance_id} 'tail -f /app/training.log'")
            print(f"   or check: vastai ssh {instance_id} 'cat /app/training.log'")
            return True
            
        except asyncio.TimeoutError:
            print(f"--> Startup script sent (timeout expected)")
            print(f"--> Check logs: vastai ssh {instance_id} 'tail -f /app/training.log'")
            ssh_process.terminate()
            return True
        
    except Exception as e:
        print(f"--> Error executing command: {e}")
        return False

async def wait_for_ssh_ready(instance_id, timeout=300):
    """Waits for SSH service to be ready on the instance with robust detection."""
    print(f"--> Waiting for SSH service to be ready on instance {instance_id}...")
    start_time = asyncio.get_event_loop().time()
    
    # Progressive retry delays: start fast, slow down for stubborn instances
    retry_delays = [5, 5, 10, 10, 15, 15, 20, 20, 25, 30]
    retry_index = 0
    
    while (asyncio.get_event_loop().time() - start_time) < timeout:
        ssh_host, ssh_port = await get_instance_ssh_info(instance_id, retries=3, delay=3)
        if not ssh_host or not ssh_port:
            print(f"    No SSH info available yet...")
            await asyncio.sleep(10)
            continue
            
        # Try multiple connection methods for reliability
        ssh_methods = [
            # Method 1: Quick connection test
            {
                "timeout": 20,
                "connect_timeout": 15,
                "description": "quick test"
            },
            # Method 2: More patient connection
            {
                "timeout": 30, 
                "connect_timeout": 25,
                "description": "patient test"
            },
            # Method 3: Very patient for slow instances
            {
                "timeout": 45,
                "connect_timeout": 40, 
                "description": "ultra-patient test"
            }
        ]
        
        for method in ssh_methods:
            try:
                print(f"    Trying SSH {method['description']} (timeout={method['timeout']}s)...")
                
                test_process = await asyncio.create_subprocess_exec(
                    "ssh", "-p", str(ssh_port),
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "UserKnownHostsFile=/dev/null",
                    "-o", f"ConnectTimeout={method['connect_timeout']}",
                    "-o", "ServerAliveInterval=5",
                    "-o", "ServerAliveCountMax=3",
                    f"root@{ssh_host}",
                    "echo 'SSH ready' && whoami && pwd",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        test_process.communicate(), 
                        timeout=method['timeout']
                    )
                    
                    if test_process.returncode == 0:
                        print(f"--> SSH service is ready on instance {instance_id}! ✅")
                        print(f"    SSH response: {stdout.decode().strip()}")
                        return True
                    else:
                        print(f"    SSH {method['description']} failed (exit code: {test_process.returncode})")
                        stderr_msg = stderr.decode().strip()
                        if stderr_msg:
                            print(f"    Error: {stderr_msg}")
                            
                except asyncio.TimeoutError:
                    print(f"    SSH {method['description']} timeout")
                    test_process.terminate()
                    await asyncio.sleep(1)  # Give termination time
                    continue
                    
            except Exception as e:
                print(f"    SSH {method['description']} failed: {type(e).__name__}")
                continue
        
        # Progressive delay between full retry cycles
        delay = retry_delays[min(retry_index, len(retry_delays) - 1)]
        print(f"    All SSH methods failed. Retrying in {delay}s...")
        await asyncio.sleep(delay)
        retry_index += 1
    
    print(f"--> ❌ Timeout waiting for SSH service on instance {instance_id} after {timeout}s")
    return False

async def wait_for_instance_ready(instance_id: int, timeout: int = 300) -> bool:
    """Waits for instance to be ready and running."""
    print(f"--> Waiting for instance {instance_id} to be ready...")
    start_time = asyncio.get_event_loop().time()
    
    while (asyncio.get_event_loop().time() - start_time) < timeout:
        try:
            process = await asyncio.create_subprocess_exec(
                "vastai", "show", "instance", str(instance_id), "--raw",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0 and stdout:
                instance_info = json.loads(stdout)
                
                if instance_info.get('actual_status') == 'running':
                    print(f"--> Instance {instance_id} is now running!")
                    # Additional wait for SSH to be ready
                    if await wait_for_ssh_ready(instance_id):
                        return True
                    else:
                        return False
                    
                print(f"    Current status: {instance_info.get('actual_status', 'unknown')}")
                await asyncio.sleep(10)
                
        except Exception as e:
            print(f"    Error checking instance status: {e}")
            await asyncio.sleep(10)
    
    print(f"--> Timeout waiting for instance {instance_id} to be ready")
    return False

async def wait_for_instance_ready_with_progress(instance_id: int, timeout: int = 300, callback=None) -> bool:
    """Enhanced wait function with progress updates"""
    print(f"--> Waiting for instance {instance_id} to be ready...")
    start_time = asyncio.get_event_loop().time()
    last_status = None
    
    while (asyncio.get_event_loop().time() - start_time) < timeout:
        try:
            process = await asyncio.create_subprocess_exec(
                "vastai", "show", "instance", str(instance_id), "--raw",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0 and stdout:
                instance_info = json.loads(stdout)
                current_status = instance_info.get('actual_status', 'unknown')
                
                if current_status != last_status:
                    print(f"    Instance status: {current_status}")
                    last_status = current_status
                
                if current_status == 'running':
                    print(f"--> Instance {instance_id} is now running!")
                    # Additional wait for SSH to be ready
                    if await wait_for_ssh_ready(instance_id):
                        return True
                    else:
                        print(f"--> SSH service failed to become ready for instance {instance_id}")
                        return False
                        
            await asyncio.sleep(10)
            
        except Exception as e:
            print(f"    Error checking instance status: {e}")
            await asyncio.sleep(10)
    
    print(f"--> Timeout waiting for instance {instance_id} to be ready")
    return False

async def destroy_instance(instance_id: int):
    """Destroys a Vast.ai instance."""
    print(f"--> Destroying instance {instance_id}...")
    try:
        process = await asyncio.create_subprocess_exec(
            "vastai", "destroy", "instance", str(instance_id),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            print("--> Instance destroyed successfully.")
        else:
            print(f"--> Error destroying instance: {stderr.decode()}")
    except Exception as e:
        print(f"--> Error destroying instance: {e}")