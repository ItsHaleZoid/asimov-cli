import subprocess
import json
import time
import os
from datetime import datetime

def search_cheapest_instance(gpu_name="H100 NVL", num_gpus=1):
    """Searches for the cheapest available interruptible bid instance on Vast.ai."""
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] --> Searching for the cheapest {num_gpus}x {gpu_name} interruptible bid instance...")
    
    # Remove GPU type restriction - let vast.ai handle validation
    # This allows for any GPU type including RTX 4090, A100, etc.
    
    gpu_query_name = gpu_name.replace(' ', '_')
    query = f'num_gpus={num_gpus} gpu_name={gpu_query_name} rentable=true'
    
    command = [
        "vastai", "search", "offers", query,
        "--type", "bid",
        "--order", "dph_total",
        "--raw"
    ]
    
    print(f"    Running command: {' '.join(command)}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
        if not result.stdout.strip():
            print(f"[{timestamp}] --> The search returned no instances for your query.")
            print(f"    Try checking available GPUs with: vastai search offers 'rentable=true' --raw")
            return None, None

        instances = json.loads(result.stdout)
        if not instances:
            print(f"[{timestamp}] --> No suitable interruptible bid instances found.")
            print(f"    Query used: {query}")
            return None, None
            
        cheapest = instances[0]
        original_bid = cheapest['dph_total']
        # Use a small markup (10%) instead of 50% to avoid overbidding
        bid_price = original_bid * 1.1
        
        print(f"--> Found interruptible bid instance {cheapest['id']} at ${cheapest['dph_total']}/hr")
        print(f"    Original bid: ${original_bid}/hr, using bid: ${bid_price:.4f}/hr (10% markup)")
        print(f"    GPU: {cheapest.get('gpu_name', 'Unknown')} | RAM: {cheapest.get('gpu_ram', 'Unknown')}MB")
        return cheapest['id'], bid_price
    except subprocess.TimeoutExpired:
        print(f"--> Timeout while searching for instances. Please check your internet connection.")
        return None, None
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"--> Error searching for instances: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"    Error details: {e.stderr}")
        return None, None

def create_instance(instance_id, docker_image, env_vars, local_train_script="training/train.py", bid_price=None):
    """Creates a Vast.ai interruptible bid instance and sets it up for training."""
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] --> Attempting to create instance {instance_id} with bid ${bid_price}...")
    
    existing_instances = get_running_instances()
    if existing_instances:
        print(f"--> Found {len(existing_instances)} running instances. Destroying them first...")
        for existing_id in existing_instances:
            destroy_instance(existing_id)

    # Build environment variables string for Docker
    env_string = ""
    for key, value in env_vars.items():
        env_string += f"-e {key}='{value}' "
    
    command = [
        "vastai", "create", "instance", str(instance_id),
        "--image", docker_image,
        "--disk", "100",
        "--bid", str(bid_price),
        "--env", env_string.strip(),
        "--onstart-cmd", "chmod +x /app/start_training.sh && /app/start_training.sh",
        "--raw"
    ]
             
    print(f"    Running command: {' '.join(command)}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        instance_info = json.loads(result.stdout)
        if instance_info.get("success"):
            new_id = instance_info['new_contract']
            print(f"--> Successfully created instance {new_id}. Waiting for it to become ready...")
            
            if wait_for_instance_ready(new_id):
                print(f"--> Instance {new_id} is ready for training setup.")
                
                # The rest of the setup (file upload, training start) is now handled in main.py
                return new_id
            else:
                print(f"--> Instance {new_id} failed to become ready. Destroying it.")
                destroy_instance(new_id)
                return None
        else:
            print(f"--> Failed to create instance. Response: {instance_info}")
            return None
    except subprocess.TimeoutExpired:
        print(f"--> Timeout while creating instance. This may indicate network issues.")
        return None
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"--> Error creating instance: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"    Error details: {e.stderr}")
        return None

def get_running_instances():
    """Gets a list of currently running instances."""
    try:
        result = subprocess.run(["vastai", "show", "instances", "--raw"], capture_output=True, text=True, check=True)
        if not result.stdout.strip():
            return []
        instances = json.loads(result.stdout)
        return [instance['id'] for instance in instances if instance.get('actual_status') == 'running']
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"--> Error getting running instances: {e}")
        return []

def get_instance_ssh_info(instance_id):
    """Gets SSH connection info for an instance."""
    try:
        result = subprocess.run(["vastai", "show", "instance", str(instance_id), "--raw"], capture_output=True, text=True, check=True)
        instance_info = json.loads(result.stdout)
        ssh_host = instance_info.get('ssh_host')
        ssh_port = instance_info.get('ssh_port')
        if ssh_host and ssh_port:
            return ssh_host, ssh_port
        return None, None
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"--> Error getting SSH info for instance {instance_id}: {e}")
        return None, None

def verify_file_upload(instance_id, remote_path):
    """Verifies that a file was successfully uploaded to the instance."""
    ssh_host, ssh_port = get_instance_ssh_info(instance_id)
    if not ssh_host:
        return False
    
    try:
        verify_command = [
            "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=10",
            f"root@{ssh_host}", f"ls -la {remote_path}"
        ]
        result = subprocess.run(verify_command, capture_output=True, text=True, timeout=15)
        return result.returncode == 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        return False

def wait_for_ssh_ready(instance_id, timeout=300):
    """Waits for the SSH service to be ready on the instance."""
    print(f"--> Waiting for SSH on instance {instance_id}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        ssh_host, ssh_port = get_instance_ssh_info(instance_id)
        if ssh_host and ssh_port:
            try:
                test_command = [
                    "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no",
                    "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=15",
                    f"root@{ssh_host}", "echo 'SSH_OK'"
                ]
                result = subprocess.run(test_command, capture_output=True, text=True, timeout=20)
                if result.returncode == 0 and 'SSH_OK' in result.stdout:
                    print(f"--> SSH is ready on instance {instance_id}.")
                    return True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass
        time.sleep(15)
    
    print(f"--> Timeout waiting for SSH on instance {instance_id}.")
    return False

def wait_for_instance_ready(instance_id, timeout=300):
    """Waits for an instance to be in the 'running' state."""
    print(f"--> Waiting for instance {instance_id} to be ready...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            result = subprocess.run(["vastai", "show", "instance", str(instance_id), "--raw"], capture_output=True, text=True, check=True)
            instance_info = json.loads(result.stdout)
            if instance_info.get('actual_status') == 'running':
                print(f"--> Instance {instance_id} is running.")
                return wait_for_ssh_ready(instance_id)
            print(f"    Current status: {instance_info.get('actual_status', 'unknown')}")
            time.sleep(10)
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"    Error checking instance status: {e}")
            time.sleep(10)
    
    print(f"--> Timeout waiting for instance {instance_id} to become ready.")
    return False

def monitor_ssh_and_upload_files(instance_id, files_to_upload, max_wait_time=600):
    """Monitors SSH and uploads files until successful or timeout."""
    print(f"\n--> Starting SSH monitoring and file upload for instance {instance_id}...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        if wait_for_ssh_ready(instance_id):
            all_files_uploaded = True
            for local_path, remote_path in files_to_upload:
                print(f"--> Uploading {local_path} to {remote_path}...")
                if not upload_file_comprehensive(instance_id, local_path, remote_path):
                    all_files_uploaded = False
                    break
            
            if all_files_uploaded:
                print("\n--> All files uploaded successfully!")
                return True
        
        print("    Retrying file upload in 20 seconds...")
        time.sleep(20)
    
    print(f"\n--> Timeout reached. Failed to upload all files to instance {instance_id}.")
    return False

def upload_file_comprehensive(instance_id, local_path, remote_path):
    """Uploads a single file with verification."""
    if not os.path.exists(local_path):
        print(f"    ERROR: Local file not found: {local_path}")
        return False

    ssh_host, ssh_port = get_instance_ssh_info(instance_id)
    if not ssh_host:
        return False
        
    try:
        remote_dir = os.path.dirname(remote_path)
        if remote_dir != '/':
            mkdir_command = [
                "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=10",
                f"root@{ssh_host}", f"mkdir -p {remote_dir}"
            ]
            subprocess.run(mkdir_command, capture_output=True, text=True, timeout=15)
        
        scp_command = [
            "scp", "-P", str(ssh_port), "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=30",
            local_path, f"root@{ssh_host}:{remote_path}"
        ]
        
        result = subprocess.run(scp_command, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0 and verify_file_upload(instance_id, remote_path):
            print(f"    Successfully uploaded and verified {os.path.basename(local_path)}")
            return True
        else:
            print(f"    ERROR: Failed to upload {os.path.basename(local_path)}.")
            if result.stderr:
                print(f"    SCP Error: {result.stderr.strip()}")
            return False
            
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"    ERROR: Exception during upload of {local_path}: {e}")
        return False

def get_instance_detailed_status(instance_id):
    """Gets and prints detailed status for a given instance."""
    print(f"--> Getting detailed status for instance {instance_id}...")
    try:
        result = subprocess.run(["vastai", "show", "instance", str(instance_id), "--raw"], capture_output=True, text=True, check=True, timeout=30)
        instance_info = json.loads(result.stdout)
        
        print("\n--- Instance Status Report ---")
        for key, value in instance_info.items():
            print(f"  {key}: {value}")
        print("----------------------------\n")
        
        return instance_info
    except Exception as e:
        print(f"--> Error getting detailed status for instance {instance_id}: {e}")
        return None

def debug_instance_status(instance_id):
    """Runs a series of debug commands on the instance."""
    print(f"\n--- Debugging Instance {instance_id} ---")
    
    commands = {
        "Directory Listing": "ls -la /app/",
        "Running Processes": "ps aux | grep python",
        "Training Log": "cat /app/training.log 2>/dev/null || echo 'Log not found'",
        "Disk Space": "df -h",
        "Memory Usage": "free -h",
    }
    
    for name, cmd in commands.items():
        print(f"\n--- {name} ---")
        try:
            result = subprocess.run(["vastai", "execute", str(instance_id), cmd], capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(result.stdout.strip() if result.stdout.strip() else "(no output)")
            else:
                print(f"  ERROR: Command failed. stderr: {result.stderr.strip()}")
        except Exception as e:
            print(f"  ERROR: Could not execute command: {e}")
    
    print("\n--- Debugging Complete ---")

def destroy_instance(instance_id):
    """Destroys a Vast.ai instance."""
    print(f"--> Destroying instance {instance_id}...")
    try:
        subprocess.run(["vastai", "destroy", "instance", str(instance_id)], check=True)
        print(f"--> Instance {instance_id} destroyed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"--> Error destroying instance {instance_id}: {e}")