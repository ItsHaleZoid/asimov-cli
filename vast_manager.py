import subprocess
import json
import time
import os

def search_cheapest_instance(gpu_name="RTX 4090", num_gpus=1):
    """Searches for the cheapest available spot instance on Vast.ai."""
    print(f"--> Searching for the cheapest {num_gpus}x {gpu_name} spot instance...")
    
    command = [
        "vastai", "search", "offers",
        f'num_gpus={num_gpus} gpu_name="{gpu_name}" rentable=true verified=true inet_down_cost=0 disk_space>=50',
        "--order", "dph_total asc",
        "--raw"
    ]
    
    print(f"    Running command: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            capture_output=True, text=True, check=True
        )
        
        if not result.stdout.strip():
            print("--> The search query returned an empty response.")
            print("    This usually means no spot instances are currently available that match your criteria.")
            return None

        instances = json.loads(result.stdout)
        if not instances:
            print("--> No suitable spot instances found.")
            return None
            
        cheapest = instances[0]
        print(f"--> Found spot instance {cheapest['id']} at ${cheapest['dph_total']}/hr")
        return cheapest['id']
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"--> Error searching for instances: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"    Vast.ai CLI error: {e.stderr}")
        return None

def create_instance(instance_id, docker_image, env_vars, local_train_script="training/train.py"):
    """Creates a Vast.ai instance and uploads local training script."""
    print(f"--> Attempting to create instance {instance_id}...")
    
    existing_instances = get_running_instances()
    if existing_instances:
        print(f"--> Found {len(existing_instances)} existing instance(s). Destroying them first...")
        for existing_id in existing_instances:
            destroy_instance(existing_id)

    # Create instance without onstart command first
    command = [
        "vastai", "create", "instance", str(instance_id),
        "--image", docker_image,
        "--disk", "50",
        "--raw"
    ]
    
    # Add environment variables to command
    for key, value in env_vars.items():
        command.extend(["-e", f"{key}={value}"])
             
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
                
                # Upload local training script
                if upload_file_to_instance(new_id, local_train_script, "/app/train.py"):
                    # Start training
                    start_training_command = f"cd /app && python train.py"
                    if execute_command_on_instance(new_id, start_training_command):
                        print(f"--> Training started successfully on instance {new_id}!")
                        return new_id
                    else:
                        print(f"--> Failed to start training on instance {new_id}")
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
    """Uploads a local file to the instance via SCP."""
    ssh_host, ssh_port = get_instance_ssh_info(instance_id)
    if not ssh_host or not ssh_port:
        print(f"--> Could not get SSH info for instance {instance_id}")
        return False
    
    print(f"--> Uploading {local_file_path} to instance {instance_id}...")
    
    try:
        scp_command = [
            "scp", "-P", str(ssh_port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            local_file_path,
            f"root@{ssh_host}:{remote_path}"
        ]
        
        subprocess.run(scp_command, capture_output=True, text=True, check=True)
        print(f"--> Successfully uploaded {os.path.basename(local_file_path)}")
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
    
    print(f"--> Executing command on instance {instance_id}: {command}")
    
    try:
        ssh_command = [
            "ssh", "-p", str(ssh_port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            f"root@{ssh_host}",
            command
        ]
        
        # Run in background so it doesn't block
        subprocess.Popen(ssh_command)
        print(f"--> Command started successfully on instance {instance_id}")
        return True
        
    except Exception as e:
        print(f"--> Error executing command: {e}")
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
                return True
                
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