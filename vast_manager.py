import subprocess
import json
import time

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

def create_instance(instance_id, docker_image, env_vars):
    """Creates a Vast.ai instance, passing configuration as environment variables."""
    print(f"--> Attempting to create instance {instance_id}...")
    
    existing_instances = get_running_instances()
    if existing_instances:
        print(f"--> Found {len(existing_instances)} existing instance(s). Destroying them first...")
        for existing_id in existing_instances:
            destroy_instance(existing_id)
    
    # Build environment variables string for the command
    env_string = " ".join([f"{key}='{value}'" for key, value in env_vars.items()])
    run_command = f"bash -c 'export {env_string} && python /app/train.py'"

    command = [
        "vastai", "create", "instance", str(instance_id),
        "--image", docker_image,
        "--disk", "50",
        "--args", run_command,
        "--raw"
    ]
    
    # Also add environment variables to the vastai command itself for redundancy
    for key, value in env_vars.items():
        command.extend(["--env", f"{key}={value}"])
             
    print(f"    Running command: {' '.join(command)}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        instance_info = json.loads(result.stdout)
        if instance_info.get("success"):
            new_id = instance_info['new_contract']
            print(f"--> Successfully created instance {new_id}. Waiting for it to start...")
            return new_id
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