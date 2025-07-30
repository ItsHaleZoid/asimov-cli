#!/usr/bin/env python3
"""
Vast.ai Multiple Instance Test Script
Tests launching multiple GPU instances simultaneously to determine parallel capacity limits.
"""

import sys
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Add parent directory to path to import vast_manager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vast_manager import get_running_instances, destroy_instance

class InstanceResult:
    def __init__(self, thread_id, instance_id=None, success=False, error=None, duration=None):
        self.thread_id = thread_id
        self.instance_id = instance_id
        self.success = success
        self.error = error
        self.duration = duration
        self.timestamp = datetime.now()

class VastMultipleInstanceTester:
    def __init__(self):
        self.results = []
        self.running_instances = []
        self.lock = threading.Lock()
        
        # Default test configuration
        self.default_config = {
            "gpu_name": "H100 NVL",
            "num_gpus": 1,
            "docker_image": "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel",
            "env_vars": {
                "HF_TOKEN": os.getenv("HF_TOKEN", "test_token"),
                "BASE_MODEL": "microsoft/DialoGPT-small",
                "DATASET": "test_dataset",
                "LORA_RANK": "8"
            }
        }
    
    def launch_single_instance(self, thread_id, config=None):
        """Launch a single instance and track the result."""
        if config is None:
            config = self.default_config
            
        start_time = time.time()
        result = InstanceResult(thread_id)
        
        try:
            print(f"[Thread {thread_id}] Starting instance search...")
            
            # Search for cheapest instance with quiet mode
            instance_id, bid_price = self.search_instance_quiet(
                gpu_name=config["gpu_name"],
                num_gpus=config["num_gpus"],
                thread_id=thread_id
            )
            
            if not instance_id:
                result.error = "No suitable instance found"
                result.duration = time.time() - start_time
                return result
            
            print(f"[Thread {thread_id}] Found instance {instance_id}, creating with bid ${bid_price:.4f}/hr...")
            
            # Create instance using test-specific function that doesn't cleanup existing instances
            created_id = self.create_test_instance(
                instance_id=instance_id,
                docker_image=config["docker_image"],
                env_vars=config["env_vars"],
                bid_price=bid_price,
                thread_id=thread_id
            )
            
            if created_id:
                result.instance_id = created_id
                result.success = True
                result.duration = time.time() - start_time
                
                with self.lock:
                    self.running_instances.append(created_id)
                    
                print(f"[Thread {thread_id}] Successfully created instance {created_id} in {result.duration:.2f}s")
            else:
                result.error = "Failed to create instance"
                result.duration = time.time() - start_time
                
        except Exception as e:
            result.error = str(e)
            result.duration = time.time() - start_time
            print(f"[Thread {thread_id}] Error: {e}")
            
        return result
    
    def search_instance_quiet(self, gpu_name="H100 NVL", num_gpus=1, thread_id=1):
        """Quiet version of search_cheapest_instance with minimal output."""
        import subprocess
        import json
        
        # Try only the most likely to succeed strategies
        strategies = [
            f'num_gpus={num_gpus} gpu_name="H100 NVL" rentable=true dph_total<=1.0 disk_space>=100',
            f'num_gpus={num_gpus} gpu_name="H100 NVL" rentable=true disk_space>=100',
            f'num_gpus={num_gpus} gpu_name~="H100" rentable=true disk_space>=100'
        ]
        
        for i, criteria in enumerate(strategies):
            try:
                command = [
                    "vastai", "search", "offers", criteria,
                    "--interruptible", "--order", "dph_total asc", "--raw"
                ]
                
                result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=15)
                
                if result.stdout.strip():
                    instances = json.loads(result.stdout)
                    if instances:
                        cheapest = sorted(instances, key=lambda x: float(x.get('dph_total', float('inf'))))[0]
                        recommended_price = float(cheapest.get('min_bid', cheapest['dph_total']))
                        bid_price = recommended_price + (recommended_price * 0.50)  # 50% buffer
                        
                        print(f"[Thread {thread_id}] Found instance {cheapest['id']} at ${bid_price:.4f}/hr")
                        return cheapest['id'], bid_price
                        
            except (subprocess.CalledProcessError, json.JSONDecodeError, subprocess.TimeoutExpired):
                continue
        
        print(f"[Thread {thread_id}] No instances found")
        return None, None
    
    def create_test_instance(self, instance_id, docker_image, env_vars, bid_price, thread_id):
        """Test-specific instance creation that doesn't interfere with other instances."""
        import subprocess
        import json
        
        print(f"[Thread {thread_id}] Creating instance {instance_id} with bid ${bid_price:.4f}/hr...")
        
        # Create instance with bid - minimal setup for testing
        command = [
            "vastai", "create", "instance", str(instance_id),
            "--image", docker_image,
            "--disk", "100",
            "--bid", str(bid_price),
            "--raw"
        ]
        
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
            instance_info = json.loads(result.stdout)
            
            if instance_info.get("success"):
                new_id = instance_info['new_contract']
                print(f"[Thread {thread_id}] Instance {new_id} created successfully")
                return new_id
            else:
                print(f"[Thread {thread_id}] Failed to create instance: {instance_info}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"[Thread {thread_id}] Instance creation timed out")
            return None
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"[Thread {thread_id}] Error creating instance: {e}")
            return None
    
    def test_parallel_launch(self, num_instances=3, max_workers=None):
        """Test launching multiple instances in parallel."""
        print(f"\n=== Testing Parallel Launch of {num_instances} Instances ===")
        print(f"Timestamp: {datetime.now()}")
        print(f"Max workers: {max_workers or 'Default'}")
        
        # Clear previous results
        self.results = []
        self.running_instances = []
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_thread = {
                executor.submit(self.launch_single_instance, i): i 
                for i in range(1, num_instances + 1)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_thread):
                thread_id = future_to_thread[future]
                try:
                    result = future.result()
                    self.results.append(result)
                except Exception as e:
                    error_result = InstanceResult(thread_id, error=str(e))
                    self.results.append(error_result)
        
        total_duration = time.time() - start_time
        
        # Print results summary
        self.print_test_results(total_duration)
        
        return self.results
    
    def test_sequential_launch(self, num_instances=3):
        """Test launching multiple instances sequentially for comparison."""
        print(f"\n=== Testing Sequential Launch of {num_instances} Instances ===")
        print(f"Timestamp: {datetime.now()}")
        
        # Clear previous results
        self.results = []
        self.running_instances = []
        
        start_time = time.time()
        
        for i in range(1, num_instances + 1):
            result = self.launch_single_instance(i)
            self.results.append(result)
            
            # Small delay between sequential launches
            time.sleep(2)
        
        total_duration = time.time() - start_time
        
        # Print results summary
        self.print_test_results(total_duration)
        
        return self.results
    
    def print_test_results(self, total_duration):
        """Print detailed test results."""
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        print(f"\n=== Test Results Summary ===")
        print(f"Total time: {total_duration:.2f}s")
        print(f"Total instances attempted: {len(self.results)}")
        print(f"Successful launches: {len(successful)}")
        print(f"Failed launches: {len(failed)}")
        print(f"Success rate: {len(successful)/len(self.results)*100:.1f}%")
        
        if successful:
            avg_duration = sum(r.duration for r in successful) / len(successful)
            print(f"Average launch time (successful): {avg_duration:.2f}s")
            print(f"Fastest launch: {min(r.duration for r in successful):.2f}s")
            print(f"Slowest launch: {max(r.duration for r in successful):.2f}s")
        
        print(f"\n=== Detailed Results ===")
        for result in sorted(self.results, key=lambda x: x.thread_id):
            status = "✓ SUCCESS" if result.success else "✗ FAILED"
            duration = f"{result.duration:.2f}s" if result.duration else "N/A"
            instance_info = f"(ID: {result.instance_id})" if result.instance_id else ""
            error_info = f"- {result.error}" if result.error else ""
            
            print(f"Thread {result.thread_id}: {status} {duration} {instance_info} {error_info}")
        
        if self.running_instances:
            print(f"\n=== Running Instances ===")
            for instance_id in self.running_instances:
                print(f"Instance {instance_id}: Running")
    
    def cleanup_all_instances(self):
        """Clean up all created instances."""
        print(f"\n=== Cleaning Up All Instances ===")
        
        # Get all running instances (not just ones we created)
        all_running = get_running_instances()
        
        if not all_running:
            print("No running instances found.")
            return
        
        print(f"Found {len(all_running)} running instances. Destroying...")
        
        for instance_id in all_running:
            try:
                destroy_instance(instance_id)
                print(f"Destroyed instance {instance_id}")
            except Exception as e:
                print(f"Failed to destroy instance {instance_id}: {e}")
        
        # Clear our tracking list
        self.running_instances = []
        print("Cleanup completed.")
    
    def stress_test(self, max_instances=10, step_size=2):
        """Perform a stress test to find the maximum parallel capacity."""
        print(f"\n=== Stress Test: Finding Maximum Parallel Capacity ===")
        print(f"Testing up to {max_instances} instances in steps of {step_size}")
        
        stress_results = {}
        
        for num_instances in range(step_size, max_instances + 1, step_size):
            print(f"\n--- Testing {num_instances} parallel instances ---")
            
            # Clean up any previous instances
            self.cleanup_all_instances()
            time.sleep(10)  # Wait for cleanup to complete
            
            # Run the test
            results = self.test_parallel_launch(num_instances)
            
            successful_count = len([r for r in results if r.success])
            success_rate = successful_count / len(results) * 100
            
            stress_results[num_instances] = {
                'attempted': len(results),
                'successful': successful_count,
                'success_rate': success_rate,
                'results': results
            }
            
            print(f"Result: {successful_count}/{len(results)} successful ({success_rate:.1f}%)")
            
            # Stop if success rate drops below 50%
            if success_rate < 50:
                print(f"Success rate dropped below 50%. Stopping stress test.")
                break
            
            # Small delay between stress test rounds
            time.sleep(30)
        
        print(f"\n=== Stress Test Summary ===")
        for num_instances, data in stress_results.items():
            print(f"{num_instances} instances: {data['successful']}/{data['attempted']} successful ({data['success_rate']:.1f}%)")
        
        return stress_results

def main():
    """Main test runner."""
    tester = VastMultipleInstanceTester()
    
    print("Vast.ai Multiple Instance Tester")
    print("================================")
    
    if len(sys.argv) < 2:
        print("Usage: python vast_multiple_test.py <test_type>")
        print("Test types:")
        print("  parallel <num_instances> [max_workers] - Test parallel launch")
        print("  sequential <num_instances> - Test sequential launch")
        print("  stress [max_instances] [step_size] - Stress test to find limits")
        print("  cleanup - Clean up all running instances")
        print("  quick - Quick test (3 parallel instances)")
        return
    
    test_type = sys.argv[1].lower()
    
    try:
        if test_type == "parallel":
            num_instances = int(sys.argv[2]) if len(sys.argv) > 2 else 3
            max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else None
            tester.test_parallel_launch(num_instances, max_workers)
            
        elif test_type == "sequential":
            num_instances = int(sys.argv[2]) if len(sys.argv) > 2 else 3
            tester.test_sequential_launch(num_instances)
            
        elif test_type == "stress":
            max_instances = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            step_size = int(sys.argv[3]) if len(sys.argv) > 3 else 2
            tester.stress_test(max_instances, step_size)
            
        elif test_type == "cleanup":
            tester.cleanup_all_instances()
            
        elif test_type == "quick":
            print("Running quick test with 3 parallel instances...")
            tester.test_parallel_launch(3)
            
        else:
            print(f"Unknown test type: {test_type}")
            return
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        print("Cleaning up instances...")
        tester.cleanup_all_instances()
    except Exception as e:
        print(f"\nError during test: {e}")
        print("Cleaning up instances...")
        tester.cleanup_all_instances()

if __name__ == "__main__":
    main()