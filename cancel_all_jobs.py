#!/usr/bin/env python3
"""
Simple script to cancel all jobs.
Usage: python cancel_all_jobs.py [--force]
"""

import requests
import os
import sys
import argparse

def cancel_all_jobs(force=False):
    """Cancel all jobs via the API server"""
    api_url = os.getenv("ASIMOV_API_URL", "http://localhost:8000")
    auth_token = "sb-zxebusnnyzvaktqpmuft-auth-token"
    
    if not auth_token:
        print("Error: ASIMOV_AUTH_TOKEN environment variable not set")
        print("Please set your authentication token:")
        print("export ASIMOV_AUTH_TOKEN='your_token_here'")
        return False
    
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    
    try:
        # Get current jobs first
        response = requests.get(f"{api_url}/api/jobs", headers=headers)
        if response.status_code == 401:
            print("Authentication failed. Please check your token.")
            return False
        elif response.status_code != 200:
            print(f"Failed to fetch jobs: {response.status_code}")
            return False
        
        jobs_data = response.json()
        jobs = jobs_data.get("jobs", [])
        active_jobs = [job for job in jobs if job.get("status") in ["initializing", "preparing", "running", "training"]]
        
        if not active_jobs:
            print("No active jobs found to cancel.")
            return True
        
        print(f"Found {len(active_jobs)} active jobs:")
        for job in active_jobs:
            print(f"  - {job['id']}: {job.get('model_name', 'Unknown')} ({job.get('status', 'Unknown')})")
        
        if not force:
            confirm = input(f"\nProceed to cancel {len(active_jobs)} jobs? (y/N): ").strip().lower()
            if confirm not in ['y', 'yes']:
                print("Operation cancelled.")
                return False
        
        # Cancel all jobs
        print("\nCancelling all jobs...")
        response = requests.post(f"{api_url}/api/jobs/cancel-all", headers=headers)
        
        if response.status_code == 401:
            print("Authentication failed. Please check your token.")
            return False
        elif response.status_code != 200:
            print(f"Failed to cancel jobs: {response.status_code}")
            print(f"Error response: {response.text}")
            return False
        
        result = response.json()
        
        print(f"\nResults:")
        print(f"Message: {result.get('message', 'Unknown')}")
        
        cancelled_jobs = result.get('cancelled_jobs', [])
        if cancelled_jobs:
            print(f"\nSuccessfully cancelled {len(cancelled_jobs)} jobs:")
            for job in cancelled_jobs:
                print(f"  ✓ {job['job_id']}")
        
        failed_cancellations = result.get('failed_cancellations', [])
        if failed_cancellations:
            print(f"\nFailed to cancel {len(failed_cancellations)} jobs:")
            for job in failed_cancellations:
                print(f"  ✗ {job['job_id']}: {job['error']}")
        
        return len(failed_cancellations) == 0
        
    except requests.RequestException as e:
        print(f"Network error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Cancel all training jobs")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompts")
    
    args = parser.parse_args()
    
    success = cancel_all_jobs(args.force)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()