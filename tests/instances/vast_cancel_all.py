#!/usr/bin/env python3
"""
Script to cancel all jobs via the API server.
Usage: python tests/instances/vast_cancel_all.py cancel [--force]
"""

import argparse
import asyncio
import aiohttp
import json
import os
import sys
from typing import Optional

# Default API server URL
DEFAULT_API_URL = "http://localhost:8000"

async def get_auth_token() -> Optional[str]:
    """Get authentication token from environment or config"""
    # Try to get token from environment variable
    token = os.getenv("ASIMOV_AUTH_TOKEN")
    if token:
        return token
    
    # You might want to add other methods to get the token here
    # For now, we'll prompt the user
    print("No authentication token found.")
    print("Please set the ASIMOV_AUTH_TOKEN environment variable or")
    print("provide your token:")
    token = input("Auth token: ").strip()
    return token if token else None

async def cancel_all_jobs(api_url: str, auth_token: str, force: bool = False) -> bool:
    """Cancel all jobs via the API"""
    
    if not force:
        print("This will cancel ALL your active jobs.")
        confirm = input("Are you sure you want to continue? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("Operation cancelled.")
            return False
    
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            # First, get the list of jobs to show what will be cancelled
            async with session.get(f"{api_url}/api/jobs", headers=headers) as response:
                if response.status == 401:
                    print("Authentication failed. Please check your token.")
                    return False
                elif response.status != 200:
                    print(f"Failed to fetch jobs: {response.status}")
                    return False
                
                jobs_data = await response.json()
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
            async with session.post(f"{api_url}/api/jobs/cancel-all", headers=headers) as response:
                if response.status == 401:
                    print("Authentication failed. Please check your token.")
                    return False
                elif response.status != 200:
                    print(f"Failed to cancel jobs: {response.status}")
                    response_text = await response.text()
                    print(f"Error response: {response_text}")
                    return False
                
                result = await response.json()
                
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
                
    except aiohttp.ClientError as e:
        print(f"Network error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

async def main():
    parser = argparse.ArgumentParser(description="Cancel all training jobs")
    parser.add_argument("action", choices=["cancel"], help="Action to perform")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompts")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help=f"API server URL (default: {DEFAULT_API_URL})")
    
    args = parser.parse_args()
    
    if args.action == "cancel":
        # Get authentication token
        auth_token = await get_auth_token()
        if not auth_token:
            print("No authentication token provided. Exiting.")
            sys.exit(1)
        
        # Cancel all jobs
        success = await cancel_all_jobs(args.api_url, auth_token, args.force)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())