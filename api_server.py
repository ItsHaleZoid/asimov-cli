from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import asyncio
import subprocess
import yaml
import uuid
import json
import os
import re
from datetime import datetime
from typing import Dict, Optional, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "ws://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainingRequest(BaseModel):
    model_id: str
    dataset_id: str
    dataset_name: str
    model_name: str

class TrainingJob:
    def __init__(self, job_id: str, request: TrainingRequest, config: dict):
        self.job_id: str = job_id
        self.config: dict = config
        self.status: str = "initializing"
        self.progress: int = 0
        self.logs: List[Dict[str, Any]] = []
        self.start_time: datetime = datetime.now()
        self.instance_id: Optional[str] = None
        self.process: Optional[asyncio.subprocess.Process] = None
        ## FIX: Store display names directly in the job object for consistent state.
        self.model_name: str = request.model_name
        self.dataset_name: str = request.dataset_name

    ## FIX: Add a method to serialize the job object to a dictionary for API/WebSocket responses.
    ## This prevents sending non-serializable objects like 'process' or 'datetime'.
    def to_dict(self) -> dict:
        return {
            "id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "logs": self.logs,
            "config": self.config,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "start_time": self.start_time.isoformat(),
            "instance_id": self.instance_id
        }

## FIX: The dictionary should store the actual TrainingJob objects, not separate dicts.
## This is the single source of truth for a job's state.
training_jobs: Dict[str, TrainingJob] = {}

# WebSocket connections for real-time updates
websocket_connections: Dict[str, WebSocket] = {}

@app.post("/api/start-training")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    
    model_mapping = {
        "mistral-instruct-7b": "mistralai/Mistral-7B-Instruct-v0.2",
        "mistral-small-24b": "mistralai/Mistral-Small-Instruct-2409",
        "mistral-codestral-22b": "mistralai/Codestral-22B-v0.1",
        "mistral-devstral-22b": "mistralai/Ministral-8B-Instruct-2410",
        "gemma-3n-2b": "google/gemma-2-2b",
        "gemma-3n-4b": "google/gemma-2-9b",
    }
    
    job_config = {
        "docker_image": "halez/lora-finetuner:latest",
        "hf_token": "hf_KJkLuvPGXojpqFfppMCcApgxRMbIsmYbDis",
        "base_model_id": model_mapping.get(request.model_id, request.model_id),
        "dataset_id": request.dataset_id,
        "lora_model_repo": f"Ihmzf/{request.model_id}-lora", # Dynamic repo name
        "gpu_name": "H100 NVL",
        "num_gpus": 1,
        "max_train_steps": 100
    }
    
    ## FIX: Create the TrainingJob object and store IT as the single source of truth.
    job = TrainingJob(job_id, request, job_config)
    training_jobs[job_id] = job
    
    background_tasks.add_task(run_training_job, job_id)
    
    return {"job_id": job_id, "status": "started"}

async def run_training_job(job_id: str):
    job = training_jobs.get(job_id)
    if not job:
        logger.error(f"Job {job_id} not found for execution.")
        return

    try:
        await update_job_status(job_id, "preparing", 5, "Writing configuration file...")
        
        config_path = f"jobs/{job.job_id}_config.yaml"
        os.makedirs("jobs", exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(job.config, f)
        
        await update_job_status(job_id, "searching_gpu", 10, "Starting training script...")
        
        process = await asyncio.create_subprocess_exec(
            'python', 'main.py', '--config', config_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        ## FIX: Store the process handle directly in our single source of truth.
        job.process = process
        
        await monitor_training_process(job_id, process)
        
    except Exception as e:
        logger.error(f"Error starting training job {job_id}: {str(e)}")
        await update_job_status(job_id, "failed", 0, f"Error: {str(e)}")

async def monitor_training_process(job_id: str, process: asyncio.subprocess.Process):
    job = training_jobs[job_id]
    
    status_keywords = {
        # Vast.ai instance setup
        "Searching for the cheapest": ("searching_gpu", 15),
        "Found \\d+ instances": ("found_gpu", 20),
        "Selected cheapest": ("gpu_selected", 22),
        "Attempting to create instance": ("creating_instance", 25),
        "Successfully created instance": ("instance_created", 30),
        "Instance .* is now running": ("instance_running", 35),
        "SSH service is ready": ("ssh_ready", 40),
        "Instance .* is ready": ("instance_ready", 45),
        
        # Script execution and training
        "Uploading training script": ("uploading_script", 50),
        "Command sent": ("setup_complete", 55),
        "loss": ("training", 60),
        
        # Completion and failure
        "Training completed successfully": ("completed", 100),
        ## FIX: More specific failure keywords based on your logs.
        "Job Failed": ("failed", 0),
        "Could not create the instance": ("failed", 0),
        "Error:": ("failed", 0),
    }

    async for line in process.stdout:
        if not line:
            break
            
        line = line.decode('utf-8').strip()
        if line:
            logger.info(f"Job {job_id}: {line}")
            
            # Update status based on keywords
            matched = False
            for keyword, (status, progress) in status_keywords.items():
                if re.search(keyword, line, re.IGNORECASE):
                    # Don't let a generic "loss" message override a final state.
                    if job.status in ["completed", "failed", "cancelled"]:
                        continue

                    if status == "training":
                        parsed_progress = parse_training_progress(line)
                        # Use parsed progress if valid, otherwise use the keyword's base progress.
                        final_progress = parsed_progress if parsed_progress else progress
                        await update_job_status(job_id, status, final_progress, line)
                    else:
                        await update_job_status(job_id, status, progress, line)
                    
                    matched = True
                    break # Stop after first match
            
            if not matched:
                await add_log(job_id, line)
            
            # Check for instance ID
            instance_match = re.search(r'instance (\d+)', line, re.IGNORECASE)
            if instance_match:
                job.instance_id = instance_match.group(1)

    await process.wait()
    
    ## FIX: Robust final status check. Trust the status set by the logs first.
    ## Only override if the job is still in an intermediate state when the process ends.
    job = training_jobs[job_id] # Re-fetch the job to get the latest state
    if job.status not in ["completed", "failed", "cancelled"]:
        stderr_output = await process.stderr.read()
        stderr_text = stderr_output.decode('utf-8').strip()

        if process.returncode != 0:
            error_message = f"Process failed with exit code {process.returncode}. Stderr: {stderr_text}"
            await update_job_status(job_id, "failed", job.progress, error_message)
        else:
            # The process exited cleanly (code 0) but never sent a "completed" signal.
            await update_job_status(job_id, "failed", job.progress, "Job ended unexpectedly without a success or failure signal.")


def parse_training_progress(line: str) -> Optional[int]:
    step_match = re.search(r'(\d+)/(\d+)\s*\[', line)
    if step_match:
        current, total = int(step_match.group(1)), int(step_match.group(2))
        return 60 + int((current / total) * 35) # Training happens between 60% and 95%
    return None

async def update_job_status(job_id: str, status: str, progress: int, message: str):
    job = training_jobs.get(job_id)
    if job:
        job.status = status
        job.progress = progress
        log_entry = {"timestamp": datetime.now().isoformat(), "message": message, "type": "status"}
        job.logs.append(log_entry)
        
        if job_id in websocket_connections:
            try:
                await websocket_connections[job_id].send_json({
                    "type": "status_update",
                    "job": job.to_dict() # Send the whole serialized job object
                })
            except Exception as e:
                logger.warning(f"Failed to send WebSocket update for job {job_id}: {e}")

async def add_log(job_id: str, message: str):
    job = training_jobs.get(job_id)
    if job:
        log_entry = {"timestamp": datetime.now().isoformat(), "message": message, "type": "log"}
        job.logs.append(log_entry)
        
        if job_id in websocket_connections:
            try:
                await websocket_connections[job_id].send_json({
                    "type": "log_update",
                    "log": log_entry
                })
            except Exception as e:
                logger.warning(f"Failed to send WebSocket log for job {job_id}: {e}")

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    websocket_connections[job_id] = websocket
    
    try:
        if job_id in training_jobs:
            ## FIX: Use the to_dict() method for a clean, serializable payload.
            await websocket.send_json({
                "type": "initial_state",
                "job": training_jobs[job_id].to_dict()
            })
        
        while True:
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
        if job_id in websocket_connections:
            del websocket_connections[job_id]
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
        if job_id in websocket_connections:
            del websocket_connections[job_id]

@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    ## FIX: Use the to_dict() method to return a clean JSON response.
    return training_jobs[job_id].to_dict()

@app.post("/api/job/{job_id}/cancel")
async def cancel_job(job_id: str):
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = training_jobs[job_id]
    
    ## FIX: Access the process handle from the job object directly.
    if job.process and job.process.returncode is None:
        try:
            job.process.terminate()
            await job.process.wait() # Ensure it's terminated
            await add_log(job_id, "Training script process terminated by user.")
        except ProcessLookupError:
            await add_log(job_id, "Training script process already finished.")
    
    ## FIX: Correctly destroy the instance using the stored ID.
    if job.instance_id:
        try:
            # Using asyncio.create_subprocess_exec for non-blocking call
            proc = await asyncio.create_subprocess_exec(
                'vastai', 'destroy', 'instance', str(job.instance_id)
            )
            await proc.wait()
            await add_log(job_id, f"Destroy command sent for Vast.ai instance {job.instance_id}")
        except Exception as e:
            logger.error(f"Error destroying instance {job.instance_id}: {e}")
            await add_log(job_id, f"Error destroying instance {job.instance_id}: {e}")
    
    await update_job_status(job_id, "cancelled", job.progress, "Job cancelled by user.")
    
    return {"status": "cancelled", "job_id": job_id}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "active_jobs": len(training_jobs)}

# This part is for running the app directly, not needed if using a production server like Gunicorn.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)