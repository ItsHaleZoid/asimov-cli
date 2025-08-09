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
    dataset_subset: str
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
        self.model_name: str = request.model_name
        self.dataset_name: str = request.dataset_name

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

training_jobs: Dict[str, TrainingJob] = {}
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
        "distilgpt2": "distilgpt2",
    }
    
    job_config = {
        "docker_image": "halez/lora-finetuner:latest",
        "hf_token": "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",
        "base_model_id": model_mapping.get(request.model_id, request.model_id),
        "dataset_id": request.dataset_id,
        "dataset_subset": request.dataset_subset,
        "lora_model_repo": f"hamzafaisal/{request.model_id}-lora",
        "gpu_name": "H100 NVL",
        "num_gpus": 1,
        "max_train_steps": 100
    }
    
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
        
        await update_job_status(job_id, "initializing", 5, "Starting training script...")
        
        # ========================================================================
        # ## CHANGE THIS LINE ##
        # We now point the subprocess to the new async_main.py script inside the
        # 'async' directory. The '-u' flag is critical for unbuffered output.
        # Pass the job_id for WebSocket status updates.
        # ========================================================================
        script_path = os.path.join('async', 'async_main.py')
        
        process = await asyncio.create_subprocess_exec(
            'python', '-u', script_path, '--config', config_path, '--job-id', job_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # The cwd should remain the project root so that relative paths
            # like 'training/train.py' and 'jobs/...' work correctly.
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        job.process = process
        
        # Start monitoring both the process output and the status file
        await asyncio.gather(
            monitor_process_logs(job_id, process),
            monitor_job_status_file(job_id)
        )
        
    except Exception as e:
        logger.error(f"Error starting training job {job_id}: {str(e)}")
        await update_job_status(job_id, "failed", 0, f"Error: {str(e)}")

async def monitor_process_logs(job_id: str, process: asyncio.subprocess.Process):
    """Monitors the stdout of the subprocess and forwards every line as a log."""
    job = training_jobs.get(job_id)
    if not job:
        return

    async for line in process.stdout:
        if not line:
            break
        line_str = line.decode('utf-8').strip()
        if line_str:
            await add_log(job_id, line_str)
            logger.info(f"Job {job_id} Log: {line_str}")

            # Extract instance ID from logs if it appears
            instance_match = re.search(r'instance (\d+)', line_str, re.IGNORECASE)
            if instance_match:
                job.instance_id = instance_match.group(1)
            
            # Check for training completion signals with multiple patterns
            completion_signals = [
                "🎉 SUCCESS: INTELLIGENT TRAINING COMPLETED!",
                "🎉 INTELLIGENT TRAINING PIPELINE COMPLETED!",
                "✅ All training stages completed successfully",
                "✅ Model uploaded to HuggingFace Hub"
            ]
            
            success_exit_signals = [
                "✅ Exit code: 0 (Success)",
                "🔄 Return code: 0",
                "📊 Status: COMPLETED"
            ]
            
            # Check for any completion signal
            for signal in completion_signals:
                if signal in line_str:
                    await update_job_status(job_id, "completed", 100, "Training completed successfully!")
                    logger.info(f"Job {job_id}: Detected completion signal: {signal}")
                    break
            else:
                # Check for success exit codes
                for signal in success_exit_signals:
                    if signal in line_str:
                        await update_job_status(job_id, "completed", 100, "Training completed with exit code 0")
                        logger.info(f"Job {job_id}: Detected success exit signal: {signal}")
                        break
                else:
                    # Check for failure signals
                    if "❌ TRAINING FAILED!" in line_str:
                        await update_job_status(job_id, "failed", job.progress, "Training failed")
                        logger.error(f"Job {job_id}: Detected training failure signal")
                    elif "❌ Exit code: 1 (Failure)" in line_str:
                        await update_job_status(job_id, "failed", job.progress, "Training failed with exit code 1")
                        logger.error(f"Job {job_id}: Detected failure exit code")


    await process.wait()
    
    # Final check if the process exits
    if job.status not in ["completed", "failed", "cancelled"]:
        if process.returncode == 0:
            # Process completed successfully
            await update_job_status(job_id, "completed", 100, "Training completed successfully with exit code 0!")
            logger.info(f"Job {job_id}: Process completed with exit code 0")
        else:
            # Non-zero exit code indicates failure
            stderr_output = await process.stderr.read()
            stderr_text = stderr_output.decode('utf-8').strip()
            error_message = f"Process failed with exit code: {process.returncode}. Stderr: {stderr_text}"
            await update_job_status(job_id, "failed", job.progress, error_message)
            logger.error(f"Job {job_id}: Process failed with exit code {process.returncode}")
    else:
        logger.info(f"Job {job_id}: Final status already set to {job.status}, respecting explicit signals")


def parse_training_progress(line: str) -> Optional[int]:
    step_match = re.search(r'(\d+)/(\d+)\s*\[', line)
    if step_match:
        current, total = int(step_match.group(1)), int(step_match.group(2))
        # Map 60-95% range for training
        return 60 + int((current / total) * 35) if total > 0 else 60
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
                    "job": job.to_dict()
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
    return training_jobs[job_id].to_dict()

async def monitor_job_status_file(job_id: str):
    """Monitor the job status file for updates and send via WebSocket"""
    status_file = f"jobs/{job_id}_status.json"
    last_status = None
    
    while job_id in training_jobs and training_jobs[job_id].status not in ["completed", "failed", "cancelled"]:
        try:
            if os.path.exists(status_file):
                with open(status_file, 'r') as f:
                    status_data = json.load(f)
                
                # Only update if status has changed
                if status_data != last_status:
                    await update_job_status(
                        job_id, 
                        status_data['status'], 
                        status_data['progress'], 
                        status_data['message']
                    )
                    last_status = status_data.copy()
            
            await asyncio.sleep(1)  # Check every second
            
        except Exception as e:
            logger.warning(f"Error monitoring status file for job {job_id}: {e}")
            await asyncio.sleep(1)

@app.post("/api/job/{job_id}/cancel")
async def cancel_job(job_id: str):
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = training_jobs[job_id]
    
    if job.process and job.process.returncode is None:
        try:
            job.process.terminate()
            await job.process.wait()
            await add_log(job_id, "Training script process terminated by user.")
        except ProcessLookupError:
            await add_log(job_id, "Training script process already finished.")
    
    if job.instance_id:
        try:
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

@app.get("/api/job/{job_id}/logs")
async def get_training_logs(job_id: str):
    """Fetch training logs directly from the Vast.ai instance"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = training_jobs[job_id]
    if not job.instance_id:
        raise HTTPException(status_code=400, detail="No instance ID available for this job")
    
    try:
        # Get the training logs from the instance
        proc = await asyncio.create_subprocess_exec(
            'vastai', 'execute', str(job.instance_id), 'cat /app/training.log',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode == 0:
            logs = stdout.decode('utf-8')
            return {"job_id": job_id, "instance_id": job.instance_id, "logs": logs}
        else:
            error_msg = stderr.decode('utf-8')
            if "No such file or directory" in error_msg:
                return {"job_id": job_id, "instance_id": job.instance_id, "logs": "", "message": "Training log file not yet created. Training may still be initializing."}
            else:
                raise HTTPException(status_code=500, detail=f"Error fetching logs: {error_msg}")
                
    except Exception as e:
        logger.error(f"Error fetching training logs for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching logs: {str(e)}")

@app.get("/api/job/{job_id}/debug")
async def debug_job(job_id: str):
    """Debug information for a training job including instance files and status"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = training_jobs[job_id]
    debug_info = {
        "job_id": job_id,
        "job_status": job.status,
        "instance_id": job.instance_id,
        "files_check": {},
        "logs_check": {}
    }
    
    if not job.instance_id:
        debug_info["error"] = "No instance ID available"
        return debug_info
    
    try:
        # Check if key files exist
        files_to_check = ['/app/train.py', '/app/training.log', '/app/setup.log', '/app/lora_instructions.py']
        
        for file_path in files_to_check:
            try:
                proc = await asyncio.create_subprocess_exec(
                    'vastai', 'execute', str(job.instance_id), f'ls -la {file_path}',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                
                if proc.returncode == 0:
                    debug_info["files_check"][file_path] = {"exists": True, "details": stdout.decode('utf-8').strip()}
                else:
                    debug_info["files_check"][file_path] = {"exists": False, "error": stderr.decode('utf-8').strip()}
            except Exception as e:
                debug_info["files_check"][file_path] = {"exists": False, "error": str(e)}
        
        # Get last few lines of available log files
        log_files = ['/app/training.log', '/app/setup.log']
        for log_file in log_files:
            try:
                proc = await asyncio.create_subprocess_exec(
                    'vastai', 'execute', str(job.instance_id), f'tail -10 {log_file} 2>/dev/null || echo "File not found"',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                debug_info["logs_check"][log_file] = stdout.decode('utf-8').strip()
            except Exception as e:
                debug_info["logs_check"][log_file] = f"Error: {str(e)}"
                
    except Exception as e:
        debug_info["error"] = f"Error accessing instance: {str(e)}"
    
    return debug_info

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "active_jobs": len(training_jobs)}

if __name__ == "__main__":
    import uvicorn
    # Make sure to run this file from the project root directory
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
