from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.websockets import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, validator
import asyncio
import subprocess
import yaml
import uuid
import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
import logging
from supabase import create_client, Client
from cryptography.fernet import Fernet
import base64
import hashlib
from collections import defaultdict
import time
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError, GatedRepoError, RepositoryNotFoundError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = "https://zxebusnnyzvaktqpmuft.supabase.co"
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp4ZWJ1c25ueXp2YWt0cXBtdWZ0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM0MzcxNDksImV4cCI6MjA2OTAxMzE0OX0.5EncAJLTx4J5biezoXGiTkANi60iMBmioVrU1qdvuBo")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Encryption key for HF tokens (in production, use environment variable)
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", base64.urlsafe_b64encode(hashlib.sha256(b"asimov-secret-key").digest()))
cipher_suite = Fernet(ENCRYPTION_KEY)

def encrypt_token(token: str) -> str:
    """Encrypt HuggingFace token for secure storage"""
    return cipher_suite.encrypt(token.encode()).decode()

def decrypt_token(encrypted_token: str) -> str:
    """Decrypt HuggingFace token for use"""
    return cipher_suite.decrypt(encrypted_token.encode()).decode()

async def verify_user_token(authorization: str = Header(None), request: Request = None) -> dict:
    """Verify user authentication token and return user info"""
    client_ip = request.client.host if request else "unknown"
    
    if not authorization or not authorization.startswith("Bearer "):
        logger.warning(f"Missing or invalid authorization header from {client_ip}")
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = authorization.split(" ")[1]
    
    # Basic token format validation
    if len(token) < 10 or len(token) > 2000:
        logger.warning(f"Invalid token length from {client_ip}")
        raise HTTPException(status_code=401, detail="Invalid token format")
    
    try:
        # Verify the token with Supabase
        response = supabase.auth.get_user(token)
        if response.user:
            logger.info(f"Successful authentication for user {response.user.id} from {client_ip}")
            return {"user_id": response.user.id, "email": response.user.email}
        else:
            logger.warning(f"Invalid token from {client_ip}")
            raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        logger.error(f"Token verification failed from {client_ip}: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_user_hf_data(user_id: str) -> tuple[str, str]:
    """Get user's encrypted HuggingFace token and username from database"""
    try:
        response = supabase.table("user_tokens").select("encrypted_hf_token, huggingface_username").eq("user_id", user_id).single().execute()
        if response.data:
            token = decrypt_token(response.data["encrypted_hf_token"])
            username = response.data["huggingface_username"]
            if not username:
                raise HTTPException(status_code=400, detail="No HuggingFace username found. Please update your settings.")
            return token, username
        else:
            raise HTTPException(status_code=400, detail="No HuggingFace token found. Please set your token in settings first.")
    except Exception as e:
        logger.error(f"Error fetching user HF data: {e}")
        raise HTTPException(status_code=500, detail="Error fetching HuggingFace data")

async def check_user_quota(user_id: str) -> dict:
    """Check if user can start new jobs based on quotas"""
    try:
        # Check if user quota exists
        response = supabase.table("user_quotas").select("*").eq("user_id", user_id).execute()
        
        if response.data and len(response.data) > 0:
            quota = response.data[0]
            # Reset daily count if new day
            if quota["last_reset_date"] != datetime.now().date().isoformat():
                supabase.table("user_quotas").update({
                    "jobs_today": 0,
                    "last_reset_date": datetime.now().date().isoformat()
                }).eq("user_id", user_id).execute()
                quota["jobs_today"] = 0
            
            return quota
        else:
            # Create default quota for new user (single job per user)
            new_quota = {
                "user_id": user_id,
                "max_concurrent_jobs": 1,
                "max_daily_jobs": 10,
                "jobs_today": 0,
                "current_jobs": 0,
                "last_reset_date": datetime.now().date().isoformat()
            }
            result = supabase.table("user_quotas").insert(new_quota).execute()
            return new_quota
    except Exception as e:
        logger.error(f"Error checking user quota: {e}")
        raise HTTPException(status_code=500, detail="Error checking user quota")

# Rate limiting storage
rate_limit_storage = defaultdict(list)

# Rate limiting middleware
class RateLimitMiddleware:
    def __init__(self, app, calls: int = 100, period: int = 60):
        self.app = app
        self.calls = calls
        self.period = period

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope)
            client_ip = request.client.host
            current_time = time.time()
            
            # Clean old entries
            cutoff_time = current_time - self.period
            rate_limit_storage[client_ip] = [
                timestamp for timestamp in rate_limit_storage[client_ip] 
                if timestamp > cutoff_time
            ]
            
            # Check rate limit
            if len(rate_limit_storage[client_ip]) >= self.calls:
                response = {
                    "detail": "Rate limit exceeded. Please try again later.",
                    "status_code": 429
                }
                await send({
                    "type": "http.response.start",
                    "status": 429,
                    "headers": [[b"content-type", b"application/json"]],
                })
                await send({
                    "type": "http.response.body",
                    "body": json.dumps(response).encode(),
                })
                return
            
            # Add current request
            rate_limit_storage[client_ip].append(current_time)
        
        await self.app(scope, receive, send)

app = FastAPI(
    title="Asimov Training API",
    description="Multi-user AI model training API with security",
    version="1.0.0"
)

# Add security middleware
app.add_middleware(RateLimitMiddleware, calls=100, period=60)  # 100 requests per minute
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "*.asimov.ai"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "ws://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # More restrictive
    allow_headers=["Authorization", "Content-Type"],  # More restrictive
)

# Add startup event to clean up stale jobs
@app.on_event("startup")
async def startup_event():
    """Run cleanup tasks when server starts"""
    logger.info("🚀 Server starting up...")
    await startup_cleanup()
    logger.info("✅ Server startup complete")

class TrainingRequest(BaseModel):
    model_id: str
    dataset_id: str
    dataset_name: str
    model_name: str
    dataset_subset: str
    
    @validator('model_id')
    def validate_model_id(cls, v):
        allowed_models = [
            "mistral-instruct-7b", "mistral-small-24b", "mistral-codestral-22b", 
            "mistral-devstral-22b", "gemma-3n-2b", "gemma-3n-4b", "distilgpt2", "gpt2", "google/gemma-3-1b-it", 
            "google/gemma-3-4b-it", "google/gemma-3-12b-it", "google/gemma-3-27b-it", "mistralai/Mistral-7B-Instruct-v0.3",
            "mistralai/Mistral-Small-Instruct-2409", "mistralai/Mistral-Large-Instruct-2407", "openai/gpt-oss-120b", "openai/gpt-oss-20b",  
        ]
        if v not in allowed_models:
            raise ValueError(f"Invalid model_id. Must be one of: {allowed_models}")
        return v
    
    @validator('dataset_id')
    def validate_dataset_id(cls, v):
        # Basic validation for HuggingFace dataset format
        if not re.match(r'^[a-zA-Z0-9_-]+(/[a-zA-Z0-9_-]+)?$', v):
            raise ValueError("Invalid dataset_id format")
        if len(v) > 200:  # Reasonable length limit
            raise ValueError("dataset_id too long")
        return v
    
    @validator('dataset_name', 'model_name')
    def validate_names(cls, v):
        if len(v) > 200:
            raise ValueError("Name too long")
        # Prevent XSS and injection attacks
        if re.search(r'[<>"\']', v):
            raise ValueError("Invalid characters in name")
        return v
    
    @validator('dataset_subset')
    def validate_subset(cls, v):
        if len(v) > 100:
            raise ValueError("dataset_subset too long")
        if re.search(r'[<>"\']', v):
            raise ValueError("Invalid characters in dataset_subset")
        return v

class TokenRequest(BaseModel):
    token: str
    huggingface_username: str
    
    @validator('token')
    def validate_token(cls, v):
        v = v.strip()
        if not v.startswith('hf_'):
            raise ValueError("Invalid HuggingFace token format")
        if len(v) < 20 or len(v) > 200:
            raise ValueError("Invalid token length")
        if not re.match(r'^hf_[a-zA-Z0-9_]+$', v):
            raise ValueError("Invalid token format")
        return v
    
    @validator('huggingface_username')
    def validate_username(cls, v):
        v = v.strip()
        if len(v) < 2 or len(v) > 39:
            raise ValueError("Username must be 2-39 characters long")
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Username must contain only alphanumeric characters, hyphens, or underscores")
        return v

class TrainingJob:
    def __init__(self, job_id: str, request: TrainingRequest, config: dict, user_id: str, user_email: str):
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
        self.user_id: str = user_id
        self.user_email: str = user_email

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
            "instance_id": self.instance_id,
            "user_id": self.user_id,
            "user_email": self.user_email
        }

training_jobs: Dict[str, TrainingJob] = {}
websocket_connections: Dict[str, WebSocket] = {}

async def cleanup_stale_jobs():
    """Clean up jobs that are stuck in active states from previous server sessions"""
    try:
        logger.info("🧹 Starting cleanup of stale jobs...")
        
        # Find jobs that are in active states but likely stale
        active_statuses = ['initializing', 'preparing', 'searching_gpu', 'found_gpu', 'creating_instance', 
                          'instance_ready', 'uploading_script', 'loading_model', 'loading_dataset', 
                          'training', 'saving', 'uploading_model']
        
        # Get all jobs with active statuses
        response = supabase.table("user_jobs").select("*").in_("status", active_statuses).execute()
        stale_jobs = response.data or []
        
        if not stale_jobs:
            logger.info("✅ No stale jobs found")
            return
        
        # Current time
        current_time = datetime.now()
        cleanup_count = 0
        
        for job in stale_jobs:
            job_id = job["job_id"]
            created_at = datetime.fromisoformat(job["created_at"].replace('Z', '+00:00'))
            
            # Consider jobs older than 1 hour as potentially stale
            hours_old = (current_time - created_at.replace(tzinfo=None)).total_seconds() / 3600
            
            if hours_old > 1:  # Job is more than 1 hour old
                logger.info(f"🚨 Marking stale job {job_id} as failed (running for {hours_old:.1f} hours)")
                
                # Update job status to failed
                supabase.table("user_jobs").update({
                    "status": "failed",
                    "completed_at": current_time.isoformat(),
                    "error_message": f"Job marked as failed due to server restart (was stuck in '{job['status']}' for {hours_old:.1f} hours)",
                    "updated_at": current_time.isoformat()
                }).eq("job_id", job_id).execute()
                
                # Add to job history
                supabase.table("job_history").insert({
                    "user_id": job["user_id"],
                    "job_id": job_id,
                    "action": "failed",
                    "details": {"message": "Marked as failed during server cleanup", "reason": "stale_job_cleanup"}
                }).execute()
                
                cleanup_count += 1
        
        logger.info(f"✅ Cleanup completed: {cleanup_count} stale jobs marked as failed")
        
    except Exception as e:
        logger.error(f"❌ Error during stale job cleanup: {e}")

# Run cleanup on startup
async def startup_cleanup():
    """Run cleanup tasks when server starts"""
    await cleanup_stale_jobs()

@app.post("/api/start-training")
async def start_training(
    request: TrainingRequest, 
    background_tasks: BackgroundTasks,
    user: dict = Depends(verify_user_token)
):
    user_id = user["user_id"]
    user_email = user["email"]
    
    # Check if user already has a running job (enforce single job per user)
    quota = await check_user_quota(user_id)
    if quota["current_jobs"] >= 1:
        raise HTTPException(
            status_code=429, 
            detail="You already have a running job. Please wait for it to complete before starting a new one."
        )
    
    if quota["jobs_today"] >= quota["max_daily_jobs"]:
        raise HTTPException(
            status_code=429, 
            detail=f"Daily job limit reached ({quota['max_daily_jobs']}). Please try again tomorrow."
        )
    
    # Get user's HuggingFace token and username
    user_hf_token, user_hf_username = await get_user_hf_data(user_id)

    # Pre-flight: verify the HF token has access to the requested model to avoid 401 during training
    try:
        api = HfApi(token=user_hf_token)
        api.model_info(request.model_id, token=user_hf_token)
    except GatedRepoError:
        raise HTTPException(
            status_code=401,
            detail=(
                f"Your Hugging Face token does not have access to the gated model '{request.model_id}'. "
                "Please request access on the model page and ensure your token has the required permissions."
            ),
        )
    except RepositoryNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"The model '{request.model_id}' was not found on the Hugging Face Hub. "
                "Please verify the model ID."
            ),
        )
    except HfHubHTTPError:
        raise HTTPException(
            status_code=401,
            detail=(
                f"Invalid Hugging Face credentials for model '{request.model_id}'. "
                "Update your token in settings and try again."
            ),
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Failed to verify access to model '{request.model_id}': {str(e)}"
            ),
        )
    
    job_id = str(uuid.uuid4())
   
    
    job_config = {
        "docker_image": "halez/lora-finetuner:latest",
        "hf_token": user_hf_token,
        "base_model_id": request.model_id,
        "dataset_id": request.dataset_id,
        "dataset_subset": request.dataset_subset,
        "lora_model_repo": f"{user_hf_username}/{request.model_id}-lora",
        "gpu_name": "H200 NVL",
        "num_gpus": 1,
        "max_train_steps": 100
    }
    
    job = TrainingJob(job_id, request, job_config, user_id, user_email)
    training_jobs[job_id] = job
    
    # Create job record in database
    try:
        supabase.table("user_jobs").insert({
            "job_id": job_id,
            "user_id": user_id,
            "job_type": "training",
            "status": "initializing",
            "model_name": request.model_name,
            "dataset_name": request.dataset_name,
            "job_params": {
                "model_id": request.model_id,
                "dataset_id": request.dataset_id,
                "dataset_subset": request.dataset_subset
            }
        }).execute()
        
        # Add to job history
        supabase.table("job_history").insert({
            "user_id": user_id,
            "job_id": job_id,
            "action": "created",
            "details": {"model_name": request.model_name, "dataset_name": request.dataset_name}
        }).execute()
        
    except Exception as e:
        logger.error(f"Error creating job record: {e}")
        # Continue anyway, as the job can still run
    
    background_tasks.add_task(run_training_job, job_id)
    
    return {"job_id": job_id, "status": "started", "user_id": user_id}

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
        old_status = job.status
        job.status = status
        job.progress = progress
        log_entry = {"timestamp": datetime.now().isoformat(), "message": message, "type": "status"}
        job.logs.append(log_entry)
        
        # Update database record
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.now().isoformat()
            }
            
            if status in ["completed", "failed", "cancelled"]:
                update_data["completed_at"] = datetime.now().isoformat()
                if status == "failed":
                    update_data["error_message"] = message
            
            supabase.table("user_jobs").update(update_data).eq("job_id", job_id).execute()
            
            # Add to job history if status changed
            if old_status != status:
                supabase.table("job_history").insert({
                    "user_id": job.user_id,
                    "job_id": job_id,
                    "action": status,
                    "details": {"message": message, "progress": progress}
                }).execute()
                
        except Exception as e:
            logger.error(f"Error updating job status in database: {e}")
        
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
async def websocket_endpoint(websocket: WebSocket, job_id: str, token: str = None):
    await websocket.accept()
    
    # Verify user has access to this job
    if job_id in training_jobs:
        job = training_jobs[job_id]
        # For now, allow access. In production, you might want to verify the token
        # and check if the user owns this job
        websocket_connections[job_id] = websocket
        
        try:
            await websocket.send_json({
                "type": "initial_state",
                "job": job.to_dict()
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
    else:
        await websocket.close(code=4004, reason="Job not found")

@app.get("/api/jobs")
async def get_user_jobs(user: dict = Depends(verify_user_token)):
    """Get all jobs for the authenticated user"""
    user_id = user["user_id"]
    
    # Get jobs from memory (active jobs)
    active_jobs = []
    for job in training_jobs.values():
        if job.user_id == user_id:
            active_jobs.append(job.to_dict())
    
    # Get jobs from database (including completed/failed jobs)
    try:
        db_response = supabase.table("user_jobs").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(50).execute()
        db_jobs = db_response.data or []
        
        # Combine active jobs with database jobs, avoiding duplicates
        active_job_ids = {job["id"] for job in active_jobs}
        all_jobs = active_jobs.copy()
        
        for db_job in db_jobs:
            if db_job["job_id"] not in active_job_ids:
                all_jobs.append({
                    "id": db_job["job_id"],
                    "status": db_job["status"],
                    "model_name": db_job["model_name"],
                    "dataset_name": db_job["dataset_name"],
                    "start_time": db_job["created_at"],
                    "completed_at": db_job.get("completed_at"),
                    "error_message": db_job.get("error_message"),
                    "user_id": db_job["user_id"],
                    "job_params": db_job.get("job_params", {})
                })
        
        return {"jobs": all_jobs, "total": len(all_jobs)}
        
    except Exception as e:
        logger.error(f"Error fetching user jobs: {e}")
        return {"jobs": active_jobs, "total": len(active_jobs)}

@app.get("/api/user/quota")
async def get_user_quota(user: dict = Depends(verify_user_token)):
    """Get user's current quota status"""
    user_id = user["user_id"]
    quota = await check_user_quota(user_id)
    return quota

@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str, user: dict = Depends(verify_user_token)):
    # First check if job is in memory (active job)
    if job_id in training_jobs:
        job = training_jobs[job_id]
        if job.user_id != user["user_id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        return job.to_dict()
    
    # If not in memory, check database for completed/failed jobs
    try:
        response = supabase.table("user_jobs").select("*").eq("job_id", job_id).single().execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        db_job = response.data
        
        # Verify user owns this job
        if db_job["user_id"] != user["user_id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Convert database job to the expected format
        job_data = {
            "id": db_job["job_id"],
            "status": db_job["status"],
            "progress": 100 if db_job["status"] in ["completed", "failed", "cancelled"] else 0,
            "logs": [],  # Database doesn't store detailed logs
            "model_name": db_job["model_name"],
            "dataset_name": db_job["dataset_name"], 
            "start_time": db_job["created_at"],
            "completed_at": db_job.get("completed_at"),
            "error_message": db_job.get("error_message"),
            "user_id": db_job["user_id"],
            "job_params": db_job.get("job_params", {}),
            "config": {},  # Not stored in database
            "instance_id": None  # Not available for completed jobs
        }
        
        return job_data
        
    except Exception as e:
        logger.error(f"Error fetching job {job_id} from database: {e}")
        raise HTTPException(status_code=404, detail="Job not found")

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
async def cancel_job(job_id: str, user: dict = Depends(verify_user_token)):
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = training_jobs[job_id]
    if job.user_id != user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
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

@app.post("/api/jobs/cancel-all")
async def cancel_all_jobs(user: dict = Depends(verify_user_token)):
    """Cancel all active jobs for the authenticated user"""
    user_id = user["user_id"]
    cancelled_jobs = []
    failed_cancellations = []
    
    # Get all active jobs for the user
    user_jobs = [job for job in training_jobs.values() if job.user_id == user_id]
    
    if not user_jobs:
        return {"message": "No active jobs found", "cancelled_jobs": [], "failed_cancellations": []}
    
    for job in user_jobs:
        try:
            job_id = job.job_id
            
            # Terminate the process if it's running
            if job.process and job.process.returncode is None:
                try:
                    job.process.terminate()
                    await job.process.wait()
                    await add_log(job_id, "Training script process terminated by user (bulk cancel).")
                except ProcessLookupError:
                    await add_log(job_id, "Training script process already finished.")
            
            # Destroy the Vast.ai instance if it exists
            if job.instance_id:
                try:
                    proc = await asyncio.create_subprocess_exec(
                        'vastai', 'destroy', 'instance', str(job.instance_id)
                    )
                    await proc.wait()
                    await add_log(job_id, f"Destroy command sent for Vast.ai instance {job.instance_id} (bulk cancel)")
                except Exception as e:
                    logger.error(f"Error destroying instance {job.instance_id}: {e}")
                    await add_log(job_id, f"Error destroying instance {job.instance_id}: {e}")
            
            # Update job status to cancelled
            await update_job_status(job_id, "cancelled", job.progress, "Job cancelled by user (bulk cancel).")
            cancelled_jobs.append({"job_id": job_id, "status": "cancelled"})
            
        except Exception as e:
            logger.error(f"Error cancelling job {job.job_id}: {e}")
            failed_cancellations.append({"job_id": job.job_id, "error": str(e)})
    
    return {
        "message": f"Cancelled {len(cancelled_jobs)} jobs",
        "cancelled_jobs": cancelled_jobs,
        "failed_cancellations": failed_cancellations
    }

@app.get("/api/job/{job_id}/logs")
async def get_training_logs(job_id: str, user: dict = Depends(verify_user_token)):
    """Fetch training logs directly from the Vast.ai instance"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = training_jobs[job_id]
    if job.user_id != user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
        
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
async def debug_job(job_id: str, user: dict = Depends(verify_user_token)):
    """Debug information for a training job including instance files and status"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = training_jobs[job_id]
    if job.user_id != user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
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

@app.post("/api/user/hf-token")
async def save_hf_token(token_request: TokenRequest, user: dict = Depends(verify_user_token)):
    """Save user's HuggingFace token and username"""
    user_id = user["user_id"]
    hf_token = token_request.token
    hf_username = token_request.huggingface_username
    
    try:
        encrypted_token = encrypt_token(hf_token)
        
        # Insert or update token
        response = supabase.table("user_tokens").select("id").eq("user_id", user_id).execute()
        
        if response.data:
            # Update existing token
            supabase.table("user_tokens").update({
                "encrypted_hf_token": encrypted_token,
                "huggingface_username": hf_username,
                "updated_at": datetime.now().isoformat()
            }).eq("user_id", user_id).execute()
        else:
            # Insert new token
            supabase.table("user_tokens").insert({
                "user_id": user_id,
                "encrypted_hf_token": encrypted_token,
                "huggingface_username": hf_username
            }).execute()
        
        logger.info(f"HuggingFace token saved for user {user_id}")
        return {"message": "HuggingFace token saved successfully"}
        
    except Exception as e:
        logger.error(f"Error saving HF token for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error saving HuggingFace token")

@app.get("/api/user/hf-token/status")
async def get_hf_token_status(user: dict = Depends(verify_user_token)):
    """Check if user has a HuggingFace token saved"""
    user_id = user["user_id"]
    
    try:
        response = supabase.table("user_tokens").select("created_at, updated_at, huggingface_username").eq("user_id", user_id).single().execute()
        if response.data:
            return {
                "has_token": True,
                "created_at": response.data["created_at"],
                "updated_at": response.data["updated_at"],
                "huggingface_username": response.data["huggingface_username"]
            }
        else:
            return {"has_token": False}
    except Exception as e:
        logger.error(f"Error checking HF token status: {e}")
        return {"has_token": False}

@app.delete("/api/user/hf-token")
async def delete_hf_token(user: dict = Depends(verify_user_token)):
    """Delete user's HuggingFace token"""
    user_id = user["user_id"]
    
    try:
        supabase.table("user_tokens").delete().eq("user_id", user_id).execute()
        return {"message": "HuggingFace token deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting HF token: {e}")
        raise HTTPException(status_code=500, detail="Error deleting HuggingFace token")

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy", 
        "active_jobs": len(training_jobs),
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    
    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    return response

# Request size limiting
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    max_size = 10 * 1024 * 1024  # 10MB
    content_length = request.headers.get("content-length")
    
    if content_length and int(content_length) > max_size:
        return HTTPException(status_code=413, detail="Request too large")
    
    return await call_next(request)

if __name__ == "__main__":
    import uvicorn
    # Make sure to run this file from the project root directory
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
