"""
Dashboard API for model management and training.
"""
import asyncio
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import psutil

router = APIRouter(prefix="/api/dashboard")

# Paths
REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = REPO_ROOT / "weights"
TRAINING_SCRIPT = REPO_ROOT / "detectors" / "yolo" / "train_ultralytics.py"
RUNS_DIR = REPO_ROOT / "runs"

# Training state management
training_jobs: Dict[str, Dict] = {}


# Pydantic models
class RoboflowDataset(BaseModel):
    api_key: str
    workspace: str
    project: str
    version: int


class TrainingConfig(BaseModel):
    job_name: str
    model_type: str  # e.g., "yolov8n.yaml", "yolov9t.yaml", "yolo11n.pt"
    roboflow_dataset: RoboflowDataset
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    device: str = "0"
    workers: int = 8


# ====================== MODEL LIBRARY ======================

@router.get("/models")
async def list_models():
    """List all model weight files in the weights directory."""
    models = []
    
    if not WEIGHTS_DIR.exists():
        return {"models": [], "total": 0}
    
    for category_dir in WEIGHTS_DIR.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith(('.', '__')):
            continue
            
        category = category_dir.name
        
        for weight_file in category_dir.rglob("*"):
            if not weight_file.is_file():
                continue
            if weight_file.suffix.lower() not in {".pt", ".pth", ".onnx", ".engine"}:
                continue
                
            try:
                stat = weight_file.stat()
                rel_path = weight_file.relative_to(WEIGHTS_DIR)
                
                models.append({
                    "name": weight_file.stem,
                    "filename": weight_file.name,
                    "category": category,
                    "path": str(rel_path),
                    "full_path": str(weight_file),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "format": weight_file.suffix[1:].upper()
                })
            except Exception:
                continue
    
    models.sort(key=lambda x: (x["category"], x["name"]))
    return {"models": models, "total": len(models)}


@router.get("/models/download/{category}/{filename}")
async def download_model(category: str, filename: str):
    """Download a specific model weight file."""
    # Security: prevent path traversal
    if ".." in category or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid path")
    
    file_path = WEIGHTS_DIR / category / filename
    
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Ensure the file is within weights directory
    try:
        file_path.relative_to(WEIGHTS_DIR)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream"
    )


# Delete endpoint disabled for safety - users cannot delete models through UI
# @router.delete("/models/{category}/{filename}")
# async def delete_model(category: str, filename: str):
#     """Delete a model weight file."""
#     if ".." in category or ".." in filename:
#         raise HTTPException(status_code=400, detail="Invalid path")
#     
#     file_path = WEIGHTS_DIR / category / filename
#     
#     if not file_path.exists():
#         raise HTTPException(status_code=404, detail="Model not found")
#     
#     try:
#         file_path.relative_to(WEIGHTS_DIR)
#         file_path.unlink()
#         return {"message": f"Deleted {filename}"}
#     except ValueError:
#         raise HTTPException(status_code=403, detail="Access denied")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")


# ====================== TRAINING ======================

@router.get("/training/models")
async def list_available_models():
    """List available model architectures for training."""
    models = [
        {"name": "YOLOv8 Nano", "value": "yolov8n.yaml", "type": "detection"},
        {"name": "YOLOv8 Small", "value": "yolov8s.yaml", "type": "detection"},
        {"name": "YOLOv8 Medium", "value": "yolov8m.yaml", "type": "detection"},
        {"name": "YOLOv8 Large", "value": "yolov8l.yaml", "type": "detection"},
        {"name": "YOLOv9 Tiny", "value": "yolov9t.yaml", "type": "detection"},
        {"name": "YOLOv9 Small", "value": "yolov9s.yaml", "type": "detection"},
        {"name": "YOLO11 Nano", "value": "yolo11n.yaml", "type": "detection"},
        {"name": "YOLO11 Small", "value": "yolo11s.yaml", "type": "detection"},
        {"name": "YOLO12 Nano", "value": "yolo12n.yaml", "type": "detection"},
    ]
    return {"models": models}


def download_roboflow_dataset_sync(dataset: RoboflowDataset, output_dir: Path) -> Path:
    """Download dataset from Roboflow (synchronous)."""
    try:
        # Use roboflow Python SDK
        from roboflow import Roboflow
        
        rf = Roboflow(api_key=dataset.api_key)
        project = rf.workspace(dataset.workspace).project(dataset.project)
        version = project.version(dataset.version)
        
        dataset_path = version.download("yolov8", location=str(output_dir))
        return Path(dataset_path)
    except ImportError:
        raise Exception("Roboflow SDK not installed. Run: pip install roboflow")
    except Exception as e:
        raise Exception(f"Failed to download dataset: {str(e)}")


async def run_training_job(job_id: str, config: TrainingConfig):
    """Background task to run training."""
    try:
        training_jobs[job_id]["status"] = "preparing"
        training_jobs[job_id]["progress"] = 0
        training_jobs[job_id]["logs"] = []
        
        # Prepare dataset
        data_yaml_path = None
        training_jobs[job_id]["message"] = "Downloading Roboflow dataset..."
        dataset_dir = REPO_ROOT / "data" / f"roboflow_{job_id}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Run download in thread pool since it's blocking
        loop = asyncio.get_event_loop()
        dataset_path = await loop.run_in_executor(
            None,
            download_roboflow_dataset_sync,
            config.roboflow_dataset,
            dataset_dir
        )
        data_yaml_path = dataset_path / "data.yaml"
        
        # Prepare training command
        training_jobs[job_id]["status"] = "training"
        training_jobs[job_id]["message"] = "Training in progress..."
        
        # Use current Python executable
        import sys
        python_exe = sys.executable
        
        cmd = [
            python_exe,
            str(TRAINING_SCRIPT),
            "--model", config.model_type,
            "--data", str(data_yaml_path),
            "--epochs", str(config.epochs),
            "--batch", str(config.batch_size),
            "--imgsz", str(config.image_size),
            "--device", config.device,
            "--workers", str(config.workers),
            "--name", job_id,
            "--project", str(RUNS_DIR / "detect"),
            "--verbose"
        ]
        
        training_jobs[job_id]["command"] = " ".join(cmd)  # Store command for debugging
        
        # Run training
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(REPO_ROOT)
        )
        
        training_jobs[job_id]["pid"] = process.pid
        
        # Monitor output
        log_lines = []
        for line in iter(process.stdout.readline, ""):
            if not line:
                break
            log_lines.append(line.strip())
            
            # Parse progress from output (simplified)
            if "Epoch" in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.lower() == "epoch" and i + 1 < len(parts):
                            epoch_info = parts[i + 1].split('/')
                            if len(epoch_info) == 2:
                                current = int(epoch_info[0])
                                total = int(epoch_info[1].rstrip(':'))
                                progress = int((current / total) * 100)
                                training_jobs[job_id]["progress"] = progress
                                training_jobs[job_id]["epoch"] = current
                                training_jobs[job_id]["total_epochs"] = total
                except Exception:
                    pass
        
        process.wait()
        
        # Store logs (keep more for debugging)
        training_jobs[job_id]["logs"] = log_lines[-500:] if len(log_lines) > 500 else log_lines
        
        if process.returncode == 0:
            training_jobs[job_id]["status"] = "completed"
            training_jobs[job_id]["progress"] = 100
            training_jobs[job_id]["message"] = "Training completed successfully"
            
            # Find best weights
            run_dir = RUNS_DIR / "detect" / job_id
            weights_dir = run_dir / "weights"
            if weights_dir.exists():
                best_pt = weights_dir / "best.pt"
                if best_pt.exists():
                    training_jobs[job_id]["best_weights"] = str(best_pt)
        else:
            training_jobs[job_id]["status"] = "failed"
            error_msg = f"Training failed with exit code {process.returncode}"
            if log_lines:
                # Get last few lines for error context
                error_msg += f"\nLast output: {' '.join(log_lines[-3:])}"
            training_jobs[job_id]["message"] = error_msg
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["message"] = f"Error: {str(e)}"
        training_jobs[job_id]["error"] = str(e)
        training_jobs[job_id]["error_trace"] = error_trace


@router.post("/training/start")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Start a new training job."""
    # Generate job ID
    job_id = f"{config.job_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize job state
    training_jobs[job_id] = {
        "id": job_id,
        "name": config.job_name,
        "status": "queued",
        "progress": 0,
        "message": "Job queued",
        "config": config.dict(),
        "started_at": datetime.now().isoformat(),
        "epoch": 0,
        "total_epochs": config.epochs
    }
    
    # Start training in background
    background_tasks.add_task(run_training_job, job_id, config)
    
    return {"job_id": job_id, "message": "Training started"}


@router.get("/training/jobs")
async def list_training_jobs():
    """List all training jobs."""
    jobs = list(training_jobs.values())
    jobs.sort(key=lambda x: x.get("started_at", ""), reverse=True)
    return {"jobs": jobs, "total": len(jobs)}


@router.get("/training/jobs/{job_id}")
async def get_training_job(job_id: str):
    """Get details of a specific training job."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return training_jobs[job_id]


@router.delete("/training/jobs/{job_id}")
async def cancel_training_job(job_id: str):
    """Cancel a running training job."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = training_jobs[job_id]
    
    if job["status"] in ["completed", "failed", "cancelled"]:
        return {"message": "Job already finished"}
    
    # Kill process if running
    if "pid" in job:
        try:
            import signal
            os.kill(job["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
    
    job["status"] = "cancelled"
    job["message"] = "Job cancelled by user"
    
    return {"message": "Job cancelled"}


@router.get("/training/logs/{job_id}")
async def stream_training_logs(job_id: str):
    """Stream training logs in real-time."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    async def log_generator():
        last_log_count = 0
        while True:
            job = training_jobs.get(job_id)
            if not job:
                break
            
            logs = job.get("logs", [])
            new_logs = logs[last_log_count:]
            
            if new_logs:
                for log_line in new_logs:
                    yield f"data: {json.dumps({'line': log_line})}\n\n"
                last_log_count = len(logs)
            
            if job["status"] in ["completed", "failed", "cancelled"]:
                yield f"data: {json.dumps({'status': job['status']})}\n\n"
                break
            
            await asyncio.sleep(1)
    
    return StreamingResponse(
        log_generator(),
        media_type="text/event-stream"
    )


# ====================== SYSTEM INFO ======================

@router.get("/system/gpu")
async def get_gpu_info():
    """Get GPU information."""
    try:
        import torch
        
        gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                
                gpus.append({
                    "id": i,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "allocated_memory_gb": round(memory_allocated / (1024**3), 2),
                    "reserved_memory_gb": round(memory_reserved / (1024**3), 2),
                    "free_memory_gb": round((props.total_memory - memory_reserved) / (1024**3), 2),
                    "utilization": round((memory_reserved / props.total_memory) * 100, 1)
                })
        
        return {
            "available": torch.cuda.is_available(),
            "device_count": len(gpus),
            "gpus": gpus,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
        }
    except ImportError:
        return {
            "available": False,
            "device_count": 0,
            "gpus": [],
            "error": "PyTorch not installed"
        }


@router.get("/system/storage")
async def get_storage_info():
    """Get storage information."""
    storage_info = []
    
    # Main workspace storage
    total, used, free = shutil.disk_usage(REPO_ROOT)
    storage_info.append({
        "location": "Workspace",
        "path": str(REPO_ROOT),
        "total_gb": round(total / (1024**3), 2),
        "used_gb": round(used / (1024**3), 2),
        "free_gb": round(free / (1024**3), 2),
        "usage_percent": round((used / total) * 100, 1)
    })
    
    # Weights directory size
    weights_size = 0
    if WEIGHTS_DIR.exists():
        for file in WEIGHTS_DIR.rglob("*"):
            if file.is_file():
                try:
                    weights_size += file.stat().st_size
                except Exception:
                    pass
    
    storage_info.append({
        "location": "Model Weights",
        "path": str(WEIGHTS_DIR),
        "size_gb": round(weights_size / (1024**3), 2)
    })
    
    # Training runs size
    runs_size = 0
    if RUNS_DIR.exists():
        for file in RUNS_DIR.rglob("*"):
            if file.is_file():
                try:
                    runs_size += file.stat().st_size
                except Exception:
                    pass
    
    storage_info.append({
        "location": "Training Runs",
        "path": str(RUNS_DIR),
        "size_gb": round(runs_size / (1024**3), 2)
    })
    
    return {"storage": storage_info}


@router.get("/system/metrics")
async def get_system_metrics():
    """Get system metrics (CPU, RAM, etc)."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    return {
        "cpu": {
            "usage_percent": cpu_percent,
            "count": psutil.cpu_count(),
            "count_physical": psutil.cpu_count(logical=False)
        },
        "memory": {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "usage_percent": memory.percent
        }
    }
