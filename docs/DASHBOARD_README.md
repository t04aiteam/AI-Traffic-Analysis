# Model Training Dashboard

A modern web-based dashboard for managing AI models and training processes.

> **Note**: The dashboard runs automatically with the webapp. No separate setup needed!

## Quick Start

1. Start the webapp server (from project root):
   ```bash
   cd webapp
   python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. Open the dashboard:
   ```
   http://localhost:8000/dashboard.html
   ```

That's it! The dashboard is ready to use. 🎉

## Features

### 📦 Model Library
- Browse all model weight files organized by category (vehicle, plate, OCR, tracking)
- View model details: name, size, format, modification date
- Download model files
- Delete unwanted models
- Search and filter models by category

### 🎯 Training Interface
- **Roboflow Integration**: Automatically download datasets from Roboflow
  - Provide API key, workspace, project, and version
  - Automatic dataset download and preparation
- **Local Dataset Support**: Use existing data.yaml files
- **Model Selection**: Choose from multiple YOLO architectures
  - YOLOv8 (nano, small, medium, large)
  - YOLOv9 (tiny, small)
  - YOLO11 (nano, small)
  - YOLO12 (nano)
- **Training Parameters**: Customize epochs, batch size, image size, device, workers

### 📊 Training Jobs
- View all training jobs with status badges
- Real-time progress tracking
- Live training logs via Server-Sent Events (SSE)
- Job management: cancel running jobs
- Download training logs
- View epoch progress and completion percentage

### 💻 System Information
- **GPU Information**: View GPU details, memory usage, utilization
- **Storage Information**: Monitor workspace, weights, and runs storage usage
- **System Metrics**: Real-time CPU and RAM monitoring
- **Stats Bar**: Quick view of system status at a glance

## Installation

The dashboard dependencies are included in `webapp/requirements.txt`. If not already installed:

```bash
cd webapp
pip install -r requirements.txt

# Optional: For Roboflow dataset integration
pip install roboflow
```

## Usage

### Downloading Models

1. Go to the **Model Library** tab
2. Use search or category filter to find models
3. Click the download icon on any model card
4. The model file will be downloaded to your computer

### Starting Training with Roboflow

1. Go to the **Training** tab
2. Enter a job name (e.g., "my_plate_detector")
3. Select a model architecture
4. Choose "Roboflow" dataset source
5. Enter your Roboflow credentials:
   - API Key: Get from [Roboflow Settings](https://app.roboflow.com/settings/api)
   - Workspace: Your workspace name
   - Project: Your project name
   - Version: Dataset version number
6. Adjust training parameters if needed
7. Click "Start Training"

### Starting Training with Local Dataset

1. Go to the **Training** tab
2. Enter a job name
3. Select a model architecture
4. Choose "Local data.yaml" dataset source
5. Enter the path to your data.yaml file (absolute or relative to project root)
6. Adjust training parameters
7. Click "Start Training"

### Monitoring Training

1. Go to the **Training Jobs** tab
2. Click on any job to open the details modal
3. View real-time progress, epochs, and logs
4. Cancel jobs if needed
5. Download logs for offline analysis

### System Monitoring

1. Go to the **System Info** tab
2. View detailed information about:
   - GPU capabilities and memory usage
   - Storage usage across workspace, weights, and training runs
   - CPU and RAM metrics
3. Click "Refresh" to update information

## API Endpoints

### Models
- `GET /api/dashboard/models` - List all models
- `GET /api/dashboard/models/download/{category}/{filename}` - Download model
- `DELETE /api/dashboard/models/{category}/{filename}` - Delete model

### Training
- `GET /api/dashboard/training/models` - List available model architectures
- `POST /api/dashboard/training/start` - Start training job
- `GET /api/dashboard/training/jobs` - List all jobs
- `GET /api/dashboard/training/jobs/{job_id}` - Get job details
- `DELETE /api/dashboard/training/jobs/{job_id}` - Cancel job
- `GET /api/dashboard/training/logs/{job_id}` - Stream job logs (SSE)

### System
- `GET /api/dashboard/system/gpu` - Get GPU information
- `GET /api/dashboard/system/storage` - Get storage information
- `GET /api/dashboard/system/metrics` - Get system metrics

## Training Output

Trained models are saved in:
```
runs/detect/{job_name}/weights/
├── best.pt       # Best model checkpoint
└── last.pt       # Last epoch checkpoint
```

## Troubleshooting

### GPU Not Detected
- Ensure PyTorch is installed with CUDA support
- Check CUDA drivers are properly installed
- Verify with: `python -c "import torch; print(torch.cuda.is_available())"`

### Roboflow Dataset Download Fails
- Verify API key is correct
- Check workspace, project, and version names
- Ensure you have access to the dataset
- Install roboflow SDK: `pip install roboflow`

### Training Fails to Start
- Check data.yaml path is correct
- Ensure model architecture name is valid
- Verify GPU device ID (use "cpu" for CPU training)
- Check logs in the training job modal

## Architecture

- **Backend**: FastAPI with async/await for efficient request handling
- **Frontend**: Vanilla JavaScript with modern ES6+ features
- **Real-time Updates**: Server-Sent Events (SSE) for log streaming
- **Training**: Subprocess management with output parsing
- **Storage**: Local file system for models and training runs

## Security Notes

- The dashboard is intended for local/trusted network use
- No authentication is implemented by default
- API keys are sent over HTTP (use HTTPS in production)
- File operations are restricted to the weights directory
- Path traversal attacks are prevented

## License

This dashboard is part of the TonAI License Plate Recognition system.
